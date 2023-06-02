#!/usr/bin/env python

# testing for action collision
# J. Hwang
# CS 81
import numpy as np
import tf
import rospy # module for ROS APIs
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Twist # message type for cmd_vel
from sensor_msgs.msg import LaserScan # message type for scan

from pid import PD 
from grid import Grid, fsm
import constants as cs
from collections import deque

class ActionRobots():
    def __init__(self,
                 controller,
                 goal_distance=cs.GOAL_DISTANCE,
                 linear_velocity=cs.LINEAR_VELOCITY,
                 angular_velocity=cs.ANGULAR_VELOCITY,
                 frequency=cs.FREQUENCY,
                 scan_angle=[cs.MIN_SCAN_ANGLE_RAD, cs.MAX_SCAN_ANGLE_RAD],
                 scan_angle_front= [cs.MIN_SCAN_ANGLE_RAD_FRONT, cs.MAX_SCAN_ANGLE_RAD_FRONT],
                 map_width=cs.MAP_WIDTH,
                 map_height=cs.MAP_HEIGHT,
                 map_resolution=cs.MAP_RESOLUTION):

        # Publisher/Subscriber
        self._cmd_pub_1 = rospy.Publisher(cs.DEFAULT_CMD_VEL_TOPIC_1, Twist, queue_size=1) # publisher to send velocity commands.
        self._cmd_pub_2 = rospy.Publisher(cs.DEFAULT_CMD_VEL_TOPIC_2, Twist, queue_size=1) # publisher to send velocity commands.

        # self._occupancy_grid_pub = rospy.Subscriber(cs.MAP_TOPIC, OccupancyGrid, queue_size=1) # subscribes to occupancy grid info
        self._laser_sub_1 = rospy.Subscriber(cs.DEFAULT_SCAN_TOPIC_1, LaserScan, self._laser_callback_1, queue_size=1) # subscriber receiving messages from the laser.
        self._laser_sub_2 = rospy.Subscriber(cs.DEFAULT_SCAN_TOPIC_2, LaserScan, self._laser_callback_2, queue_size=1) # subscriber receiving messages from the laser.

        self._odom_sub_1 = rospy.Subscriber(cs.DEFAULT_ODOM_1, Odometry, self._odom_callback_1, queue_size=1)
        self._odom_sub_2 = rospy.Subscriber(cs.DEFAULT_ODOM_2, Odometry, self._odom_callback_2, queue_size=1)

        # Parameters
        self.linear_velocity_1 = linear_velocity
        self.angular_velocity_1 = angular_velocity
        self.frequency = frequency
        self.scan_angle_1 = scan_angle
        self.goal_distance_1 = goal_distance
        self.duration = 1000

        self.linear_velocity_2 = linear_velocity
        self.angular_velocity_2 = angular_velocity

        self.scan_angle_2 = scan_angle
        self.goal_distance_2 = goal_distance

        # Robot's state and controller
        self._fsm_1 = fsm.WAITING_FOR_LASER
        self.controller_1 = controller 

        self._fsm_2 = fsm.WAITING_FOR_LASER
        self.controller_2 = controller 

        # Previous and current errors (distance to the wall)
        self.prev_err_1 = None
        self.curr_err_1 = None
        self.prev_err_2 = None
        self.curr_err_2 = None

        # Keep track of the time when prev and curr errors were computed (used to get d_t for the d term in PD controller)
        self.prev_err_timestamp_1 = rospy.get_rostime()
        self.curr_err_timestamp_1 = rospy.get_rostime()
        self.prev_err_timestamp_2 = rospy.get_rostime()
        self.curr_err_timestamp_2 = rospy.get_rostime()    

        # Odom
        self.x_1 = 0
        self.y_1 = 0
        self.yaw_1 = 0

        self.x_2 = 0
        self.y_2 = 0
        self.yaw_2 = 0

        # Map
        self.map_width = map_width
        self.map_height = map_height
        self.map_resolution = map_resolution
        self.map = Grid(map_width, map_height, map_resolution)

        self.listener_1 = tf.TransformListener()
        self.listener_2 = tf.TransformListener()
        self.laser_msg_1 = None # update laser msg at callback
        self.laser_msg_2 = None # update laser msg at callback


        # - Sample Input for action collision testing
        self.path_1 = deque()
        self.path_2 = deque()
        self.object_locations_1 = []
        self.object_locations_2 = []
        self.avoid_pattern_1 = False
        self.avoid_pattern_2 = False
        self.scan_angle_front_1 = scan_angle_front
        self.scan_angle_front_2 = scan_angle_front
        self.curr_front_distance_1 = 0
        self.curr_front_distance_2 = 0
        self.prev_distance_front_1 = 0
        self.prev_distance_front_2 = 0
        self.prev_time_1 = rospy.get_time()
        self.prev_time_2 = rospy.get_time()
        self.start_time = rospy.get_rostime()
        self.stop_distance = 1
        self.temp_curr_front_distance_1 = 0
    '''Processing of odom message'''
    def _odom_callback_1(self, msg):
        position = msg.pose.pose.position
        self.x_1 = position.x
        self.y_1 = position.y

        orientation = msg.pose.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        self.yaw_1 = tf.transformations.euler_from_quaternion(quaternion)[2]

    def _odom_callback_2(self, msg):
        position = msg.pose.pose.position
        self.x_2 = position.x
        self.y_2 = position.y

        orientation = msg.pose.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        self.yaw_2 = tf.transformations.euler_from_quaternion(quaternion)[2]

    '''
    Process laser message.
    Process only when the robot's state is WAITING_FOR_LASER
    Update previous and current errors and their timestamps
    '''
    def _laser_callback_1(self, msg):

        self.curr_front_distance_1 = self.minimum_distance_from_range(self.scan_angle_front_1[0], self.scan_angle_front_1[1], msg)
        self.temp_curr_front_distance_1 = self.curr_front_distance_1

        if self._fsm_1 == fsm.WAITING_FOR_LASER:
            self.laser_msg_1 = msg

            # Get start and end index that corresponds to the robot's field of view
            min_index = max(int(np.floor((self.scan_angle_1[0] - msg.angle_min) / msg.angle_increment)), 0)
            max_index = min(int(np.ceil((self.scan_angle_1[1] - msg.angle_min) / msg.angle_increment)), len(msg.ranges)-1)
            min_distance = np.min(msg.ranges[min_index:max_index+1]) # find closest distance to the right wall
            timestamp = rospy.get_rostime() # the time current error was computed

            self._fsm_1 = fsm.MOVE_FORWARD
            self._fsm_2 = fsm.MOVE_FORWARD

        elif self._fsm_1 == fsm.MOVE_FORWARD:
    
            print "time"

            time = rospy.get_time()
            print "--"
            print time
            print self.prev_time_1
            # Testing detection
            self.curr_front_distance_1 = self.minimum_distance_from_range(self.scan_angle_front_1[0], self.scan_angle_front_1[1], msg)
            self.temp_curr_front_distance_1 = self.curr_front_distance_1
            # self.avoid_pattern_1 = self.moving_target(float(self.prev_distance_front_1), float(self.curr_front_distance_1), float(timestamp) ,float(self.prev_timestamp_1), float(self.linear_velocity_1))
            if time != self.prev_time_1:
                if (self.curr_front_distance_1 - self.prev_distance_front_1) < self.linear_velocity_1:
                    self.avoid_pattern_1 = self.moving_target(float(self.prev_distance_front_1), float(self.curr_front_distance_1), float(self.prev_time_1) ,float(time), float(self.linear_velocity_1))
                self.prev_distance_front_1 = self.curr_front_distance_1

                print self.avoid_pattern_1
                print "front distance"
                print self.curr_front_distance_1
                if self.avoid_pattern_1 is True and self.curr_front_distance_1 < 1.7:
                    print "DETECTED POTENTIAL COLLISION ****************************"
                    print "starting avoidance manuever"
                    self.start_distance = self.curr_front_distance_1
                    self.stop_distance = self.start_distance
                    self.stop_1()   ##Stop 1
            self.prev_time_1 = time

        elif self._fsm_1 == fsm.STRAIGHT:
            print "Straight"

        elif self._fsm_1 == fsm.AVOID_BASIC:
            self.avoid_sidestep()
            
        elif self._fsm_1 == fsm.STOP:
            print "stopping"
            if self.avoid_pattern_1 is True:
                print "avoiding motion"
                self._fsm_1 = fsm.AVOID_BASIC
            else:
                print "continue path"
                self._fsm_1 = fsm.MOVE_FORWARD

        elif self._fsm_1 == fsm.NEXT_AVOID:
            print "next"
        
        # rospy.sleep(0.01)

 
    def _laser_callback_2(self, msg):
        if self._fsm_2 == fsm.WAITING_FOR_LASER:
            self.laser_msg_2 = msg

            # Get start and end index that corresponds to the robot's field of view
            min_index = max(int(np.floor((self.scan_angle_2[0] - msg.angle_min) / msg.angle_increment)), 0)
            max_index = min(int(np.ceil((self.scan_angle_2[1] - msg.angle_min) / msg.angle_increment)), len(msg.ranges)-1)
            

            min_distance = np.min(msg.ranges[min_index:max_index+1]) # find closest distance to the right wall
            
            error = self.goal_distance_2 - min_distance # current error
            timestamp = rospy.get_rostime() # the time current error was computed

            # if this is first error, set prev err and its timestamp to be same as the curr err so that d term would be zero 
            if self.curr_err_2 is None:
                self.prev_err_2 = error
                self.prev_err_timestamp_2 = timestamp
            
            # if there is prev error, update prev err and its timestamp to more recent past
            else:
                self.prev_err_2 = self.curr_err_2
                self.prev_err_timestamp_2 = self.curr_err_timestamp_2
            
            # update curr err and its timestamp
            self.curr_err_2 = error
            self.curr_err_timestamp_2 = timestamp

            self._fsm_2 = fsm.MOVE # wait until the robot finishes its move before processing a new message

    # ======== Collision Helper Functions ==============================================
    def translate(self, d):
        print "translate"
        self.distance = d
        self.duration = d / self.linear_velocity_1
        self.start_time = rospy.get_rostime()
        self.sum_flag = True # Summing Laserscan distances
        self._fsm_1 = fsm.MOVE_FORWARD

    #Takes input angle and rotates relative angle
    def rotate_rel(self, angle):
        print "rotate relative"
        self.rotate_value = angle
        self.is_rotate_absolute = False #rel by default
        self._fsm_1 = fsm.ROTATE_CALC

    # Sidestep Avoidance Pattern
    def avoid_sidestep(self):
        i = 0
        a = 0
        while a <= 3:

            if a == 0 or a == 3:
                #Turn Right
                self.duration = rospy.Duration(abs(np.pi / 2.0) / self.angular_velocity_1)
                self._fsm_1 = fsm.ROTATE_SQUARE_RIGHT
                i = 0
                while self._fsm_1 != fsm.NEXT_AVOID:
                    if i == 0:
                        self.rotate_rel(-np.pi / 2)
                        i = 1
                if a == 3:
                    break
            else:
                #Turn Left
                self.duration = rospy.Duration(abs(np.pi / 2.0) / self.angular_velocity_1)
                self._fsm_1 = fsm.ROTATE_SQUARE_LEFT
                i = 0
                while self._fsm_1 != fsm.NEXT_AVOID:
                    if i == 0:
                        self.rotate_rel(np.pi / 2)
                        i = 1
            #straight
            self._fsm_1 = fsm.MOVE_FORWARD
            
            if a % 2 == 0:
                dist = 0.7
            else:
                dist = self.stop_distance * 2
                print "long run"
            i = 0
            while self._fsm_1 != fsm.NEXT_AVOID:
                if i == 0:
                    # self.sum_flag = True
                    self.translate(dist)
                    i = 1
            a += 1
        
        self.avoid_pattern_1 = False
        self.prev_distance_front_1 = self.temp_curr_front_distance_1
        self.stop_1()
        print "Done with Avoid Basic"

    # Checks scan velocity and if greater than expected then moving target
    def moving_target(self, min_a, min_b, time_a, time_b, linear_velocity):
        print "moving:"
        print min_a
        print min_b
        print (((min_a - min_b) / (time_b - time_a)))
        print ((((min_a - min_b) / (time_b - time_a))) / linear_velocity)
        # print np.round((((min_a - min_b) / (time_b - time_a))) / linear_velocity)
        print "--"
        # return np.round((((min_a - min_b) / (time_b - time_a)) - linear_velocity) / linear_velocity) > 0.01
        return ((((min_a - min_b) / (time_b - time_a))) / linear_velocity) > 1.5

    # Logic to return the min distance in the angle range
    def minimum_distance_from_range(self, scan_angle_min, scan_angle_max, msg):
        min_index = max(int(np.floor((scan_angle_min - msg.angle_min) / msg.angle_increment)), 0)
        max_index = min(int(np.ceil((scan_angle_max - msg.angle_min) / msg.angle_increment)), len(msg.ranges)-1)
        return np.min(msg.ranges[min_index:max_index+1])
    
    # ======== Control ==============================================

    def control(self):
        rate = rospy.Rate(self.frequency)
        while not rospy.is_shutdown():
            
            # MOVE =====================================
            if self._fsm_1 == fsm.MOVE_FORWARD:

                if rospy.get_rostime() - self.start_time >= rospy.Duration(self.duration):
                    print "Done moving forward"
                    self._fsm_1 = fsm.NEXT_AVOID
                else:
                    self.move_1(self.linear_velocity_1, 0)
                    self.move_2(self.linear_velocity_2, 0)
            # STOP =====================================
            elif self._fsm_1 == fsm.STOP:
                self.stop_1()
            
            # ROTATE CALC =====================================
            elif self._fsm_1 == fsm.ROTATE_CALC:
                self.start_time = rospy.get_rostime()

                self.duration = rospy.Duration(abs(self.rotate_value) /self.angular_velocity_1)
                self._fsm_1 = fsm.ROTATE

            # ROTATE =====================================
            elif self._fsm_1 == fsm.ROTATE:
                if rospy.get_rostime() - self.start_time >= self.duration:
                    print "Done rotating"
                    # self._fsm_1 = fsm.STOP
                    self._fsm_1 = fsm.NEXT_AVOID
                    # self.stop()

                else:
                    self.move_1(0, np.sign(self.rotate_value) * self.angular_velocity_1)


            rate.sleep()


    '''Send a velocity command (linear vel in m/s, angular vel in rad/s).'''
    def move_1(self, linear_vel, angular_vel):
        twist_msg = Twist()
        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        self._cmd_pub_1.publish(twist_msg)

    def move_2(self, linear_vel, angular_vel):
        twist_msg = Twist()
        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        self._cmd_pub_2.publish(twist_msg)

    '''Stop the robot.'''
    def stop_1(self):
        twist_msg = Twist()
        self._cmd_pub_1.publish(twist_msg)
        self._fsm_1 = fsm.STOP
        

    def stop_2(self):
        twist_msg = Twist()
        self._cmd_pub_2.publish(twist_msg)
        self._fsm_2 = fsm.STOP

    def stop_all(self):
        self.stop_1()
        self.stop_2()

def main():
    """Main function."""
    # 1st. initialization of node.
    rospy.init_node("action_robots")

    # Initialize controller 
    controller = PD(cs.K_P, cs.K_D)

    # Initialization 
    action_robots = ActionRobots(controller)

    # Sleep for a few seconds to wait for the registration.
    rospy.sleep(2)

    # If interrupted, send a stop command before interrupting.
    rospy.on_shutdown(action_robots.stop_all)

    try:
        action_robots.control()

    except rospy.ROSInterruptException:
        rospy.logerr("ROS node interrupted.")

if __name__ == "__main__":
    """Run the main function."""
    main()