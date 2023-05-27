#!/usr/bin/env python

# testing for action collision

import math
import numpy as np
import tf
import rospy # module for ROS APIs
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Twist # message type for cmd_vel
from sensor_msgs.msg import LaserScan # message type for scan

from pid import PD 
from grid import Grid, fsm
import constants as cs


class ActionRobots():
    def __init__(self,
                 controller,
                 goal_distance=cs.GOAL_DISTANCE,
                 linear_velocity=cs.LINEAR_VELOCITY,
                 angular_velocity=cs.ANGULAR_VELOCITY,
                 frequency=cs.FREQUENCY,
                 scan_angle=[cs.MIN_SCAN_ANGLE_RAD, cs.MAX_SCAN_ANGLE_RAD],
                 map_width=cs.MAP_WIDTH,
                 map_height=cs.MAP_HEIGHT,
                 map_resolution=cs.MAP_RESOLUTION):

        # Publisher/Subscriber
        self._cmd_pub_1 = rospy.Publisher(cs.DEFAULT_CMD_VEL_TOPIC_1, Twist, queue_size=1) # publisher to send velocity commands.
        self._cmd_pub_2 = rospy.Publisher(cs.DEFAULT_CMD_VEL_TOPIC_2, Twist, queue_size=1) # publisher to send velocity commands.

        self._occupancy_grid_pub = rospy.Subscriber(cs.MAP_TOPIC, OccupancyGrid, queue_size=1) # subscribes to occupancy grid info
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
        if self._fsm_1 == fsm.WAITING_FOR_LASER:
            self.laser_msg_1 = msg

            # Get start and end index that corresponds to the robot's field of view
            min_index = max(int(np.floor((self.scan_angle_1[0] - msg.angle_min) / msg.angle_increment)), 0)
            max_index = min(int(np.ceil((self.scan_angle_1[1] - msg.angle_min) / msg.angle_increment)), len(msg.ranges)-1)
            
            min_distance = np.min(msg.ranges[min_index:max_index+1]) # find closest distance to the right wall
            
            error = self.goal_distance_1 - min_distance # current error
            timestamp = rospy.get_rostime() # the time current error was computed

            # if this is first error, set prev err and its timestamp to be same as the curr err so that d term would be zero 
            if self.curr_err_1 is None:
                self.prev_err_1 = error
                self.prev_err_timestamp_1 = timestamp
            
            # if there is prev error, update prev err and its timestamp to more recent past
            else:
                self.prev_err_1 = self.curr_err_1
                self.prev_err_timestamp_1 = self.curr_err_timestamp_1
            
            # update curr err and its timestamp
            self.curr_err_1 = error
            self.curr_err_timestamp_1 = timestamp

            self._fsm_1 = fsm.MOVE # wait until the robot finishes its move before processing a new message

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

    def control(self):
        rate = rospy.Rate(self.frequency) # loop at 10 Hz.

        while not rospy.is_shutdown(): # keep looping until user presses Ctrl+C
            if self._fsm_1 == fsm.MOVE:
                d_t = self.curr_err_timestamp_1.to_sec() - self.prev_err_timestamp_1.to_sec() # time elapsed between prev and curr error
                rotation_angle = self.controller_1.step(self.prev_err_1, self.curr_err_1, d_t)
                self.move_1(self.linear_velocity_1, rotation_angle)
                self.publish_map()
                self._fsm_1 = fsm.WAITING_FOR_LASER # don't move and wait until a new laser message has been processed
        
            if self._fsm_2 == fsm.MOVE:
                d_t = self.curr_err_timestamp_2.to_sec() - self.prev_err_timestamp_2.to_sec() # time elapsed between prev and curr error
                rotation_angle = self.controller_2.step(self.prev_err_2, self.curr_err_2, d_t)
                self.move_2(self.linear_velocity_2,rotation_angle)
                # self.publish_map()
                self._fsm_2 = fsm.WAITING_FOR_LASER # don't move and wait until a new laser message has been processed
        
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
        twist_msg.angular.z = -1 * angular_vel
        self._cmd_pub_2.publish(twist_msg)

    '''Stop the robot.'''
    def stop(self):
        twist_msg = Twist()
        self._cmd_pub_1.publish(twist_msg)
        self._fsm_1 = fsm.STOP
        
        self._cmd_pub_2.publish(twist_msg)
        self._fsm_2 = fsm.STOP


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
    rospy.on_shutdown(action_robots.stop)

    try:
        action_robots.control()

    except rospy.ROSInterruptException:
        rospy.logerr("ROS node interrupted.")

if __name__ == "__main__":
    """Run the main function."""
    main()