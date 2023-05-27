#!/usr/bin/env python

# Code adapted from PA2 and PA4

import math
from enum import Enum
import numpy as np
import tf
import rospy # module for ROS APIs
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Twist # message type for cmd_vel
from sensor_msgs.msg import LaserScan # message type for scan

# Topics
DEFAULT_CMD_VEL_TOPIC_1 = 'robot_0/cmd_vel'
DEFAULT_SCAN_TOPIC_1 = 'robot_0/base_scan'
DEFAULT_CMD_VEL_TOPIC_2 = 'robot_1/cmd_vel'
DEFAULT_SCAN_TOPIC_2 = 'robot_1/base_scan'
DEFAULT_ODOM_1 = 'robot_0/odom'
DEFAULT_ODOM_2 = 'robot_1/odom'
# Frequency at which the loop operates
FREQUENCY = 10 # Hz.

# Velocities 
LINEAR_VELOCITY = 0.2 # m/s
ANGULAR_VELOCITY = 0.2 # rad/s

# Field of view
MIN_SCAN_ANGLE_RAD = -math.pi
MAX_SCAN_ANGLE_RAD = math.pi

# Goal distance to the wall
GOAL_DISTANCE = 0.5 # m

# PD parameters
K_P = 4 # proportional gain
K_D = 50 # derivative gain

# Map
MAP_WIDTH = 1000
MAP_HEIGHT = 1000
MAP_RESOLUTION = 0.2

# FSM for robot's state
class fsm(Enum):
    STOP = 0
    WAITING_FOR_LASER = 1
    MOVE = 2

# PD controller
class PD():
    def __init__(self, kp, kd):
        self._p = kp # proportional gain
        self._d = kd # derivative gain
        
    ''' 
    Compute new actuation
    If there is no prev error, curr_err = prev_err and d_t is zero so that the d term will not be accounted for.
    '''
    def step(self, prev_err, curr_err, dt):
        u = self._p * curr_err # p term

        if dt != 0: # dt is zero if there is no prev error
            u += self._d * (curr_err - prev_err) / dt # add d term
        return u

# Map
class Grid:
    def __init__(self, width, height, resolution):
        self.grid = -1 * np.ones((height, width)) # initially set every coordinate as unknown
        self.width = width
        self.height = height
        self.resolution = resolution

    def cell_at(self, x, y):
        return self.grid[y][x]

    def update(self, x, y, value):
        self.grid[y][x] = value

class Maze():
    def __init__(self,
                 controller,
                 goal_distance=GOAL_DISTANCE,
                 linear_velocity=LINEAR_VELOCITY,
                 angular_velocity=ANGULAR_VELOCITY,
                 frequency=FREQUENCY,
                 scan_angle=[MIN_SCAN_ANGLE_RAD, MAX_SCAN_ANGLE_RAD],
                 map_width=MAP_WIDTH,
                 map_height=MAP_HEIGHT,
                 map_resolution=MAP_RESOLUTION):

        # Publisher/Subscriber
        self._cmd_pub_1 = rospy.Publisher(DEFAULT_CMD_VEL_TOPIC_1, Twist, queue_size=1) # publisher to send velocity commands.
        self._cmd_pub_2 = rospy.Publisher(DEFAULT_CMD_VEL_TOPIC_2, Twist, queue_size=1) # publisher to send velocity commands.

        self._occupancy_grid_pub = rospy.Publisher('map', OccupancyGrid, queue_size=1) # publishes occupancy grid info
        self._laser_sub_1 = rospy.Subscriber(DEFAULT_SCAN_TOPIC_1, LaserScan, self._laser_callback_1, queue_size=1) # subscriber receiving messages from the laser.
        self._laser_sub_2 = rospy.Subscriber(DEFAULT_SCAN_TOPIC_2, LaserScan, self._laser_callback_2, queue_size=1) # subscriber receiving messages from the laser.

        self._odom_sub_1 = rospy.Subscriber(DEFAULT_ODOM_1, Odometry, self._odom_callback_1, queue_size=1)
        self._odom_sub_2 = rospy.Subscriber(DEFAULT_ODOM_2, Odometry, self._odom_callback_2, queue_size=1)

        # Parameters
        self.linear_velocity_1 = linear_velocity
        self.angular_velocity_1 = angular_velocity
        self.frequency = frequency
        self.scan_angle_1 = scan_angle
        self.goal_distance_1 = goal_distance

        self.linear_velocity_2 = linear_velocity
        self.angular_velocity_2 = angular_velocity

        self.scan_angle_2 = scan_angle
        self.goal_distance_2 = goal_distance + 0.3

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

    '''Publish occupancy grid message'''
    def publish_map(self):
        if self.laser_msg_1 != None:
            msg = self.laser_msg_1

            # update grid
            for i in range(len(msg.ranges)):
                angle = msg.angle_min + msg.angle_increment * i
                self.update_map(angle, msg.ranges[i]) # updates map in the direction defined by the angle and distance

            # create and publish occupancy grid message
            occupancy_grid_msg = OccupancyGrid()
            occupancy_grid_msg.header.frame_id = "map"
            occupancy_grid_msg.header.stamp = rospy.Time.now()

            occupancy_grid_msg.info.width = self.map_width
            occupancy_grid_msg.info.height = self.map_height
            occupancy_grid_msg.info.resolution = self.map_resolution

            occupancy_grid_msg.info.origin.position.x = -(self.map_width/2) * self.map_resolution
            occupancy_grid_msg.info.origin.position.y = -(self.map_height/2) * self.map_resolution
            occupancy_grid_msg.info.origin.orientation.w = 0

            occupancy_grid_msg.data = self.map.grid.flatten()

            self._occupancy_grid_pub.publish(occupancy_grid_msg)

    '''Updates map in a direction defined by the angle and distance'''
    def update_map(self, angle, distance):
        # find transformation from base_link to odom
        (trans, rot) = self.listener_1.lookupTransform('robot_0/odom', 'robot_0/base_laser_link', rospy.Time(0))

        angle += tf.transformations.euler_from_quaternion(rot)[2] # angle in odom reference

        # find robot's position in grid
        grid_x1 = self.map_width/2 + int(trans[0] / self.map_resolution)
        grid_y1 = self.map_height/2 + int(trans[1] / self.map_resolution)

        # find the position of the obstacle (defined by the angle and distance) in grid
        odom_x2 = trans[0] + np.cos(angle) * distance
        odom_y2 = trans[1] + np.sin(angle) * distance
        grid_x2 = self.map_width/2 + int(odom_x2 / self.map_resolution)
        grid_y2 = self.map_height/2 + int(odom_y2 / self.map_resolution)

        self.trace_line(grid_x1, grid_y1, grid_x2, grid_y2) # update values along this line

    '''
    Brehensem's Line Drawing algorithm to update values for coordinates along the line between (x1, y1) and (x2, y2).
    (x1, y1) and (x2, y2) should be provided in grid coordinates.

    Note that the algorithm iteslf works only when 0 < slope <= 1 and x1 <= x2 (we either move N or NE direction).
    If conditions don't meet, make following modifications.
    
    if Case 1 (x1 == x2 or y1 == y2):
        loop through points in x or y direction to trace horizontal/vertical line. 
    else:
        if Case 2 (x1 > x2):
            Consider (x2, y2) as start and (x1, y1) as end.

        if Case 3 (slope < 0):
            Consider (x1, -y1) as start and (x2, -y2) as end.
            For each point (x, y) in line, update (x, -y) instead.
            For us, this means reflecting with respect to the center of the grid.

        if Case 4 (slope > 1):
            Consider (y1, x1) as start and (y2, x2) as end.
            For each point (x, y) in line, update (y, x) instead.

    Reference: https://medium.com/geekculture/bresenhams-line-drawing-algorithm-2e0e953901b3 
    '''
    def trace_line(self, x1, y1, x2, y2):
        # case 1: x1 == x2, loop through y coordinates
        if x1 == x2:
            y = y1

            while y != y2:
                self.map.update(x1, y, 0)
                y = y + np.sign(y2 - y1) * 1

            self.map.update(x2, y2, 100)
            return
        
        # case 1: x1 == x2, loop through x coordinates
        elif y1 == y2:
            x = x1

            while x != x2:
                self.map.update(x, y1, 0)
                x = x + np.sign(x2 - x1) * 1
            
            self.map.update(x2, y2, 100)
            return

        slope = (y2 - y1) / float(x2 - x1)
        x1_greater_than_x2 = x1 > x2 # case 2
        slope_is_negative = slope < 0 # case 3
        slope_greater_than_one = abs(slope) > 1 # case 4

        # start and end points to use for the algorithm
        b_x1, b_y1, b_x2, b_y2 = x1, y1, x2, y2
 
        if x1_greater_than_x2: 
            b_x1, b_y1, b_x2, b_y2 = b_x2, b_y2, b_x1, b_y1 # consider (x2, y2) as start and (x1, y1) as end
        if slope_is_negative:
            b_y1, b_y2 = self.map_height - b_y1, self.map_height - b_y2 # consider (x1, -y1) as start and (x2, -y2) as end
        if slope_greater_than_one:
            b_x1, b_y1, b_x2, b_y2 = b_y1, b_x1, b_y2, b_x2 # consider (y1, x1) as start and (y2, x2) as end

        # define parameters for algorithm
        dx = b_x2 - b_x1
        dy = b_y2 - b_y1
        d = 2 * dy - dx # decision parameter to move E or NE
        d_E = 2 * dy
        d_NE = 2 * (dy - dx)
        
        x, y = b_x1, b_y1

        # move E or NE based on the decision parameter
        while x != b_x2:
            if d <= 0:
                d += d_E
                x += 1
            else:
                d += d_NE
                x += 1
                y += 1

            reversed_x, reversed_y = x, y # reverse modifications to find coordinates in the original grid

            if slope_greater_than_one:
                reversed_x, reversed_y = reversed_y, reversed_x # exchange x and y
            if slope_is_negative:
                reversed_y =  self.map_height - reversed_y # exchange the sign of y
            if 0 <= reversed_x < self.map_width and 0 <= reversed_y < self.map_height:
                self.map.update(reversed_x, reversed_y, 0) # mark point between robot's position and obstacle as free space
        
        self.map.update(x1, y1, 0) # mark robot's current position as free space
        self.map.update(x2, y2, 100) # mark obstacle's position as occupied

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
    rospy.init_node("detection_robots")

    # Initialize controller 
    controller = PD(K_P, K_D)

    # Initialization of the class for the random walk.
    maze = Maze(controller)

    # Sleep for a few seconds to wait for the registration.
    rospy.sleep(2)

    # If interrupted, send a stop command before interrupting.
    rospy.on_shutdown(maze.stop)

    try:
        maze.control()

    except rospy.ROSInterruptException:
        rospy.logerr("ROS node interrupted.")

if __name__ == "__main__":
    """Run the main function."""
    main()