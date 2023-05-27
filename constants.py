#!/usr/bin/env python

# Topics
DEFAULT_CMD_VEL_TOPIC_1 = 'robot_2/cmd_vel'
DEFAULT_SCAN_TOPIC_1 = 'robot_2/base_scan'
DEFAULT_CMD_VEL_TOPIC_2 = 'robot_3/cmd_vel'
DEFAULT_SCAN_TOPIC_2 = 'robot_3/base_scan'
DEFAULT_ODOM_1 = 'robot_2/odom'
DEFAULT_ODOM_2 = 'robot_3/odom'

MAP_TOPIC = 'map'

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
