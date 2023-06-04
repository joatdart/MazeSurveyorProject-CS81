#!/usr/bin/env python
# J. Hwang

import math

# Topics
DEFAULT_CMD_VEL_TOPIC_1 = 'robot_2/cmd_vel'
DEFAULT_SCAN_TOPIC_1 = 'robot_2/base_scan'
DEFAULT_CMD_VEL_TOPIC_2 = 'robot_3/cmd_vel'
DEFAULT_SCAN_TOPIC_2 = 'robot_3/base_scan'
DEFAULT_ODOM_1 = 'robot_2/odom'
DEFAULT_ODOM_2 = 'robot_3/odom'

MAP_TOPIC = 'map'

# Frequency at which the loop operates
FREQUENCY = 5 # Hz.

# Velocities 
LINEAR_VELOCITY = 0.2 # m/s
ANGULAR_VELOCITY = 0.5 # rad/s

# Field of view
MIN_SCAN_ANGLE_RAD = -math.pi
MAX_SCAN_ANGLE_RAD = math.pi

MIN_SCAN_ANGLE_RAD_FRONT = -15.0 / 180 * math.pi;
MAX_SCAN_ANGLE_RAD_FRONT = 45.0 / 180 * math.pi;

# Goal distance to the wall
GOAL_DISTANCE = 0.5 # m

# PD parameters
K_P = 4 # proportional gain
K_D = 50 # derivative gain

# Map
MAP_WIDTH = 1000
MAP_HEIGHT = 1000
MAP_RESOLUTION = 0.2

# Collision
TRANSLATE_DISTANCE = 1 #m
ROTATE_REL_VALUE = math.pi / 6
ROTATE_ABS_VALUE = math.pi / 6
COLLISION_DISTANCE = 1.6

# Action Robot
S_GOAL = [
            (0.5, 0.5),
            (1, 1)
        ]

S_EXIT = [0.35, 0.15]
S_ENTRY = [0.35, 0.15]

ORIGIN_X_1 = 2
ORIGIN_Y_1 = 2

ORIGIN_X_2 = 2
ORIGIN_Y_2 = 1