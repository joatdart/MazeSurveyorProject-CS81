#!/usr/bin/env python

from enum import Enum

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

# FSM for robot's state
class fsm(Enum):
    STOP = 0
    WAITING_FOR_LASER = 1
    MOVE = 2
    COLLISION_DETECTED = 3
    AVOID_WITH_PID = 4
    AVOID_WITH_BACKTRACK = 5