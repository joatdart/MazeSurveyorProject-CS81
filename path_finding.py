#!/usr/bin/env python
# path planning (traveling salesman problem)
# written by: Andrea Robang
# note: still editing/debugging!

import rospy
import math
import numpy as np
import tf


from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

import heapq

ROBOT_ORIGIN_X = 5
ROBOT_ORIGIN_Y = 5
ROBOT_RADIUS = 4 # in # of cells


class Grid:
    def __init__(self, occupancy_grid_data, width, height, resolution):
        enlarge = np.reshape(occupancy_grid_data, (height, width))
        self.grid = self.enlarge_obstacles(ROBOT_RADIUS, enlarge)
        self.resolution = resolution

    def cell_at(self, x, y):
        return self.grid[y, x]
    
    def enlarge_obstacles(self, robot_radius, map_array):
        inflated_obstacles = np.zeros_like(map_array)
        for y in range(map_array.shape[0]):
            for x in range(map_array.shape[1]):
                if map_array[y][x] == 100:
                    inflated_obstacles[max(0, y - robot_radius):min(int(map_array.shape[0]), y + robot_radius + 1),
                                      max(0, x - robot_radius):min(int(map_array.shape[1]), x + robot_radius + 1)] = 100

        return inflated_obstacles

class MazeTraverser:
    def __init__(self):
        
        self.points = []  # List of points to visit
        self.path = []  # The optimized path
        self.map = None
        self.current = (0, 0, 0)

        self.sub = rospy.Subscriber("map", OccupancyGrid, self.map_callback, queue_size=1)
        self.pose_pub = rospy.Publisher("pose", PoseStamped, queue_size=1)

        self._cmd_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)    
        self._laser_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        
        # Wait for the occupancy grid to be received
        print("waiting for occupancy")
        rospy.wait_for_message("map", OccupancyGrid)
        # Wait a moment to let the map data propagate
        rospy.sleep(0.1)
        
        print("init")

    def odom_callback(self, msg):
        # Extract the x and y coordinates from the Odometry message
        x = msg.pose.pose.position.x + ROBOT_ORIGIN_X
        y = msg.pose.pose.position.y + ROBOT_ORIGIN_Y
        theta = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
        
        self.current = (x, y, theta)

    def map_callback(self, msg):
        self.map = Grid(msg.data, msg.info.width, msg.info.height, msg.info.resolution)

    def publish_pose(self, pose, i):
        pose_msg = PoseStamped()
        pose_msg.header.seq = i
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = pose[0]
        pose_msg.pose.position.y = pose[1]
        pose_msg.pose.position.z = 0.0
        quaternion = tf.transformations.quaternion_from_euler(0, 0, pose[2])
        pose_msg.pose.orientation.x = quaternion[0]
        pose_msg.pose.orientation.y = quaternion[1]
        pose_msg.pose.orientation.z = quaternion[2]
        pose_msg.pose.orientation.w = quaternion[3]
        
        self.pose_pub.publish(pose_msg)

        return pose_msg

    def calculate_distance(self, point1, point2):
        # Calculate the Euclidean distance between two points
        x1, y1 = self.get_coordinates(point1)
        x2, y2 = self.get_coordinates(point2)
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Heuristic function (Euclidean distance from a point to the starting point)
    def heuristic(self, start, point):
        return MazeTraverser.calculate_distance(point, start)

    def get_coordinates(self, point):

        return point

    def is_obstacle_between(point1, point2, occupancy_grid):
        x1, y1 = point1
        x2, y2 = point2

        # Perform line-of-sight check using Bresenham's algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while x1 != x2 or y1 != y2:
            if occupancy_grid[y1][x1] == 1:
                return True

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        return False

    def solve_tsp(self, points, obstacles):
        # Generate the maze graph (considering obstacles)
        print("starting tsp")
        n = len(points)
        graph = {}
        for i in range(n):
            graph[i] = {}
            for j in range(n):
                if i != j:
                    if not MazeTraverser.is_obstacle_between(points[i], points[j], self.map):
                        graph[i][j] = MazeTraverser.distance(points[i], points[j])

        start = points[0]
        open_set = [(MazeTraverser.heuristic(start), 0, [0])]
        closed_set = set()

        while open_set:
            _, cost, path = heapq.heappop(open_set)
            current_point = path[-1]

            if current_point in closed_set:
                continue

            if len(path) == n and current_point == 0:
                return path

            closed_set.add(current_point)

            for neighbor, edge_cost in graph[current_point].items():
                if neighbor not in closed_set:
                    new_cost = cost + edge_cost
                    new_path = path + [neighbor]
                    heapq.heappush(open_set, (new_cost + MazeTraverser.heuristic(points[neighbor]), new_cost, new_path))

    def move(self, linear_vel, angular_vel):
        """Send a velocity command (linear vel in m/s, angular vel in rad/s)."""
        # Setting velocities.
        twist_msg = Twist()

        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        self._cmd_pub.publish(twist_msg)

    def sustained_move(self, linear_vel, angular_vel, duration):
        start_time = rospy.get_rostime()
        while not rospy.is_shutdown():
            self.move(linear_vel, angular_vel)

            if rospy.get_rostime() - start_time >= rospy.Duration(duration):
                break

    def point_move(self, x, y):
        rate = rospy.Rate(10)
        
        # Calculate the angle and distance to the target point
        dx = x - self.current[0]
        dy = y - self.current[1]
        theta = math.atan2(dy, dx)

        target_distance = (dx ** 2 + dy ** 2) ** 0.5
        target_angle = (theta - self.current[2]) % (2 * math.pi)

        angular_vel = math.pi/2
        linear_vel = 0.2
        # Rotate to face the target point
        # counter-clockwise
        if target_angle < math.pi:
          duration = target_angle/angular_vel
          self.sustained_move(0, angular_vel, duration)

        # clockwise
        else:
          target_angle = 2*math.pi - target_angle
          duration = target_angle/angular_vel
          self.sustained_move(0, -angular_vel, duration)

        # Move forward to the target point
        duration = target_distance/linear_vel
        self.sustained_move(linear_vel, 0, duration)
            
        rate.sleep()

    def navigate(self):
        # publish poses in pose topic
        for i, pose in enumerate(self.path):
          self.publish_pose(pose, i)            

        # traverse path
        for next in self.path:
            # convert grid to world
            point = self.map_to_world(next[0], next[1])
            self.point_move(point[1], point[0])

        self.pub.publish('Navigation complete.')

def main():

    rospy.init_node('maze_traverser', anonymous=True)

    rate = rospy.Rate(10)
    traverser = MazeTraverser()
    while not rospy.is_shutdown():
        traverser.points = [(5,5),  (7,7), (6,6), (8,7), (4, 6)] 
        if traverser.points:
            # Replace with your own points
            obstacles = [((1, 1), (2, 3)), ((3, 2), (9, 9))]  # Replace with obstacles in the maze

            traverser.path = MazeTraverser.solve_tsp(traverser.points, obstacles)
            print("TSP Solution:", traverser.path)
            traverser.navigate()
            traverser.points = []
    
    rospy.sleep(2)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
