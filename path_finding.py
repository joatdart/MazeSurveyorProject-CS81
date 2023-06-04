#!/usr/bin/env python
# written by: Andrea Robang
# path finding node for cs81 final project

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Pose, Twist
from nav_msgs.msg import Path
from std_msgs.msg import Header
import math
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from Queue import PriorityQueue

# Global variables
# NOTED: EDIT AS NECESSARY
ROBOT_ORIGIN_X = 2
ROBOT_ORIGIN_Y = 2
ROBOT_RADIUS = 4 # in # of cells
FREQUENCY = 10
MAX_ANGULAR_VEL = math.pi/2
MAX_LINEAR_VEL = 0.2

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



class ThetaStarNode:
    def __init__(self, goal_coordinates, start):
        self.goal_poses = [ThetaStarNode.createPose(self, x, y, 1) for (x, y) in goal_coordinates]
        self.current = (0, 0, 0)
        self.start_pose = ThetaStarNode.createPose(self, start[0], start[1], 0)
        self.map = None

        rospy.init_node('theta_star_node')
       

        self._cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self._rate = rospy.Rate(10)  # 10 Hz

        self.sub = rospy.Subscriber('map', OccupancyGrid, self.occupancyGridCallback)
        self._laser_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        print "starting"
        # Wait for the occupancy grid to be received
        rospy.wait_for_message("map", OccupancyGrid)

        rospy.sleep(1)

    def occupancyGridCallback(self, msg):
        self.map = Grid(msg.data, msg.info.width, msg.info.height, msg.info.resolution)

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x + ROBOT_ORIGIN_X
        y = msg.pose.pose.position.y + ROBOT_ORIGIN_Y
        theta = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
        
        self.current = self.createPose(x, y, theta)

    def createPose(self, x, y, w):
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.0
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = w
        return pose

    def calculatePoseEuclideanDistance(self, p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def findClosestGoalPose(self, start, goal_poses):
        closest_distance = float('inf')
        closest_pose = None

        for goal_pose in goal_poses:
            distance = ThetaStarNode.calculatePoseEuclideanDistance(self, start.position, goal_pose.position)
            if distance < closest_distance:
                closest_distance = distance
                closest_pose = goal_pose

        return closest_pose
    
    def move(self, linear_vel, angular_vel):
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
            
    def reconstructPath(self, parents, goal_cell):
        # Reconstruct the path from goal to start using the parents dictionary
        path = []
        current_cell = goal_cell

        while current_cell is not None:
            pose = self.cellToPose(current_cell)
            path.append(pose)
            current_cell = parents[current_cell]

        path.reverse()
        return path

    def poseToCell(self, pose):
        x = int(pose.position.x / self.map.resolution)
        y = int(pose.position.y / self.map.resolution)
        return (x, y)


    def cellToPose(self, cell):
        # Convert cell coordinates to pose coordinates
        x = cell[0] * self.map.resolution
        y = cell[1] * self.map.resolution
        return ThetaStarNode.createPose(self, x, y, 1)

    def calculateCellEuclideanDistance(self, node1, node2):
        # Euclidean distance between two nodes
        dx = node2[0] - node1[0]
        dy = node2[1] - node1[1]
        return math.sqrt(dx**2 + dy**2)

    def is_in_priority_queue(self, pq, item):
        for _, q_item in pq.queue:
            if q_item == item:
                return True
        return False

    def findThetaStarPath(self, start, goal):
        start_cell = self.poseToCell(start)
        goal_cell = self.poseToCell(goal)

        # Initialize the open and closed lists
        open_list = PriorityQueue()
        open_list.put(start_cell)
        closed_list = set()

        # Create a dictionary to store the parent of each cell
        parents = {}
        parents[start_cell] = None

        # Create a dictionary to store the cost to reach each cell
        costs = {}
        costs[start_cell] = 0

        print "running tsp"
        while not open_list.empty():
            current_cell = open_list.get()

            if current_cell == goal_cell:
                # Path found, reconstruct it from the parents dictionary
                path = self.reconstructPath(parents, current_cell)
                return path

            closed_list.add(current_cell)

            neighbors = self.generateNeighbors(current_cell)
            for neighbor_cell in neighbors:
                if neighbor_cell in closed_list:
                    continue

                tentative_cost = costs[current_cell] + self.calculateCellEuclideanDistance(current_cell, neighbor_cell)

                if not ThetaStarNode.is_in_priority_queue(self, open_list, neighbor_cell):
                    open_list.put(neighbor_cell)

                elif tentative_cost >= costs[neighbor_cell]:
                    continue


                parents[neighbor_cell] = current_cell
                costs[neighbor_cell] = tentative_cost

        # No path found
        return None


    def generateNeighbors(self, cell):
        neighbors = []
        x, y = cell[0], cell[1]

        # Define the possible movement directions (4-connected)
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        for dx, dy in directions:
            neighbor_x, neighbor_y = x + dx, y + dy

            if self.isValidCell(neighbor_x, neighbor_y) and not self.map.cell_at(neighbor_x, neighbor_y):
                neighbors.append((neighbor_x, neighbor_y))
        return neighbors


    def isValidCell(self, x, y):
        # Check if the given cell is within the map boundaries
        map_width = self.map.grid.shape[1]
        map_height = self.map.grid.shape[0]
        return 0 <= x < map_width and 0 <= y < map_height

    # NOTED: CAN REPLACE WITH YOUR OWN MOVEMENT FUNCTION (AND SUBSEQUENT FUNCTIONS USED E.G. MOVE, SUSTAINED_MOVE)
    def point_move(self, x, y):
        rate = rospy.Rate(FREQUENCY)
        
        # Calculate the angle and distance to the target point
        dx = x - self.current.position.x
        dy = y - self.current.position.y
        theta = math.atan2(dy, dx)

        target_distance = (dx ** 2 + dy ** 2) ** 0.5
        target_angle = (theta - self.current.orientation.w) % (2 * math.pi)

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

    def navigateToGoalPoses(self):
        if self.map is not None and self.current is not None and len(self.goal_poses) > 0:
                
            while self.goal_poses:
                closest_goal_pose = ThetaStarNode.findClosestGoalPose(self, self.current, self.goal_poses)
                path = self.findThetaStarPath(self.current, closest_goal_pose)
                for point in path:
                    print (point.position.x," and ", point.position.y)
                    x = point.position.x
                    y = point.position.y

                    # NOTED: MOVEMENT FUNCTIONALITY, EDIT BASED ON COLLISION NODE
                    ThetaStarNode.point_move(self, x, y)

                    # Wait for a short duration before sending the next command
                    rospy.sleep(0.1)

                self.goal_poses.remove(closest_goal_pose)

        self._rate.sleep()
    
    def getTS_Path(self):
        result = []
        if self.map is not None and self.current is not None and len(self.goal_poses) > 0:
                
            while self.goal_poses:
                closest_goal_pose = ThetaStarNode.findClosestGoalPose(self, self.current, self.goal_poses)
                path = self.findThetaStarPath(self.current, closest_goal_pose)
                result.append(path)
                self.goal_poses.remove(closest_goal_pose)

        self._rate.sleep()
        return result

def main():
    goal_coordinates = [
        (0.5, 0.5),
        (1, 1)
    ]

    start_pose = (ROBOT_ORIGIN_X, ROBOT_ORIGIN_Y)

    node = ThetaStarNode(goal_coordinates, start_pose)
    # node.navigateToGoalPoses()
    print(node.getTS_Path())

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

