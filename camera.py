#!/usr/bin/env python

'''
ROS node that detects colored objects via camera
Authors: Tai Wan Kim, Kamakshi Moparthi
Date: May-June 2023

Implementation Summary
    1. Read rgb image from camera
    2. Locate colored object using cv2 and save pixel coordinates
    3. Read point cloud from depth camera
    4. Find point cloud coordinates for the detected object's pixel coordinates
    5. Publish 3D coordinates as a PoseArray
'''

# Modules and libraries
import rospy
import numpy as np
import cv2
import tf
import math
from PIL import Image as PILImage
from enum import Enum
from tf.transformations import quaternion_matrix
import sensor_msgs.point_cloud2 as pc2

# Topics
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Pose, PoseArray

# Parameters for OpenCV
LOWER, UPPER = 0, 1
RED_INDEX, GREEN_INDEX, BLUE_INDEX = 2, 1, 0
RED, GREEN, BLUE, BLACK = (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 0)
RED_TEXT, GREEN_TEXT, BLUE_TEXT, BLACK_TEXT = 'red', 'green', 'blue', 'black'
COLOR_THRESHOLD = np.array([[[100, 50, 50], [130, 255, 255]], 
                            [[40, 20, 50], [90, 255, 255]], 
                            [[0, 50, 50], [10, 255, 255]]])

IMAGE_PATH = 'image.jpg' # file path to store rgb image
OUTPUT_IMAGE_PATH = 'bounded.jpg' # file path to store rgb image with bounding box around detected object
PIXEL_COUNT = 1200

FREQUENCY = 10 # loop frequency
ERROR_THRESHOLD = 1

class fsm(Enum):
    WAITING_FOR_CAMERA = 0 # wait for camera callback
    SAVE_IMAGE = 1 # save rgb image
    DETECT_OBJECT = 2 # find colored object 
    WAITING_FOR_PC = 3 # wait for point cloud callback
    GET_LOCATION = 4 # get 3d coordinate of object
    PUBLISH = 5 # publish object poses

class Camera():
    def __init__(self):
        self._pose_array_pub = rospy.Publisher('object_pose_sequence', PoseArray, queue_size=10)
        self._camera_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self._camera_callback)
        self.point_cloud_sub = rospy.Subscriber('/camera/depth/points', PointCloud2, self._pc_callback)

        self.image = None
        self.bounding_box = None
        self.object_detected = False
        self.point_cloud = None

        self.listener = tf.TransformListener()
        self.fsm = fsm.WAITING_FOR_CAMERA
        
        self.object_pose_list = []
        self.error_threshold = ERROR_THRESHOLD

        # Color lower and upper bounds for Blue, Green, and Red colors in the HSV representation
        self.color_threshold = COLOR_THRESHOLD

    '''Read from '/camera/rgb/image_raw' and save it as self.image (Author: Tay)'''
    def _camera_callback(self, msg):
        if self.fsm == fsm.WAITING_FOR_CAMERA:
            self.image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            self.fsm = fsm.SAVE_IMAGE
    
    '''Read from '/camera/depth/points' and save msg as self.point_cloud (Author: Tay)'''
    def _pc_callback(self, msg):
        if self.fsm == fsm.WAITING_FOR_PC:
            self.point_cloud = msg
            self.fsm = fsm.GET_LOCATION

    '''Save rgb camera readings as PIL Image at a given file path (Author: Tay)'''
    def save_image(self):
        if self.fsm == fsm.SAVE_IMAGE:
            # print("save rgb image")
            # print("IMAGE in SAVE_IMAGE",self.image)
            img = PILImage.fromarray(np.uint8(self.image))
            img.save(IMAGE_PATH)
            # cv2.imwrite('image.jpg',self.image)
            self.fsm = fsm.DETECT_OBJECT

    '''
    Detect colored object via cv2.
    Reads rgb image and convert it to HSV represenation.
    Draw bounding box around colored object via @draw_bounding_box.
    Save image with bounding box.
    (Author: Kamakshi)
    '''
    def identify_colored_object(self):
        if self.fsm == fsm.DETECT_OBJECT:
            # print("get bounding box for colored object")

            image  = cv2.imread(IMAGE_PATH)
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            image = self.draw_bounding_box(image, hsv_image, RED_INDEX)
            # image = self.drawBoundingBox(image, hsvImage, GREEN_INDEX)
            # image = self.draw_bounding_box(image, hsv_image, BLUE_INDEX)

            cv2.imshow('Boximage', image)
            # cv2.waitKey(0)
            cv2.imwrite(OUTPUT_IMAGE_PATH, image)

            # print("bounding_box: " + str(self.bounding_box))
            self.fsm = fsm.WAITING_FOR_PC

    '''
    Draw a bounding box around the object of a specific color.
    Inputs:
        image: original image
        hsvImage: 'image' converted to HSV representation
        colorCode: color to draw bounding box

    Returns:
        image: new image with bounding box around colored objects 

    (Author: Kamakshi)
    '''
    def draw_bounding_box(self, image, hsv_image, color_code):
        masked_image = cv2.inRange(hsv_image, self.color_threshold[color_code][LOWER],self.color_threshold[color_code][UPPER])
        contours, _ = cv2.findContours(masked_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        if len(contours) > 0:
            # print("object detected!")
            self.object_detected = True

            for contour in contours:
                contour_area = cv2.contourArea(contour)
                if contour_area > PIXEL_COUNT:
                    x, y, w, h = cv2.boundingRect(contour)
                    t = self.get_color_info(color_code)
                    cv2.rectangle(image, (x, y), (x + w, y + h), t[1], 2)
                    self.bounding_box = (x, y, w, h)
                    cv2.putText(image, t[0], (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.9, t[1], 2)
        else:
            # print("object not found.")
            self.object_detected = False

        return image

    '''
    Get point cloud x, y, z values corresponding to the center of the bounding box of the colored object
    For simplicity, we only take the x,y values and disregard z (assume that the colored object is on the ground).
    (Author: Tay)
    '''
    def get_3d_coordinates(self):
        if self.fsm == fsm.GET_LOCATION:
            if self.point_cloud is not None and self.bounding_box is not None:
                pc = self.point_cloud # point cloud message

                # print("header: " + str(pc.header))
                # print("width " + str(pc.width))
                # print("height: " + str(pc.height))
                # print("length: " + str(len(pc.data)))

                col = self.bounding_box[0] + int(self.bounding_box[2]/2)
                row = self.bounding_box[1] + int(self.bounding_box[3]/2)
                index = row * pc.width + col # get index of center of the object (point cloud data is 1D array)

                # print("row: " + str(row))
                # print("col: " + str(col))
                # print("index: " +  str(index))

                # get x, y, z value for given index
                for i, p in enumerate(pc2.read_points(pc, field_names=("x", "y", "z"), skip_nans=False)):
                    if i == index:
                        x, y, z = p[0], p[1], p[2]
                        break

                # transform pose in camera reference frame to odom reference frame
                (trans, rot) = self.listener.lookupTransform('odom', 'camera_depth_frame', rospy.Time(0))
                rotation_matrix = quaternion_matrix(rot)
                transform_matrix = np.identity(4)
                transform_matrix[:3, :3] = rotation_matrix[:3, :3]
                transform_matrix[:3, 3] = trans
                point_camera = np.array([[x], [y], [z], [1]])
                point_odom = np.matmul(transform_matrix, point_camera)

                # print("3D coordinate of detected object: " + str(point_odom))
                self.update_object_list(point_odom[0], point_odom[1])
            self.fsm = fsm.PUBLISH

    '''add (x, y) as the pose of the new object (Author: Tay)'''
    def update_object_list(self, x, y):
        new_object_found = True
        
        # prevent same object from being added multiple times
        # add new object only if it is not within 1m to objects already in the list
        for object_pose in self.object_pose_list:
            if self.compute_distance(object_pose, (x,y)) < self.error_threshold:
                new_object_found = False
        
        if new_object_found:
            print("new object added to list")
            self.object_pose_list.append((x,y))
            print("object_list has " + str(len(self.object_pose_list)) + " objects")
            for object_pose in self.object_pose_list:
                print(object_pose)

    '''Publish list of pose (x, y) of detected objects as a PoseArray (Author: Tay)'''
    def publish_object_pose_list(self):
        if self.fsm == fsm.PUBLISH:
            # print("publish pose array")
            pose_array_msg = PoseArray()
            pose_array_msg.header.stamp = rospy.Time.now()
            pose_array_msg.header.frame_id = "odom"

            for object_pose in self.object_pose_list:
                pose = Pose()
                pose.position.x = object_pose[0]
                pose.position.y = object_pose[1]
                pose_array_msg.poses.append(pose)

            self._pose_array_pub.publish(pose_array_msg)
            self.fsm = fsm.WAITING_FOR_CAMERA

    '''compute distance between two points (each point is a (x, y) tuple'''
    def compute_distance(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    def get_color_info(self, index):
        if index == RED_INDEX:
            return (RED_TEXT, RED)
        elif index == BLUE_INDEX:
            return (BLUE_TEXT, BLUE)
        else: 
            return (GREEN_TEXT, GREEN)

    '''Control'''
    def spin(self):
        rate = rospy.Rate(FREQUENCY)
        print("entered spin")

        while not rospy.is_shutdown():            
            self.save_image()
            self.identify_colored_object()
            self.get_3d_coordinates()
            self.publish_object_pose_list()

        rate.sleep()

def main():
    rospy.init_node("camera")
    camera = Camera()
    rospy.sleep(5)

    try:
        camera.spin()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS node interrupted.")

if __name__ == "__main__":
    main()