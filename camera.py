#!/usr/bin/env python

"""
Rosbot Camera Set Up

Authors: Tai Wan Kim, Kamakshi Moparthi​
Specific components are marked with their corresponding authors
"""

import rospy
import numpy as np
from PIL import Image as PILImage
from enum import Enum
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge

DEFAULT_CAMERA_TOPIC = '/camera/rgb/image_raw'
DEFAULT_CAMERA_DEPTH_TOPIC = '/camera/depth/image_raw'
# DEFAULT_RGB_DEPTH = '/camera/aligned_depth_to_color/image_raw'

FREQUENCY = 10

LOWER = 0
UPPER  = 1

RED_INDEX = 2
GREEN_INDEX = 1
BLUE_INDEX = 0

RED = (0,0,255) 
GREEN = (0,255,0)
BLUE = (255,0,0)
BLACK = (0,0,0)

RED_TEXT = 'red'
GREEN_TEXT = 'green'
BLUE_TEXT = 'blue'
BLACK_TEXT = 'black'

IMAGE_PATH = 'image.jpg'
OUTPUT_IMAGE_PATH = 'bounded.jpg'
PIXEL_COUNT = 1200

class fsm(Enum):
    WAITING_FOR_CAMERA = 0
    SAVE = 1

class Camera():
    def __init__(self):
        self._camera_sub = rospy.Subscriber(DEFAULT_CAMERA_TOPIC, Image, self._camera_callback)
        self.depthSub = rospy.Subscriber(DEFAULT_CAMERA_DEPTH_TOPIC,Image,self.depthCallback)
        
        self.image = None
        self.depthImage = None

        #Color lower and upper bounds for Blue, Green, and Red colors in the HSV representation
        self.threshold = np.array([[[100, 50, 50],[130, 255, 255]],[[40, 20, 50],[90, 255, 255]],[[0, 50, 50],[10, 255, 255]]])
        self.fsm = fsm.WAITING_FOR_CAMERA
        self.boundingBox = None
 
    """
    This method is used to convert the raw ROS image to a format 
    that could be used by the openCV packages and assigns it to the class variables

    Author: Kamakshi 
    """
    def depthCallback(self,msg):
        bridge = CvBridge()
        self.depthImage = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    """
    Author: Tay
    """
    def _camera_callback(self, msg):
        if self.fsm == fsm.WAITING_FOR_CAMERA:
            self.image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            print("camera image updated!")
            self.fsm = fsm.SAVE

    """
    Write stored camera image as a png file for demonstration

    Author: Tay
    """
    def save_image(self):
        print("save image:")
        img = PILImage.fromarray(np.uint8(self.image))
        img.save(IMAGE_PATH)
        #cv2.imwrite('image.jpg',self.image)

    """
    This method is repsonsible for returning the distace between the object and the robot.
    The depthImage has distance value from the surface to the robot. 

    The obstacles are spherical in shape and the the distance between 
    the robot is measured from the centre of the sphere and the robot.
    The distance between the centre of sphere to the robot is same as 
    the depth value at point where the sphere touches the ground.

    The point where the sphere touches the ground is (x+ w/2 , y+h)

    Returns: the distance between the robot and the centre of the spherical obstacle

    Author: Kamakshi 
    """
    def readDepthInfo(self):
        coordinate = (self.boundingBox[0]+int(self.boundingBox[2]/2),self.boundingBox[1] +self.boundingBox[3])
        distance  = self.depthImage[coordinate[0]][coordinate[1]]
        print("Robot<->sphere distance" ,distance)
        return distance
    
    """
    This method returns the information specific to a color using the predefined color codes or indexes.
    The returned value is used for masking the specific colored region in the image

    Input : 
                index : The color index . The inputs can either be RED_INDEX,
                            BLUE_INDEX or GREEN_INDEX,each corresponding to the value 2,0 or 1
    Returns :
                A tuple with the a string indicating the color and the color value 
                For example : For the input RED_INDEX , with value 2
                The returned value is  (RED_TEXT,RED) , i.e ('red',(0,0,255))

    Author:  Kamakshi
    """
    def getColorInfo(self,index):
        if index == RED_INDEX:
            return (RED_TEXT,RED)
        elif index == BLUE_INDEX:
            return (BLUE_TEXT,BLUE)
        else:
            return (GREEN_TEXT,GREEN)
    
    """
    This method is reponsible for drawing a bounding box around the object of a specific color.
    For the images the origin is at the top left corner of the image. 
    So the bounding box coordinates are stored in the boundingBox class variable, 
    with the lop left coordinates of the box with the width and height values
    This method is generic and draws a bounding box around 
    any object of a given specific color based on the specified color object

    Inputs:
                image : Original image
                hsvImage :  'image' converted to Hue , Saturation , Value representation
                colorCode : The specific colored around which we intend to draw the bounding box
    
    Returns:
                 image : the original image with the bounding box around the specific colored objects 

    Author: Kamakshi
    """
    def drawBoundingBox(self,image,hsvImage,colorCode):
        maskedImage = cv2.inRange(hsvImage, self.threshold[colorCode][LOWER],self.threshold[colorCode][UPPER])
        contours, _ = cv2.findContours(maskedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]    
        
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area > PIXEL_COUNT:
                x, y, w, h = cv2.boundingRect(contour)
                t = self.getColorInfo(colorCode)
                cv2.rectangle(image, (x, y), (x + w, y + h), t[1], 2)
                self.boundingBox = (x,y,w,h)
                cv2.putText(image, t[0], (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.9, t[1], 2)
        return image
 

    """
    This method reads the input image and converted it to HSV represenation.
    This method invokes the @drawBoundingBox method
    Finally stores the modified image.

    Author: Kamakshi
    """
    def identifyColoredObject(self):
        image  = cv2.imread(IMAGE_PATH)
        hsvImage = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        image = self.drawBoundingBox(image,hsvImage,RED_INDEX)

        #image = self.drawBoundingBox(image,hsvImage,GREEN_INDEX)
        #image = self.drawBoundingBox(image,hsvImage,BLUE_INDEX)

        #cv2.imshow('Boximage', image)
        #cv2.waitKey(0)
        cv2.imwrite(OUTPUT_IMAGE_PATH,image)

    
    """
    Control

    Author: Tay
    """
    def spin(self):
        rate = rospy.Rate(FREQUENCY)
        print("entered spin")

        while not rospy.is_shutdown():
            if self.fsm == fsm.SAVE:
                # print("save camera image of size " + str(self.image.shape))
                # self.save_image()

                self.identifyColoredObject()
                self.readDepthInfo()
                self.fsm = fsm.WAITING_FOR_CAMERA

        rate.sleep()

def main():
    """Main function."""
    rospy.init_node("camera")
    camera = Camera()
    # rospy.sleep(2)

    try:
        camera.spin()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS node interrupted.")

if __name__ == "__main__":
    """Run the main function."""
    main()