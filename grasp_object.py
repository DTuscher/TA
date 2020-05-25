import os
print(os.getcwd())
import matplotlib.pyplot as plt
import cv2
from realsense_sensor import RealsenseSensor
import imutils
import numpy as np
import cv_utils
import sys


cam = RealsenseSensor("realsense_config.json") 
cam.start()
img, d = cam.frames()


object_type = input("Welches Objekt willst du greifen? r: Rechteck, c: Kreis, a: beliebig   ")

img, d = cam.frames()

# check which object to be grasped
if object_type == "r":
    grasp_point = cv_utils.find_shapes_in_image(img, d, intrinsics=cam.getIntrinsics())
elif object_type == "c":
    grasp_point = cv_utils.findCirclesInMask(img, d, intrinsics=cam.getIntrinsics())
elif object_type == "a":
    grasp_point = cv_utils.calcGraspPointContours(img, d, intrinsics=cam.getIntrinsics())
else:
    print("Solche Objekte kann ich nicht.")
    sys.exit()
if grasp_point:
    x,y,z = grasp_point
    print("Griffpunkt in Kamerakoordinaten: ",x,y,z)
    answer = input("Willst du das Objekt wirklich greifen? y/n   ")
    if answer == "n":
        sys.exit()
