import os
print(os.getcwd())
import matplotlib.pyplot as plt
import cv2
from realsense_sensor import RealsenseSensor
import imutils
import numpy as np
import cv_utils
import sys

from scipy.spatial.transform import Rotation as R


cam = RealsenseSensor("realsense_config.json") 
cam.start()
img, d = cam.frames()
 
t = [0.5, 0.5, 0.5]              # x, y, z
r_quat = [0, 0, 0.707, 0.707]    # x, y, z, w

rot = R.from_quat(r_quat)

r_mat = rot.as_matrix()


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
    
    # griffpunkt 
    grasp_pos_cam = np.array([x,y,z])

    # berechne den griffpunkt in weltkoordinaten
    grasp_pos_world = t + r_mat @ grasp_pos_cam

    print("Griffpunkt in Welt: ",grasp_pos_world)
    # pre-grasp position

    pre_grasp_pos = grasp_pos_world

    # pre-grasp 10 cm über dem objekt
    pre_grasp_pos[2] += 0.1

    print("Pre-Grasp position: ",pre_grasp_pos)







    
