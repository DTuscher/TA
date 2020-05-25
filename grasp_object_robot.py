import os
print(os.getcwd())
import matplotlib.pyplot as plt
import cv2
from realsense_sensor import RealsenseSensor
import imutils
import numpy as np
import cv_utils
import sys
import urx

from scipy.spatial.transform import Rotation as R

import time



cam = RealsenseSensor("realsense_config.json") 
cam.start()
img, d = cam.frames()

print("Verbinde zu roboter")

rob = urx.Robot("192.168.0.100")

# geschwindigkeit
v = 0.05

# beschleunigung
a = 0.3

rob.set_tcp((0, 0, 0.1, 0, 0, 0))
rob.set_payload(2, (0, 0, 0.1))

# Kameratransformation 
t = [0.5, 0.5, 0.5]              # x, y, z
r_quat = [0, 0, 0.707, 0.707]    # x, y, z, w
rot = R.from_quat(r_quat)

# als rotationsmatrix
r_mat = rot.as_matrix()

# observer pose
q_home = [0, 0, 0, 0, 0, 0]

# ablegeposition
bin_pos = np.array([0.5, 0.5, 0.5])

def close_gripper():
    rob.set_digital_out(0, True)

def open_gripper():
    rob.set_digital_out(0, False)

def get_pose(pos, orientation=[0, 0, 0.707, 0.707]):
    if not isinstance(pos, list):
        pos = pos.tolist()
    return pos + orientation

# go to home !!!!!!!!!!!!!!!! HARDCODED VEL AND ACC !!!!!!
rob.movej(q_home, acc=0.01, vel=0.01)


object_type = input("Welches Objekt willst du greifen? r: Rechteck, c: Kreis, a: beliebig   ")

img, d = cam.frames()

# check which object to be grasped
if object_type == "r":
    grasp_point = cv_utils.find_rects_in_image(img, d, intrinsics=cam.getIntrinsics())
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

    answer = input("Willst du das Objekt wirklich greifen? y/n   ")
    if answer == "n":
        sys.exit()

    # pose von position holen mit fester orientierung
    pre_grasp_pose = get_pose(pre_grasp_pos)

    # pre-grasp pose anfahren mit normaler geschwindigkeit
    rob.movel(pre_grasp_pose, acc=a, vel=v)
    
    # pose von position holen mit fester orientierung
    grasp_pose = get_pose(grasp_pos_world)

    # grasp pose anfahren mit HALBER beschleunigung und HALBER geschwindigkeit
    rob.movel(grasp_pose, acc=a/2, vel=v/2)

    # greifer schliessen
    close_gripper()

    # warten
    time.sleep(2)

    # bin pose mit fester orientierung berechnung
    bin_pose = get_pose(bin_pos)

    # zu ablegestelle fahren mit normaler geschwindigkeit
    rob.movel(bin_pose, acc=a, vel=v)

    # greifer öffnen
    open_gripper()

    # warten
    time.sleep(2)

    # go to home !!!!!!!!!!!!!!!! HARDCODED VEL AND ACC !!!!!!
    rob.movej(q_home, acc=0.01, vel=0.01)