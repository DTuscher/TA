import os
print(os.getcwd())
import matplotlib.pyplot as plt
import cv2
from realsense_sensor import RealsenseSensor
import imutils
import numpy as np
import cv_utils


cam = RealsenseSensor("realsense_config.json") 
cam.start()
img, d = cam.frames()



while True:
    img, d = cam.frames()
    
    graspPosition = cv_utils.calcGraspPoint(img, d, intrinsics=cam.getIntrinsics())
    if graspPosition:
        
        x,y,z = graspPosition
        print(x,y,z)

    
        
    k = cv2.waitKey(33)
    if k == 27:
        break