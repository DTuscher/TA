import os
print(os.getcwd())
import matplotlib.pyplot as plt
import cv2
from realsense_sensor import RealsenseSensor
import imutils
import numpy as np


def findContoursInMask(mask):
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnts = imutils.grab_contours(cnts)
        return cnts

def redMask(img_bgr):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(img_hsv, (0, 100, 100), (5, 255, 255))
    mask2 = cv2.inRange(img_hsv, (160, 100, 100), (179, 255, 255))
    mask = mask1 | mask2
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask

def greenMask(img_bgr):
    greenLower = (36,100,100)
    greenUpper = (86, 255, 255)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask


def calcDepth(d, u, v):
    range_x = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    range_y = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    range_x = np.arange(-3, 4, 1)
    range_y = np.arange(-3, 4, 1)
    sum_ranges = len(range_x) * len(range_y)
    cumulated_depth = 0
    for x in range_x:
        for y in range_y:
            if d.shape[0] > u+x and u+x > 0 and v+y > 0 and v+y < d.shape[1]:
                val = d[u + x][v + y]
                if np.isnan(val):
                    sum_ranges -= 1
                else:
                    cumulated_depth += val
            else:
                sum_ranges -= 1
    return cumulated_depth / (sum_ranges)


def findRectsInMask(mask):
    cnts = findContoursInMask(mask)
    centers = []
    if cnts:
        for c in cnts:
            #c = max(cnts, key=cv2.contourArea)
            ((x,y), r) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center=(int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
            centers.append((x,y))
        return centers, cnts

def findRectsInMas(mas):
    cnts = findContoursInMask(mas)
    centers = []
    if cnts:
        for c in cnts:
            #c = max(cnts, key=cv2.contourArea)
            ((x,y), r) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center=(int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
            centers.append((x,y))
        return centers, cnts

def plotCircleAroundCenter(img, x, y, color=(255, 0, 0)):
    img = cv2.circle(img,(int(x),int(y)),2,color,3)
    return img


def find_shapes_in_image(edged):
    rects = []
    out_cnts = []
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
 
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	    cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int((M["m10"] / M["m00"]))
        cY = int((M["m01"] / M["m00"]))
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:
            rect = cv2.boundingRect(approx)
            rects.append(rect)
            out_cnts.append(c)
    return rects, out_cnts
    

cam = RealsenseSensor("realsense_config.json") 
cam.start()
img, d = cam.frames()



while True:
    img, d = cam.frames()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) [55:800, 100:525]


    cv2.imshow("Gray", gray)
    mask_green = greenMask(img)
    mask_red = redMask(img)
    mask = mask_red | mask_green
    ret = findRectsInMask(mask)
    if ret is not None:
        centers, cnts = ret
        for c in centers:
            depthval = calcDepth(d, int(c[1]), int(c[0]))
            print(depthval)
            img = plotCircleAroundCenter(img, c[0], c[1])
            

        cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)

    #cv2.drawContours(d, d_rects, -1, (0, 255, 0), 2)

    cv2.imshow("RGB", img)
    cv2.imshow("Depth", d)
    edged = cv2.Canny(gray, 30, 150)
    cv2.imshow("Edged", edged)
    cv2.imshow("img", img)


        
        
        
    contourss = cv2.findContours

    
    
    for contour in contourss:
        approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
        cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5
        if len(approx) == 3:
            cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        elif len(approx) == 4:
            x1 ,y1, w, h = cv2.boundingRect(approx)
            aspectRatio = float(w)/h
            print(aspectRatio)
            if aspectRatio >= 0.95 and aspectRatio <= 1.05:
                cv2.putText(img, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            else:
                cv2.putText(img, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        elif len(approx) == 5:
            cv2.putText(img, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        elif len(approx) == 10:
            cv2.putText(img, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        else:
            cv2.putText(img, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    k = cv2.waitKey(33)
    if k == 27:
        break