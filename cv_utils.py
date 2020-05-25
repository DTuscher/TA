import cv2 
import numpy as np
import imutils
from realsense_sensor import RealsenseSensor

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
            #c = max(cnts, key=cv2.concv2.imshow("hull", mask)tourArea)
            ((x,y), r) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center=(int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
            centers.append((x,y))
        return centers, cnts

def findCirclesInMask(img, d, intrinsics):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_c = cv2.medianBlur(gray,	5)
    circles	= cv2.HoughCircles(img_c,cv2.HOUGH_GRADIENT,1,120,param1=100,param2=30,minRadius=10,maxRadius=20)
    
 
    if circles is not None:
        
        circles = np.round(circles[0, :]).astype("int")
        
        for (x, y, r) in circles:

            cv2.circle(img, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                        
            depthVal = calcDepth(d, int(y), int(x))
            
            c_x = -depthVal* (x - intrinsics["px"]) / intrinsics["fx"]
            c_y = depthVal * (y - intrinsics["py"]) / intrinsics["fy"]
            c_z = depthVal
            
            cv2.imshow("Depth", d)
            

            cv2.imshow("output", img[:, :, ::-1])
            return c_x, c_y, c_z
            
    return 0, 0 ,0


def plotCircleAroundCenter(img, x, y, color=(255, 0, 0)):
    img = cv2.circle(img,(int(x),int(y)),2,color,3)
    return img

def plotBoundingRect(img, c):
    rect = cv2.boundingRect(c)
    x,y,w,h = rect
    return cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

def plotBoundingRect1 (img, c):
    rect = cv2.boundingRect(c)
    x,y,w,h = rect
    return cv2.rectangle(img,(x,y),(x+w,y+h),(-1,255,3),2)


def find_rects_in_image(img,d,intrinsics):
    thresh = preprocess_img(img)

    #contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	    cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    greatest_rect = None
    max_circ = None
    max_v = 0
    max_cont = None
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int((M["m10"] / M["m00"]))
        cY = int((M["m01"] / M["m00"]))
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        print(approx)
        if len(approx) == 4:
            rect = cv2.boundingRect(approx)
            x, y, w, h = rect
            circ = cv2.minEnclosingCircle(approx)
            max_cont = approx
            print(rect)
            
            v = w * h
            print(v)
            if v > (img.shape[0] * img.shape[1]) / 2:
                continue
            elif v > max_v:
                greatest_rect = rect
                max_v = v
                max_circ = circ
        
    if greatest_rect is not None:
        x, y, w, h = greatest_rect

        (x, y), r = max_circ
        plotCircleAroundCenter(img, x, y)
            
        depthVal = calcDepth(d, int(y), int(x))

        c_x = -depthVal* (x - intrinsics["px"]) / intrinsics["fx"]
        c_y = depthVal * (y - intrinsics["py"]) / intrinsics["fy"]

        c_z = depthVal

        cv2.drawContours(img,[max_cont],-1, (0, 255, 0), 2)

        cv2.imshow("Depth", d)
        cv2.imshow("output", img[:, :, ::-1])
        return c_x, c_y, c_z




def preprocess_img(img):
    
    cv2.imshow("img", img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    blur_img = cv2.GaussianBlur(img.copy(), (5,5), 5)
    edged = cv2.Canny(blur_img, 80, 150)
    edged = cv2.dilate(edged, None, iterations=2)
    blurred = cv2.GaussianBlur(edged, (5, 5), 20)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    
    cv2.imshow("gray", gray )
    cv2.imshow("thresh", thresh)
    
    cv2.imshow("blurred", blurred)
    cv2.imshow("blur_img", blur_img)
    cv2.imshow("edged", edged)

    return thresh



def calcGraspPointContours(img, d, intrinsics):
    thresh = preprocess_img(img)
    rects = []
    out_cnts = []
    
    #contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	    cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if cnts:

        max_contour_all = max(cnts, key=cv2.contourArea)
        mask = np.zeros(img.shape[:2])
        cv2.drawContours(mask, max_contour_all, -1, (255), 1)
        
        cv2.imshow("biggest_cont", mask)

        ((x,y), r) = cv2.minEnclosingCircle(max_contour_all)
        
        depthVal = calcDepth(d, int(y), int(x))
        
        img = plotCircleAroundCenter(img, x, y)
        img = plotBoundingRect(img, max_contour_all)
        cv2.drawContours(img,cnts,-1, (0, 255, 0), 2)
        
        cv2.imshow("Depth", d)
        cv2.imshow("img", img[:, :, ::-1])

    

        c_x = -depthVal* (x - intrinsics["px"]) / intrinsics["fx"]
        c_y = depthVal * (y - intrinsics["py"])/ intrinsics["fy"]
        c_z = depthVal

    

        return c_x, c_y, c_z

    
