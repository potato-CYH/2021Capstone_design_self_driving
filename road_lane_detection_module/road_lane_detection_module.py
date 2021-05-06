
import cv2
import numpy as np
import matplotlib as plt

def white_yellow_color_detect(src):
    #HSV color detection for yellow 
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    #mask generation for white
    mask_white = cv2.inRange(src, (220, 220, 220),(255,255,255))
    #mask generation for yellow
    mask_yellow = cv2.inRange(hsv, (18, 94, 140),(48, 255, 255))

    #Bitwhie and mask with original image color
    filter_white = cv2.bitwise_and(src, src, mask=mask_white)
    filter_yellow = cv2.bitwise_and(src, src, mask=mask_yellow)
    
    #add white and yellow filter in the filtered numpy array
    filtered = cv2.bitwise_or(filter_white, filter_yellow)

    return filtered

def ROI_setting(src):
    roi_mask = np.zeros_like(src) #generate the empty image with size of function source, used for ROI setting
    #get width(x) and height(y) of image(src) - tuple
    (y, x) = (src.shape[0], src.shape[1]) 
    #set roi shape position
    pts = np.array([[(x/6,y),(x*(5/6),y),(x*(2/3),y*(2/3)),(x/3,y*(2/3))]], dtype=np.int32)
    cv2.fillPoly(roi_mask, pts, (255,255,255))

    #overlap source and roi mask
    mul = cv2.bitwise_and(src, roi_mask)

    return mul


img = cv2.imread('C:/Users/Administrator/Desktop/Self-driving-toy-car-using-ML-CV-/road_lane_detection_module/test1.jpg')
test=white_yellow_color_detect(img)
t=ROI_setting(test)
s=cv2.Canny(t, 50, 260)
dst = cv2.cornerHarris(s, 2, 3, 0.04)
cv2.imshow('test', s)
cv2.waitKey(0)
cv2.destroyAllWindows()