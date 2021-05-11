
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

def white_yellow_color_detect(src):
    #HSV color detection for yellow 
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    #mask generation for white
    mask_white = cv2.inRange(src, (140, 140, 140),(255,255,255))
    #mask generation for yellow
    mask_yellow = cv2.inRange(hsv, (18, 94, 140),(48, 255, 255))

    #Bitwhie and mask with original image color
    filter_white = cv2.bitwise_and(src, src, mask=mask_white)
    filter_yellow = cv2.bitwise_and(src, src, mask=mask_yellow)
    
    #add white and yellow filter in the filtered numpy array
    filtered = cv2.bitwise_or(filter_white, filter_yellow)

    return filtered

def ROI_Pers_transform(src):
    roi_mask = np.zeros_like(src) #generate the empty image with size of function source, used for ROI setting
    #get width(x) and height(y) of image(src) - tuple
    (y, x) = (src.shape[0], src.shape[1]) 
    #set roi shape position
    pts = np.array([[(x/10,y),(x*(9/10),y),(x*(7/10),y*(2/3)),(x*(3/10),y*(2/3))]], dtype=np.int32)
    cv2.fillPoly(roi_mask, pts, (255,255,255))

    #points for perspective transform
    pts1 = np.array([[x*(3/10),y*(2/3)],[x*(7/10),y*(2/3)],[x/10,y],[x*(9/10),y]], dtype = np.float32)
    pts2 = np.array([[0,0],[x,0],[0,y],[x,y]],dtype=np.float32)
    #overlap source and roi mask
    mul = cv2.bitwise_and(src, roi_mask)

    perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)
    rev_pers_mat = cv2.getPerspectiveTransform(pts2, pts1)

    persp = cv2.warpPerspective(mul, perspective_matrix, (x,y))
    _gray = cv2.cvtColor(persp, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(_gray, 100, 255, cv2.THRESH_BINARY)

    #cv2.imshow('test2',thresh)
    return ret, thresh, rev_pers_mat



def plothistogram(src):
    histogram = np.sum(src[src.shape[0]//2:,:], axis = 0)
    midpoint = np.int32(histogram.shape[0]/2)
    leftbase = np.argmax(histogram[:midpoint])
    rightbase = np.argmax(histogram[midpoint:])+ midpoint

    return leftbase, rightbase

#slide_window_search 알고리즘 출처 : https://moon-coco.tistory.com/entry/OpenCV%EC%B0%A8%EC%84%A0-%EC%9D%B8%EC%8B%9D
def slide_window_search(src, left_current, right_current):
    out_img = np.dstack((src, src, src))

    nwindows = 4
    window_height = np.int32(src.shape[0] / nwindows)
    nonzero = src.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    margin = 100
    minpix = 50
    left_lane = []
    right_lane = []
    color = [0, 255, 0]
    thickness = 2

    for w in range(nwindows):
        win_y_low = src.shape[0] - (w+1)*window_height
        win_y_high = src.shape[0] - w*window_height
        win_xleft_low = left_current - margin
        win_xleft_high = left_current + margin
        win_xright_low = right_current - margin
        win_xright_high = right_current + margin

        cv2.rectangle(out_img,(win_xleft_low, win_y_low),(win_xleft_high, win_y_high),color, thickness)
        cv2.rectangle(out_img,(win_xright_low, win_y_low),(win_xright_high, win_y_high),color,thickness)
        good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high)&(nonzero_x >= win_xleft_low)&(nonzero_x < win_xleft_high)).nonzero()[0]
        good_right = ((nonzero_y >= win_y_low)&(nonzero_y < win_y_high)&(nonzero_x >= win_xright_low)&(nonzero_x < win_xright_high)).nonzero()[0]
        left_lane.append(good_left)
        right_lane.append(good_right)
        
        #cv2.imshow('sliding windows', out_img)

        if len(good_left)>minpix:
            left_current = np.int32(np.mean(nonzero_x[good_left]))
        if len(good_right) > minpix:
            right_current = np.int32(np.mean(nonzero_x[good_right]))

        left_lane = np.concatenate(left_lane)
        right_lane = np.concatenate(right_lane)

        leftx =nonzero_x[left_lane]
        lefty = nonzero_y[left_lane]
        rightx = nonzero_x[right_lane]
        righty = nonzero_y[right_lane]

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, src.shape[0] - 1, src.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        ltx = np.trunc(left_fitx)
        rtx = np.trunc(right_fitx)

        out_img[nonzero_y[left_lane], nonzero_x[left_lane]] = [255, 0, 0]
        out_img[nonzero_y[right_lane], nonzero_x[right_lane]] = [0, 0, 255]

        #plt.imshow(out_img)
        #plt.plot(left_fitx,ploty, color = 'yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        #plt.xlim(0, 1280)
        #plt.ylim(720, 0)
        #plt.show()

        ret={'left_fitx' : ltx, 'right_fitx' : rtx, 'ploty':ploty}

        return ret


def draw_lane_line(original_src, wrapped_src, minv, draw_info):
    left_fitx = draw_info['left_fitx']
    right_fitx = draw_info['right_fitx']
    ploty = draw_info['ploty']

    wrap_zero = np.zeros_like(wrapped_src).astype(np.uint8)
    color_wrap = np.dstack((wrap_zero, wrap_zero, wrap_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    mean_x = np.mean((left_fitx,right_fitx),axis = 0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

    first_point_x = np.int_(pts_mean[0][0][0])
    first_point_y = np.int_(pts_mean[0][0][1])
    last_point_x = np.int_(pts_mean[0][len(pts_mean[0])-1][0])
    last_point_y = np.int_(pts_mean[0][len(pts_mean[0])-1][1])
    criteria_first_point_x = np.int_(original_src.shape[1]/2)
    criteria_first_point_y = 0
    criteria_last_point_x = np.int_(original_src.shape[1]/2)
    criteria_last_point_y = first_point_y

    vector_ax = last_point_x - first_point_y
    vector_ay = last_point_y - first_point_y
    vector_bx = criteria_last_point_x - criteria_first_point_x
    vector_by = criteria_last_point_y - criteria_first_point_y

    vector_a = np.array([vector_ax, vector_ay])
    vector_b = np.array([vector_bx, vector_by])

    vector_a_scale = math.sqrt(math.pow(vector_ax, 2)+math.pow(vector_ay,2))
    vector_b_scale = math.sqrt(math.pow(vector_bx, 2)+math.pow(vector_by,2))
    vector_dot = np.dot(vector_a, vector_b)

    cosine = vector_dot/(vector_a_scale*vector_b_scale)
    theta = math.acos(cosine)

    cv2.fillPoly(color_wrap, np.int_([pts]), (0, 255, 0))
    cv2.fillPoly(color_wrap, np.int_([pts_mean]), (0, 255, 0))
    cv2.line(color_wrap, (first_point_x, first_point_y),(last_point_x,last_point_y) ,(0, 255,255),4)
    cv2.line(color_wrap, (criteria_first_point_x,criteria_first_point_y),(criteria_last_point_x,criteria_last_point_y),(0, 0, 255), 4)
    cv2.putText(color_wrap, "curv = "+str(2*theta), (np.int_((original_src.shape[1])*(1/2)), np.int_(original_src.shape[0]-50)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,0), 2, cv2.LINE_AA)
    newwarp = cv2.warpPerspective((color_wrap), minv, (original_src.shape[1], original_src.shape[0]))
    result = cv2.addWeighted((original_src), 1, newwarp, 0.5, 0)
    
    angle = 2*theta

    return pts_mean, result, angle


img = cv2.imread('C:/Users/Administrator/Desktop/Self-driving-toy-car-using-ML-CV-/road_lane_detection_module/road1.jpg')
test=white_yellow_color_detect(img)
t, k, minv=ROI_Pers_transform(test)
leftbase, rightbase = plothistogram(k)
draw_line = slide_window_search(k, leftbase, rightbase)
pts_mean, result, angle = draw_lane_line(img, k, minv, draw_line)
cv2.imshow('test',result)
cv2.waitKey(0)
cv2.destroyAllWindows()

