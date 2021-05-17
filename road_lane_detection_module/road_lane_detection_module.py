import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

roi_left_bottom = (0,889)
roi_left_top = (274, 0)
roi_right_bottom = (1570, 889)
roi_right_top =  (1894,0)

persp_left_bottom = (400,889)
persp_right_bottom = (1338,889)
persp_right_top = (956,650)
persp_left_top =  (850,655)

def roi_white_yellow_color_detect(src):
    #HSV color detection for yellow 
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    roi_mask = np.zeros_like(src) #generate the empty image with size of function source, used for ROI setting
    #get width(x) and height(y) of image(src) - tuple
    (y, x) = (src.shape[0], src.shape[1]) 
    #set roi shape position
    #pts = np.array([[(0,y),(x,y),((2.5/6)*x,y*(1/2)),((3.5/6)*x,y*(1/2))]], dtype=np.int32)
    cv2.rectangle(roi_mask, roi_left_bottom,roi_left_top, (255,255,255), -1)
    cv2.rectangle(roi_mask, roi_right_bottom,roi_right_top,(255,255,255), -1)

    #points for perspective transform
    #pts1 = np.array([[0,y],[x,y],[(2.5/6)*x,y/2],[(3.5/6)*x,y/2]], dtype = np.float32)
    #overlap source and roi mask
    mul = cv2.bitwise_and(src, roi_mask)
    #mask generation for white
    mask_white = cv2.inRange(mul, (140, 140, 140),(255,255,255))
    #mask generation for yellow
    mask_yellow = cv2.inRange(mul, (18, 94, 140),(48, 255, 255))

    #Bitwhie and mask with original image color
    filter_white = cv2.bitwise_and(mul, mul, mask=mask_white)
    filter_yellow = cv2.bitwise_and(mul, mul, mask=mask_yellow)
    
    #add white and yellow filter in the filtered numpy array
    filtered = cv2.bitwise_or(filter_white, filter_yellow)
    #cv2.imshow('tiltered1',mul)
    #cv2.imshow('tiltered',filtered)
    return filtered



def Pers_transform(src):
  
    pts1 = np.array([[persp_left_bottom,persp_right_bottom,persp_right_top,persp_left_top]], dtype=np.float32)
    pts2 = np.array([[0,src.shape[0]],[src.shape[1],src.shape[0]],[src.shape[1],0],[0,0]],dtype=np.float32)
    #overlap source and roi mask
    
    perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)
    rev_pers_mat = cv2.getPerspectiveTransform(pts2, pts1)
    
    #image perspection
    persp = cv2.warpPerspective(src, perspective_matrix, (src.shape[1],src.shape[0]))
   
    #cv2.imshow('test2',thresh)
    #cv2.imshow('persp', persp)
    return persp, rev_pers_mat



def binaryzation(src):
    _gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(_gray, 60, 255, cv2.THRESH_BINARY)

    return ret, thresh
   
    

def plothistogram(src):
    histogram = np.sum(src[src.shape[0]//2:,:], axis = 0)
    midpoint = np.int32(histogram.shape[0]/2)
    leftbase = np.argmax(histogram[:midpoint])
    rightbase = np.argmax(histogram[midpoint:])+ midpoint

    return leftbase, rightbase



#slide_window_search 알고리즘 출처 : https://moon-coco.tistory.com/entry/OpenCV%EC%B0%A8%EC%84%A0-%EC%9D%B8%EC%8B%9D
def slide_window_search(src, left_current, right_current):
    out_img = np.dstack((src, src, src))

    nwindows = 8
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
        
        cv2.imshow('sliding windows', out_img)

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

    first_point_x = pts_mean[0][0][0]
    first_point_y = pts_mean[0][0][1]
    last_point_x = pts_mean[0][len(pts_mean[0])-1][0]
    last_point_y = pts_mean[0][len(pts_mean[0])-1][1]
    #len(pts_mean[0]

    criteria_first_point_x = original_src.shape[1]/2
    criteria_first_point_y = 0
    criteria_last_point_x = original_src.shape[1]/2
    criteria_last_point_y = original_src.shape[0]
    vector_ax = last_point_x - first_point_y
    vector_ay = last_point_y - first_point_y
    vector_bx = criteria_last_point_x - criteria_first_point_x
    vector_by = criteria_last_point_y - criteria_first_point_y
    tant = vector_ay/vector_ax

    vector_a = np.array([vector_ax, vector_ay])
    vector_b = np.array([vector_bx, vector_by])

    vector_a_scale = math.sqrt(math.pow(vector_ax, 2)+math.pow(vector_ay,2))
    vector_b_scale = math.sqrt(math.pow(vector_bx, 2)+math.pow(vector_by,2))
    vector_dot = np.dot(vector_a, vector_b)

    cosine = vector_dot/(vector_a_scale*vector_b_scale)
    theta = math.acos(cosine)

    cv2.fillPoly(color_wrap, np.int_([pts]), (255, 0, 0))
    cv2.fillPoly(color_wrap, np.int_([pts_mean]), (255, 0, 0))
    cv2.line(color_wrap, (np.int_(first_point_x), np.int_(first_point_y)),(np.int_(last_point_x), np.int_(last_point_y)) ,(0, 255,0),4)
    cv2.line(color_wrap, (np.int_(criteria_first_point_x),np.int_(criteria_first_point_y)),(np.int_(criteria_last_point_x),np.int_(criteria_last_point_y)),(0, 0, 255), 4)
    cv2.putText(original_src, "curv = "+str(2*theta), (np.int_((original_src.shape[1])*(1/2)), np.int_(original_src.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2, cv2.LINE_AA)
    newwarp = cv2.warpPerspective((color_wrap), minv, (original_src.shape[1], original_src.shape[0]))
    result = cv2.addWeighted((original_src), 1, newwarp, 0.5, 0)
    
    print(first_point_x, first_point_y)
    print(last_point_x, last_point_y)
    print('criteria_fisrt'+str(criteria_first_point_x) + ',' +str(criteria_first_point_y))
    print(criteria_last_point_x, criteria_last_point_y)
    angle = 2*theta
    #cv2.imshow('color warp', color_wrap)

    return pts_mean, result, angle


cap = cv2.VideoCapture('C:/Users/Administrator/Desktop/Self-driving-toy-car-using-ML-CV-/road_lane_detection_module/vod_test.mp4')

while(True):
    con, vod = cap.read()

    if con == False:
        continue

    try:
        persp, rev_pers_mat = Pers_transform(vod)  #pers_transform(src)[0] : return perspected image , pers_transform(src)[1] = return reverse perspection matrix
        test=roi_white_yellow_color_detect(persp)
        bin_img = binaryzation(test)[1]  #binarization(src)[1] : return the binarized image
        leftbase, rightbase = plothistogram(bin_img)
        ret = slide_window_search(bin_img, leftbase, rightbase)
        pts_mean, result, angle = draw_lane_line(vod, bin_img, rev_pers_mat , ret)
        cv2.imshow('result', result)
    except:
        continue

    if cv2.waitKey(1)&0xFF == 27:
        break

cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
