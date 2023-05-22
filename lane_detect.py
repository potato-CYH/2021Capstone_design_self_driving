import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


def roi(src):
    x = int(src.shape[1])
    y = int(src.shape[0])

    _shape = np.array(
        [[int(0.1*x), int(y)], [int(0.1*x), int(0.1*y)], [int(0.4*x), int(0.1*y)], [int(0.4*x), int(y)], [int(0.7*x), int(y)], [int(0.7*x), int(0.1*y)],[int(0.9*x), int(0.1*y)], [int(0.9*x), int(y)], [int(0.2*x), int(y)]])

    mask = np.zeros_like(src)

    if len(src.shape) > 2:
        channel_count = src.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, np.int32([_shape]), ignore_mask_color)
    masked_image = cv2.bitwise_and(src, mask)
    #cv2.imshow('mask', masked_image)
    return masked_image

def white_yellow_color_detect(src):
    
    hls = cv2.cvtColor(src, cv2.COLOR_BGR2HLS)

    lower = np.array([11,150, 10])
    upper = np.array([255, 255, 255])

    yellow_lower = np.array([0, 85, 81])
    yellow_upper = np.array([190, 255, 255])

    yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(hls, lower, upper)
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    masked = cv2.bitwise_and(src, src, mask = mask)

    #cv2.imshow('color_mask', masked)
    return masked


def Pers_transform(src):

    (h, w) = (src.shape[0], src.shape[1])

    #source = np.float32([[w // 2 -60, h * 0.53], [w // 2 + 60, h * 0.53], [w * 0.3, h], [w, h]])
    #destination = np.float32([[0, 0], [w-350, 0], [400, h], [w-150, h]])
    source = np.float32([[w // 2 -30, h * 0.53], [w // 2 + 60, h * 0.53], [w * 0.3, h], [w, h]])
    destination = np.float32([[0, 0], [w-350, 0], [400, h], [w-150, h]])
    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    minv = cv2.getPerspectiveTransform(destination, source)
    _image = cv2.warpPerspective(src, transform_matrix, (w, h))
    #cv2.imshow('test', _image)
    return _image, minv



def binaryzation(src):
    _gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(_gray, 60, 255, cv2.THRESH_BINARY)
    #cv2.imshow('bin', thresh)
    return ret, thresh
   
    

def plothistogram(src):
    #이미지 세로축 합(차선임을 판단할 수 있는 부분 선별)
    histogram = np.sum(src[src.shape[0]//2:,:], axis = 0)
    # 선별된 차선 영역의 중심값 구함
    midpoint = np.int32(histogram.shape[0]/2)
    #왼쪽 영역에서 흰색 픽셀이 가장 많이 누적된 x좌표 => 왼쪽 차선의 중앙좌표라고 생각할 수 있음
    leftbase = np.argmax(histogram[:midpoint])
    #오른쪽 영역에서 흰색 픽셀이 가장 많이 누적된 x좌표 => 오른쪽 차선의 중앙좌표라고 생각할 수 있음
    rightbase = np.argmax(histogram[midpoint:])+ midpoint
    #plt.plot(histogram, label='histogram')
    #plt.vlines(midpoint, 0, max(histogram), colors='red', linestyle='solid', linewidth=3, label='midpoint')
    #plt.vlines(leftbase, 0, max(histogram), colors='green', linestyle='--', linewidth=2, label='leftbase')
    #plt.vlines(rightbase, 0, max(histogram), colors='orange', linestyle='--', linewidth=2, label='rightbase')
    #plt.legend(loc='best', ncol=3)
    #plt.show()
    return leftbase, rightbase


def slide_window_search(src, left_current, right_current):
    out_img = np.dstack((src, src, src))
    #나눌 window 개수
    nwindows = 24
    window_height = np.int32(src.shape[0] / nwindows)
    nonzero = src.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    margin = 100
    minpix = 50
    left_lane = []
    right_lane = []
    
    #window 그리기용 선 설정
    color = [0, 255, 0]
    thickness = 2

    #슬라이딩 윈도우 시작
    for w in range(nwindows):

        #윈도우 아랫부분
        win_y_low = src.shape[0] - (w+1)*window_height
        #윈도우 윗부분
        win_y_high = src.shape[0] - w*window_height
        #왼쪽 차선 좌, 우
        win_xleft_low = left_current - margin
        win_xleft_high = left_current + margin
        
        #오른쪽 차선 좌, 우
        win_xright_low = right_current - margin
        win_xright_high = right_current + margin

        #윈도우 그리기
        cv2.rectangle(out_img,(win_xleft_low, win_y_low),(win_xleft_high, win_y_high),color, thickness)
        cv2.rectangle(out_img,(win_xright_low, win_y_low),(win_xright_high, win_y_high),color,thickness)
        
        #왼쪽 , 오른쪽 차선 별로 차선이 감지되는 영역 추출
        good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high)&(nonzero_x >= win_xleft_low)&(nonzero_x < win_xleft_high)).nonzero()[0]
        good_right = ((nonzero_y >= win_y_low)&(nonzero_y < win_y_high)&(nonzero_x >= win_xright_low)&(nonzero_x < win_xright_high)).nonzero()[0]
        left_lane.append(good_left)
        right_lane.append(good_right)
        
        # 각 윈도우마다 감지된 현재 차선 위치 갱신
        if len(good_left)>minpix:
            left_current = np.int32(np.mean(nonzero_x[good_left]))
        if len(good_right) > minpix:
            right_current = np.int32(np.mean(nonzero_x[good_right]))

    # 차선 배열을 1차원 변환
    left_lane = np.concatenate(left_lane)
    right_lane = np.concatenate(right_lane)

    leftx =nonzero_x[left_lane]
    lefty = nonzero_y[left_lane]
    rightx = nonzero_x[right_lane]
    righty = nonzero_y[right_lane]

    # 윈도우 내 감지된 영역듪로부터 직선 검출
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    #직선을 부드럽게 하기
    ploty = np.linspace(0, src.shape[0] - 1, src.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    #소수점 제거
    ltx = np.trunc(left_fitx)
    rtx = np.trunc(right_fitx)

    out_img[nonzero_y[left_lane], nonzero_x[left_lane]] = [255, 0, 0]
    out_img[nonzero_y[right_lane], nonzero_x[right_lane]] = [0, 0, 255]

    #cv2.imshow('sliding windows', out_img)
    #plt.imshow(out_img)
    #plt.plot(left_fitx,ploty, color = 'yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)
    #plt.show()

    ret=[ltx, rtx, ploty]

    return ret



def draw_lane_line(original_src, wrapped_src, minv, draw_info):
    left_fitx = draw_info[0]
    right_fitx = draw_info[1]
    ploty = draw_info[2]

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

    cv2.fillPoly(color_wrap, np.int_([pts]), (0, 255, 255))
    cv2.fillPoly(color_wrap, np.int_([pts_mean]), (0, 255, 255))
    cv2.line(color_wrap, (np.int_(first_point_x), np.int_(first_point_y)),(np.int_(last_point_x), np.int_(last_point_y)) ,(0, 255,0),4)
    #cv2.line(color_wrap, (np.int_(criteria_first_point_x),np.int_(criteria_first_point_y)),(np.int_(criteria_last_point_x),np.int_(criteria_last_point_y)),(0, 0, 255), 4)
    #cv2.putText(original_src, "curv = "+str(2*theta), (np.int_((original_src.shape[1])*(1/2)), np.int_(original_src.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2, cv2.LINE_AA)
    newwarp = cv2.warpPerspective((color_wrap), minv, (original_src.shape[1], original_src.shape[0]))
    result = cv2.addWeighted((original_src), 1, newwarp, 0.5, 0)
    
    print(first_point_x, first_point_y)
    print(last_point_x, last_point_y)
    print('criteria_fisrt'+str(criteria_first_point_x) + ',' +str(criteria_first_point_y))
    print(criteria_last_point_x, criteria_last_point_y)
    angle = 2*theta
    #cv2.imshow('color warp', color_wrap)

    return pts_mean, result, angle


cam = cv2.VideoCapture('vod.mp4')

while(True):
    check, frame = cam.read()

    try:
        #cv2.imshow('rr',frame)
        persp, rev_pers_mat = Pers_transform(frame)  #pers_transform(src)[0] : return perspected image , pers_transform(src)[1] = return reverse perspection matrix
        test=white_yellow_color_detect(persp)
        roied = roi(test)
        bin_img = binaryzation(roied)[1]  #binarization(src)[1] : return the binarized image
        leftbase, rightbase = plothistogram(bin_img)
        ret = slide_window_search(bin_img, leftbase, rightbase)
        pts_mean, result, angle = draw_lane_line(frame, bin_img, rev_pers_mat , ret)
        cv2.imshow('result', result)
    except:
        continue

    if cv2.waitKey(1)&0xFF == 27:
        break

cv2.waitKey(0)


cv2.destroyAllWindows()