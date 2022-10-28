import cv2 as cv
import numpy as np
import argparse
import utils
import time

capture = cv.VideoCapture('Videos/Clean Driving.mp4')

# Init argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = 'path to the image file')
ap.add_argument("-c", "--coords", help = 'comma seperated list of source points')
args = vars(ap.parse_args())

def mouse_click(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x, ', ', y)

new_frame_time = 0
prev_frame_time = 0

font = cv.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = capture.read()

    draw = frame.copy()

    #
    # Perspective transform on ROI
    #

    label = cv.putText(draw, "ROI Border", (224, 200), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0, 0, 255), 1)

    # 
    c1 = cv.circle(draw, (324, 211), 5, (0, 0, 255), -1)
    c2 = cv.circle(draw, (449, 211), 5, (0, 0, 255), -1)
    c3 = cv.circle(draw, (166, 358), 5, (0, 0, 255), -1)
    c4 = cv.circle(draw, (627, 358), 5, (0, 0, 255), -1)

    tlr = cv.line(draw, (324, 211), (449, 211), (0, 0, 255), 1)
    tb = cv.line(draw, (449, 211), (627, 358), (0, 0, 255), 1)
    brl = cv.line(draw, (627, 358), (166, 358), (0, 0, 255), 1)
    bt = cv.line(draw, (166, 358), (324, 211), (0, 0, 255), 1)

    pts = np.array([
        [324, 211],
        [449, 211],
        [166, 358],
        [627, 358]
    ], dtype = 'float32')

    dstpts = np.array([
        [0, 0],
        [449, 0],
        [0, 358],
        [449, 358]
    ], dtype = 'float32')

    matrix = cv.getPerspectiveTransform(pts, dstpts)
    trans = cv.warpPerspective(frame, matrix, (449, 358))
    #trans = utils.four_point_transform(frame, pts)

    # Define ROI
    gx1, gx2, gy1, gy2 = 166, 627, 211, 358
    roi = frame[gy1:gy2, gx1:gx2]

    bounding = cv.rectangle(frame, (166, 211), (627, 358), (255, 255, 255), 2)

    #
    # Grayscale frames in ROI
    #
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

    #
    # BGR to HSV
    #
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    #
    # Gaussian blur the grayscale
    #
    blur = cv.GaussianBlur(gray, (5, 5), 0)

    #
    # Get white colors in frame from mask
    #
    mask1 = cv.inRange(trans, (140, 140, 140), (255, 255, 255))

    #
    # Canny detection
    #
    canny = cv.Canny(blur, 50, 150)
    #cv.erode(canny, (2, 2), canny)

    #
    # Line detection using canny
    #
    line_disp = np.copy(frame) * 0

    lines = cv.HoughLinesP(canny, 1, ((np.pi) / 180), 15, np.array([]), 50, 20)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv.line(line_disp,(x1+gx1,y1+gy1),(x2+gx2,y2+gy2),(255,0,0), 5)

    #
    # Binary thresholding
    #
    kernel_size = 3
    ret, thresh = cv.threshold(canny, 150, 255, cv.THRESH_BINARY)

    #
    # Sobel testing
    #
    sobel = cv.Sobel(thresh, cv.CV_64F, 0, 1, 3)

    #
    # Contour detection
    #
    cnts = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for i in range(len(cnts)):
        cnt = cnts[i]
        if (cv.contourArea(cnt) > 75):
            (x,y,w,h) = cv.boundingRect(cnt)
            cv.rectangle(frame, (x+gx1,y+gy1), ((x+gx1)+w, (y+gy1)+h), (0,0,255), 2)
            cv.putText(frame, "Lane Line", (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    cv.drawContours(frame, cnts, -1, (0, 255, 0), 1)

    lines_edge = cv.addWeighted(frame, 0.8, frame, 1, 0)

    #
    # Frame deltatime calculation
    #
    new_frame_time = time.time()

    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    fps = int(fps)

    fps = str(fps)

    cv.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv.LINE_AA)

    cv.imshow('Frame', frame)
    #cv.imshow('Draw', draw)
    #cv.imshow('Mask', mask1)
    #cv.imshow('Gray', gray)
    #cv.imshow('HSV', hsv)
    #cv.imshow('Blur', blur)
    #cv.imshow('Canny', canny)
    #cv.imshow('Perspective Transformed', trans)
    #cv.imshow('Lines', lines_edge)
    #cv.imshow('ROI', roi)
    #cv.imshow('Bin Thresh', thresh)
    #cv.imshow('Sobel', sobel)

    cv.setMouseCallback('Frame', mouse_click)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()