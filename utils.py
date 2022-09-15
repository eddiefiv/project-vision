import cv2 as cv
import numpy as np

def order_points(pts):
    # Initiate array of coordinates 
    # [1   2]
    # [3   4]

    rect = np.zeros((4, 2), dtype='float32')

    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    # Obtain a consistent order of the points and unpack then individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Determine the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + (((br[1] - bl[1]) ** 2)))
    
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + (((tr[1] - tl[1]) ** 2)))
    maxWidth = max(int(widthA), int(widthB))
    
    # Determine the height of the new image

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + (((tr[1] - br[1]) ** 2)))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + (((tl[1] - bl[1]) ** 2)))
    maxHeight = max(int(heightA), int(heightB))

    # Construct destination points; birds eye view coordinates
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]],
        dtype='float32')

    # Apply the perspective transform
    n = cv.getPerspectiveTransform(rect, dst)
    trans = cv.warpPerspective(image, n, (maxWidth, maxHeight))

    # Return
    return trans