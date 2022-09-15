import imutils
import cv2 as cv
import numpy as np

img = cv.imread('Photos/Lines.jpg')
#cv.imshow('Initial', img)

blank = np.zeros(img.shape, dtype='uint8')

gray = cv.cvtColor(img,  cv.COLOR_BGR2GRAY)

canny = cv.Canny(gray, 50, 100)

ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
#cv.imshow('Thresh', thresh)

items = cv.findContours(canny.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
contours = items[0] if len(items) == 2 else items[1]
cnts = imutils.grab_contours(items)
mask = np.ones(img.shape[:2], dtype='uint8') * 255
print(f'{len(contours)} contour(s) found!')

cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
#cv.imshow('Contours Drawn', blank)

def is_contour_bad(c):
	# approximate the contour
	peri = cv.arcLength(c, True)
	approx = cv.approxPolyDP(c, 0.02 * peri, True)

	# the contour is 'bad' if it is not a rectangle
	return not len(approx) == 4

# Loop over contours
for c in cnts:
    if is_contour_bad(c):
        cv.drawContours(mask, [c], -1, 0 ,-1)

# Remove bad contours
img = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Mask', mask)
cv.imshow('After', img)

cv.waitKey(0)