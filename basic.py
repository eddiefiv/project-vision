from turtle import resizemode
import cv2 as cv

img = cv.imread('Photos/car1.jpg')
lines = cv.imread('Photos/Lines.jpg')
#cv.imshow('Car', img)

# Converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray', gray)

# Blur
blur = cv.GaussianBlur(lines, (7, 7), cv.BORDER_DEFAULT)
#cv.imshow('Blur', blur)

# Edge cascade
canny = cv.Canny(blur, 175, 125)
cv.imshow('Canny Image', canny)

# Dilating the image
dilated = cv.dilate(canny, (3, 3), iterations = 2)
cv.imshow('Dilated', dilated)

cv.waitKey(0)