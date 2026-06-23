import cv2
import numpy as np

img1 = cv2.imread('t1.jpg',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('t2.jpg',cv2.IMREAD_GRAYSCALE)

diff = cv2.absdiff(img1,img2)
ret, binary = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
result=cv2.dilate(binary, None, iterations=3)

cv2.imshow('time_1', img1)
cv2.imshow('time_2', img2)
cv2.imshow('absdiff result', binary )
cv2.imshow('dilation result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()