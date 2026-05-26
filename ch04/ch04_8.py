import numpy as np 
import cv2 as cv2 
from matplotlib import pyplot as plt 


img = cv2.imread('kpop.jpg',cv2.IMREAD_GRAYSCALE)

blur_gray=cv2.GaussianBlur(img,(3,3),0)

th1 = 50
th2 = 150
edges=cv2.Canny(blur_gray, th1, th2)
cv2.imwrite('canny.jpg', edges)

erotion=cv2.erode(edges, (5, 5), iterations=1) 
dilation=cv2.dilate(erotion, (5,5), iterations=1) 
cv2.imwrite('erode_dilate.jpg', dilation)


dilation=cv2.dilate(edges, (5, 5), iterations=1) 
erotion=cv2.erode(dilation, (5,5), iterations=1) 
cv2.imwrite('dilate_erode.jpg', erotion)
