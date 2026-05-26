import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt


img = cv2.imread('kpop.jpg',0)
blur = cv2.blur(img, (7, 7), 0) #先做模糊化，卷積核選擇值為7
th1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,7,1)
th2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,7,1)

#自適應閾值，使用算術平均自適應法，卷積核區塊大小選擇值為7，偏差量為1

cv2.imshow('original image', img)

cv2.imshow('blur image', blur)

cv2.imshow('original image after adatpive-thresholding', th1)

cv2.imshow('blur image after adatpive-thresholding', th2)

plt.hist(th1.ravel(),256,[0,256])
plt.show() 

cv2.waitKey(0)
cv2.destroyAllWindows()
