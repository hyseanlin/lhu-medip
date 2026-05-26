import cv2 as cv2
from matplotlib import pyplot as plt

img = cv2.imread('kpop.jpg',cv2.IMREAD_GRAYSCALE) 
#cv2.IMRAD_GARYSCALE 以灰階方式來讀取Lena影像 

plt.hist(img.ravel(),256,[0,256])
#透過img.ravel()將像素資料轉換成一維
plt.show()
