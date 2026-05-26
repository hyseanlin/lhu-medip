import numpy as np 
import cv2 as cv2 
from matplotlib import pyplot as plt 

img = cv2.imread('kpop.jpg',cv2.IMREAD_GRAYSCALE)

for ksize in [3, 5, 7, 9]:
    blur_gray=cv2.GaussianBlur(img,(3,3),0)
    th1 = 50
    th2 = th1*3
    edges=cv2.Canny(blur_gray, th1, th2) 
    erotion=cv2.erode(edges, (ksize, ksize), iterations=1) 
    cv2.imwrite(f'canny_{th1}_{th2}_erode_{ksize}.jpg', erotion)

# tmp = np.hstack((img,edges, erotion)) 
# cv2.imshow('image', tmp) 
# cv2.waitKey(0) 
# cv2.destroyAllWindows() 

