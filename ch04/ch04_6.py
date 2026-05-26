import numpy as np 
import cv2 as cv2 
from matplotlib import pyplot as plt 


img = cv2.imread('kpop.jpg',cv2.IMREAD_GRAYSCALE)

for th1 in [50]:
    for i in [1, 3, 5, 7]:
        th2 = th1 * 3 # th1:th2   1:2
        blur_gray=cv2.GaussianBlur(img,(3,3),0)
        
        edges=cv2.Canny(blur_gray, th1, th2)
        
        dilation=cv2.dilate(edges, (5,5), iterations=i) 
        # tmp = np.hstack((img,edges, dilation)) 
        
        # cv2.imwrite(f'cann_{th1}_{th2}.jpg', edges)
        cv2.imwrite(f'cann_{th1}_{th2}_dilate_{i}.jpg', dilation)
    
# cv2.imshow('image', tmp) 
# cv2.waitKey(0) 
# cv2.destroyAllWindows() 

