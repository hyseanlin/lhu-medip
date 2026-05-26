import cv2
import numpy as np

img = cv2.imread('kpop.jpg')

cv2.imshow('original image', img)
for N in [9]:
    blur_avg = cv2.blur(img, (N, N)) #N設為13 
    blur_gauss = cv2.GaussianBlur(img, (N, N), 0) #N設為13 
    blur_med = cv2.medianBlur(img, N, 0) #N設為13 
    cv2.imshow(f'image filtered by Averaging {N}x{N}', blur_avg)
    cv2.imshow(f'image filtered by Gaussian {N}x{N}', blur_gauss)
    cv2.imshow(f'image filtered by Median {N}x{N}', blur_med)
    
cv2.waitKey(0)
cv2.destroyAllWindows()
