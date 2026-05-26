import cv2
import numpy as np

img = cv2.imread('kpop.jpg')

cv2.imshow('original image', img)
for N in [3, 5, 7, 9, 13]:
    blur = cv2.blur(img, (N, N)) # N設為13  
    cv2.imshow(f'image filtered by {N}x{N}', blur)
    
cv2.waitKey(0)
cv2.destroyAllWindows()
