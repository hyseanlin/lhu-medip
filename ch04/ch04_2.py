
import cv2 as cv2 
from matplotlib import pyplot as plt 

img = cv2.imread('too_dark.jpg',cv2.IMREAD_GRAYSCALE)
plt.hist(img.ravel(),256,[0,256])
plt.show() 

equa = cv2.equalizeHist(img) 
plt.hist(equa.ravel(),256,[0,256])

plt.show() 

cv2.imwrite('too_bright_gray.jpg', img)
cv2.imwrite('too_bright_equa.jpg', equa)

cv2.waitKey(0) 
cv2.destroyAllWindows() 
