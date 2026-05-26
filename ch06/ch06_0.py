import cv2 as cv2
import numpy as np
from skimage import io
from image_registration import cross_correlation_shifts
from scipy.ndimage import shift
image = cv2.imread('kpop.jpg',cv2.IMREAD_GRAYSCALE) 
offset_image=cv2.imread('kpop_trans1.jpg',cv2.IMREAD_GRAYSCALE)
xoff, yoff = cross_correlation_shifts(image, offset_image)
corrected_image = shift(offset_image, shift=(-yoff,-xoff), mode='constant')
results = np.hstack((image,offset_image, corrected_image))  
cv2.imshow('image', results)
cv2.waitKey(0)
cv2.destroyAllWindows()
