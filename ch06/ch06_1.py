import cv2
import numpy as np
from image_registration import cross_correlation_shifts
from scipy.ndimage import shift

image = cv2.imread('map_010pb_left_0.jpg', cv2.IMREAD_GRAYSCALE)
offset_image = cv2.imread('map_010pb_right_0.jpg', cv2.IMREAD_GRAYSCALE)

xoff, yoff = cross_correlation_shifts(image, offset_image)
print(f'xoff={xoff}, yoff={yoff}')

corrected_image = shift(offset_image, shift=(-yoff, xoff), mode='constant')
corrected_image = corrected_image.astype(np.uint8)

# Convert grayscale reference image to BGR
base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Create reddish version of corrected_image
red_overlay = np.zeros_like(base)
red_overlay[:, :, 2] = corrected_image   # Put image into Red channel

# Blend images
alpha = 0.6
result = cv2.addWeighted(base, 1.0, red_overlay, alpha, 0)

cv2.imshow('reference image', image)
cv2.imshow('target image', offset_image)
cv2.imshow('registered image', corrected_image)
cv2.imshow('redish registered image', red_overlay)
cv2.imshow('overlay result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()