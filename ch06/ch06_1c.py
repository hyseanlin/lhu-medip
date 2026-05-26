import cv2
import numpy as np
from image_registration import cross_correlation_shifts

# Read images
image = cv2.imread('map_010pb_left_0.jpg', cv2.IMREAD_GRAYSCALE)
offset_image = cv2.imread('map_010pb_right_0.jpg', cv2.IMREAD_GRAYSCALE)

# Compute translation offset
xoff, yoff = cross_correlation_shifts(image, offset_image)
print(f'xoff={xoff}, yoff={yoff}')

h1, w1 = image.shape
h2, w2 = offset_image.shape

dx = int(round(xoff))
dy = int(round(-yoff))

min_x = min(0, dx)
min_y = min(0, dy)

max_x = max(w1, dx + w2)
max_y = max(h1, dy + h2)

canvas_w = max_x - min_x
canvas_h = max_y - min_y

base_x = -min_x
base_y = -min_y

# ------------------------------------------------------------
# Create RGB canvases
# ------------------------------------------------------------

# Original/reference image in RGB format
result_rgb = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

# Put original grayscale image into all RGB channels
result_rgb[
    base_y:base_y + h1,
    base_x:base_x + w1,
    :
] = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

# Create shifted target RGB overlay
offset_rgb = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

# Put offset image into RED channel only
offset_rgb[
    base_y + dy:base_y + dy + h2,
    base_x + dx:base_x + dx + w2,
    0
] = offset_image   # RGB: channel 0 = Red

# Blend original RGB image and red shifted image
result_rgb = cv2.addWeighted(result_rgb, 1.0, offset_rgb, 0.7, 0)

# ------------------------------------------------------------
# For OpenCV display, convert RGB to BGR
# ------------------------------------------------------------

result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

cv2.imshow('reference image', image)
cv2.imshow('target image', offset_image)
cv2.imshow('overlay result', result_bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()