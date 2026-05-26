import cv2
import numpy as np
from image_registration import cross_correlation_shifts
from scipy.ndimage import shift

# Read images
image = cv2.imread('map_010pb_left_0.jpg', cv2.IMREAD_GRAYSCALE)
offset_image = cv2.imread('map_010pb_right_0.jpg', cv2.IMREAD_GRAYSCALE)

# Compute translation offset
xoff, yoff = cross_correlation_shifts(image, offset_image)
print(f'xoff={xoff}, yoff={yoff}')

# ------------------------------------------------------------
# Create a canvas large enough to contain BOTH images
# after alignment
# ------------------------------------------------------------

h1, w1 = image.shape
h2, w2 = offset_image.shape

# Shift applied to offset_image
dx = int(round(xoff))
dy = int(round(-yoff))

# Compute canvas bounds
min_x = min(0, dx)
min_y = min(0, dy)

max_x = max(w1, dx + w2)
max_y = max(h1, dy + h2)

canvas_w = max_x - min_x
canvas_h = max_y - min_y

# Offsets to place images into positive coordinates
base_x = -min_x
base_y = -min_y

# ------------------------------------------------------------
# Create canvases
# ------------------------------------------------------------

base_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
offset_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

# Place reference image
base_canvas[
    base_y:base_y + h1,
    base_x:base_x + w1
] = image

# Place offset image directly at shifted canvas position
offset_canvas[
    base_y + dy:base_y + dy + h2,
    base_x + dx:base_x + dx + w2
] = offset_image

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

# Convert reference image to BGR
base_bgr = cv2.cvtColor(base_canvas, cv2.COLOR_GRAY2BGR)

# Create red-colored overlay image
overlay_bgr = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

# Put shifted image into RED channel
overlay_bgr[:, :, 2] = offset_canvas

# Blend both images
result = cv2.addWeighted(base_bgr, 1.0, overlay_bgr, 0.7, 0)

# ------------------------------------------------------------
# Show results
# ------------------------------------------------------------

cv2.imshow('reference image', image)
cv2.imshow('target image', offset_image)
cv2.imshow('registered image', offset_canvas)
cv2.imshow('overlay result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()