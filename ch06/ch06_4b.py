import cv2
import numpy as np

# --------------------------------------------------
# Read image
# --------------------------------------------------
image = cv2.imread("map_010pb_left_0.jpg")

if image is None:
    print("Error: image not found")
    exit()

cv2.imshow("input", image)

# --------------------------------------------------
# Create ORB detector
# --------------------------------------------------
orb = cv2.ORB_create(
    nfeatures=1000
)

# --------------------------------------------------
# Detect keypoints
# --------------------------------------------------
kps = orb.detect(image, None)

# --------------------------------------------------
# Create overlay image
# --------------------------------------------------
overlay = image.copy()

# --------------------------------------------------
# Draw small transparent keypoints manually
# --------------------------------------------------
for kp in kps:
    x, y = map(int, kp.pt)

    radius = 3 #(small point)
    cv2.circle(
        overlay,
        (x, y),
        radius,
        (0, 0, 255),   # red
        -1             # filled circle
    )

# --------------------------------------------------
# Blend overlay with original image
# --------------------------------------------------
alpha = 0.35

result = cv2.addWeighted(
    overlay,
    alpha,
    image,
    1 - alpha,
    0
)

# --------------------------------------------------
# Show result
# --------------------------------------------------
cv2.imshow("result", result)

# --------------------------------------------------
# Save result
# --------------------------------------------------
cv2.imwrite("orb_result.jpg", result)

# --------------------------------------------------
# Wait and close
# --------------------------------------------------
cv2.waitKey(0)
cv2.destroyAllWindows()