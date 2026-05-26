import cv2 
import numpy as np
image = cv2.imread("map_010pb_left_0.jpg")

cv2.imshow("input", image)

orb = cv2.ORB_create()
kps = orb.detect(image)

result = cv2.drawKeypoints(image, kps, None, -1, cv2. DRAW_MATCHES_FLAGS_DEFAULT)


cv2.imshow("result", result)
cv2.imwrite('orb_result.jpg', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
