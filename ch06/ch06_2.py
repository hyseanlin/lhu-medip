import cv2 
import numpy as np

image = cv2.imread("kpop.jpg")

target = cv2.imread("kpop_rotated.jpg")

orb = cv2.ORB_create()

kps = orb.detect(image)
result_orig = cv2.drawKeypoints(image, kps, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DEFAULT)

kps_tgt = orb.detect(target)
result_tgt = cv2.drawKeypoints(target, kps_tgt, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DEFAULT)


cv2.imshow("keypoints extracted from the reference image", result_orig)
cv2.imshow("keypoints extracted from the target image", result_tgt)


cv2.waitKey(0)
cv2.destroyAllWindows()
