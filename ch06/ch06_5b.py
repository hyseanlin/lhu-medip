import numpy as np
import cv2
from matplotlib import pyplot as plt

target = cv2.imread('map_010pb_left_0.jpg') 
reference= cv2.imread('map_010pb_right_0.jpg')

sift = cv2.xfeatures2d.SIFT_create()
kp1, descriptor1 = sift.detectAndCompute(target,None)
kp2, descriptor2 = sift.detectAndCompute(reference,None)

bf = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)
matches = bf.match(descriptor1,descriptor2)
matches = sorted(matches, key = lambda x:x.distance)
candidate_count = 50
result = cv2.drawMatches(target, kp1, reference, kp2, matches[:candidate_count], None)

cv2.imshow("result", result)
cv2.imwrite('sift_result.jpg', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
