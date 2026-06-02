import numpy as np
import cv2
from matplotlib import pyplot as plt

target = cv2.imread('30084_0.jpg') 
reference= cv2.imread('30085_0.jpg')
# 特徵擷取
sift = cv2.xfeatures2d.SIFT_create()
kp1, descriptor1 = sift.detectAndCompute(target,None)
kp2, descriptor2 = sift.detectAndCompute(reference,None)
# 特徵比對(匹配)
bf = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)
matches = bf.match(descriptor1,descriptor2)
matches = sorted(matches, key = lambda x:x.distance)
# 繪製最匹配的代表
candidate_count = 15
result = cv2.drawMatches(target, kp1, reference, kp2, matches[:candidate_count], None)

cv2.imshow("result", result)
cv2.imwrite('sift_result.jpg', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
