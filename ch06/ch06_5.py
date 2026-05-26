import numpy as np
import cv2
from matplotlib import pyplot as plt

target = cv2.imread('map_010pb_left_0.jpg') 
reference= cv2.imread('map_010pb_right_0.jpg')
# feature extraction (特徵提取/擷取)
orb = cv2.ORB_create(nfeatures=1000)
kp1, descriptor1 = orb.detectAndCompute(target,None)
kp2, descriptor2 = orb.detectAndCompute(reference,None)
# feature matching (特徵匹配)
bf = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)
matches = bf.match(descriptor1,descriptor2)
# 依據最匹配結果進行排序
matches = sorted(matches, key = lambda x:x.distance)
candidate_count = 3
result = cv2.drawMatches(target, kp1, reference, kp2, matches[:candidate_count], None)

cv2.imshow("result", result)
cv2.imwrite('orb_result.jpg', result)
cv2.waitKey(0)
cv2.destroyAllWindows()