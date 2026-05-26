
import cv2
file_name = 'kpop.jpg'

# 利用 matplotlib 讀影像檔案
image = cv2.imread(file_name)

(h, w, d) = image.shape
center = ( w//2, h//2 )
angle = -15
scale = 1.5

P = cv2.getRotationMatrix2D(center, angle, scale)

rot_image = cv2.warpAffine(image, P, (w, h))

# 顯示圖片
cv2.imshow('original', image)
cv2.imshow('rotated', rot_image)

cv2.imwrite('kpop_rotated.jpg', rot_image)

cv2.waitKey(0)
cv2.destroyAllWindows()