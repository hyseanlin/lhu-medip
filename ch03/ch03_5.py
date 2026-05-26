
import cv2
file_name = 'kpop.jpg'

# 利用 matplotlib 讀影像檔案
img = cv2.imread(file_name)

(h, w, d) = img.shape

img_resized_lanczos = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation= cv2.INTER_LANCZOS4)

img_resized_nn = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation= cv2.INTER_NEAREST)


# 顯示圖片
cv2.imshow('original', img)
cv2.imshow('resized with lanczos', img_resized_lanczos)
cv2.imshow('resized with nearest', img_resized_nn)


cv2.waitKey(0)
cv2.destroyAllWindows()