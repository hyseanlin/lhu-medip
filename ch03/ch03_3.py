import matplotlib.pyplot as plt;
import matplotlib.image as img;

import cv2
file_name = 'kpop.jpg'

# 利用 matplotlib 讀影像檔案
image = img.imread(file_name)

# 顯示圖片
plt.imshow(image)
plt.show()

# 裁切
cut_img = image[150:340, 150:500]

# 顯示圖片
plt.imshow(cut_img)
plt.show()
