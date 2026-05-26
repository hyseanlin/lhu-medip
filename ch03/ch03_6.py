import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

# 自訂函式
def	modify_contrast_and_brightness(img,alpha=1.0,beta=0.0):
    img = img.astype(np.float32)
    img = img * alpha + beta
    img = np.clip(img, 0, 255) #將array的數值範圍設定成0-255 避免溢位
    img = img.astype(np.uint8)
    return img

# 主程式
img = cv2.imread('kpop.jpg')
# 傳入的參數為img, alpha, beta
modified_image= modify_contrast_and_brightness(img, 0.6, 100.0)
cv2.imshow("Original",img)
cv2.imshow("Modified",modified_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
