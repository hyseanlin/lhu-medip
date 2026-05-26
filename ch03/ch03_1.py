import cv2
file_name = 'kpop.jpg'
# 讀檔案
img = cv2.imread(file_name)

img2 = img.copy()

# 改變特定位置或區域的像素值((B, G, R)的顏色值)
# B
img2[243:250, 340:350, 0] = 0
# G
img2[243:250, 340:350, 1] = 255
# R
img2[243:250, 340:350, 2] = 0

cv2.imwrite('kpop_v1.bmp', img2)

# 顯示圖片
cv2.imshow('original: ' + file_name, img)
cv2.imshow('modified: ' + file_name, img2)

# 擷取使用者的鍵盤輸入
cv2.waitKey(0) # 無窮等待直到使用者按下任一鍵
               # 才會往下執行下一行指令
               
cv2.destroyAllWindows()
