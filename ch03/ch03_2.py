import cv2
file_name = 'kpop.jpg'
# 讀檔案
img = cv2.imread(file_name)

# 裁切

cut_img = img[150:340, 150:500]

# 顯示圖片
cv2.imshow('original: ' + file_name, img)
cv2.imshow('cut: ' + file_name, cut_img )

# 擷取使用者的鍵盤輸入
cv2.waitKey(0) # 無窮等待直到使用者按下任一鍵
               # 才會往下執行下一行指令
               
cv2.destroyAllWindows()
