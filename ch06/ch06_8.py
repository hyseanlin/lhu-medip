import numpy as np
import cv2
from matplotlib import pyplot as plt

def grabcut(imgpath, iterCount=5):
    # 讀取影像
    img = cv2.imread(imgpath)

    # 使用滑鼠選取 ROI
    # 回傳格式：(x, y, width, height)
    rect = cv2.selectROI("Select ROI", img, showCrosshair=True)
    cv2.destroyWindow("Select ROI")

    # 建立 mask，用來儲存cv2.grabCut執行後的回傳結果
    mask = np.zeros(img.shape[:2], np.uint8)

    # 建立背景模型，用來儲存 GMM (Gaussian Mixture Model) 參數。
    bgdModel = np.zeros((1, 65), np.float64)
    # 建立前景模型，用來儲存 GMM (Gaussian Mixture Model) 參數。
    fgdModel = np.zeros((1, 65), np.float64)

    # 執行 GrabCut
    cv2.grabCut(
        img,
        mask,
        rect,
        bgdModel,
        fgdModel,
        iterCount,
        cv2.GC_INIT_WITH_RECT
    )
    '''
    cv2.grabCut執行後的mask會有四種狀態
        0 確定背景
        1 確定前景
        2 可能背景
        3 可能前景
    '''
    # 將 mask 轉換成 0/1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    # 套用 mask
    result = img * mask2[:, :, np.newaxis]

    # 顯示結果
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.imshow(cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("GrabCut")
    plt.axis('off')

    plt.show()


# ----------------主程式----------------
imgpath = "apple.png"
grabcut(imgpath, iterCount=5)