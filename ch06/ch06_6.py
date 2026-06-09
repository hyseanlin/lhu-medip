import numpy as np
import cv2
import os

def watershed(image):

    # 建立輸出資料夾
    output_dir = "watershed_results"
    os.makedirs(output_dir, exist_ok=True)

    # 使用邊緣保留濾波先對影像去除雜訊
    blur = cv2.pyrMeanShiftFiltering(image, sp=10, sr=100)
    cv2.imwrite(os.path.join(output_dir, "01_blur.jpg"), blur)

    # 將全彩影像轉成灰階影像
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, "02_gray.jpg"), gray)

    # 使用大津二值化將影像二值化
    ret, binary = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    cv2.imwrite(os.path.join(output_dir, "03_binary.jpg"), binary)

    # 使用影像形態學中的開運算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel=kernel,
        iterations=3
    )
    cv2.imwrite(os.path.join(output_dir, "04_opening.jpg"), opening)

    # 針對確認的背景區域進行膨脹運算
    sure_bg = cv2.dilate(opening, kernel, iterations=4)
    cv2.imwrite(os.path.join(output_dir, "05_sure_bg.jpg"), sure_bg)

    # 使用距離變換
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 3)

    dist_out = cv2.normalize(
        dist,
        None,
        0,
        1.0,
        cv2.NORM_MINMAX
    )

    # 將距離圖轉成可視化格式
    dist_vis = (dist_out * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "06_distance_transform.jpg"), dist_vis)

    # 找出前景
    ret, surface = cv2.threshold(
        dist_out,
        dist_out.max() * 0.6,
        255,
        cv2.THRESH_BINARY
    )

    sure_fg = np.uint8(surface)
    cv2.imwrite(os.path.join(output_dir, "07_sure_fg.jpg"), sure_fg)

    # 未知區域
    unknown = cv2.subtract(sure_bg, sure_fg)
    cv2.imwrite(os.path.join(output_dir, "08_unknown.jpg"), unknown)

    # Connected Components
    ret, markers = cv2.connectedComponents(sure_fg)

    # 標籤加1
    markers = markers + 1

    # 未知區域設為0
    markers[unknown == 255] = 0

    # 將 marker 正規化後存檔方便觀察
    markers_vis = cv2.normalize(
        markers.astype(np.float32),
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    cv2.imwrite(os.path.join(output_dir, "09_markers_before_watershed.jpg"), markers_vis)

    # Watershed
    markers = cv2.watershed(image, markers)

    # Watershed後的marker
    markers_after_vis = cv2.normalize(
        markers.astype(np.float32),
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    cv2.imwrite(
        os.path.join(output_dir, "10_markers_after_watershed.jpg"),
        markers_after_vis
    )

    # 邊界標記為紅色
    result = image.copy()
    result[markers == -1] = [0, 0, 255]

    cv2.imwrite(os.path.join(output_dir, "11_final_result.jpg"), result)

    print(f"All results saved in: {output_dir}")

    return result


image = cv2.imread("coin.jpg")

if image is None:
    print("Cannot read image.")
else:
    result = watershed(image)