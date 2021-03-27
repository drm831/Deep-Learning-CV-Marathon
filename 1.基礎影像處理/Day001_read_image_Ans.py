# 作業
# 思考一下我們前面有提到圖片是矩陣，但維度可能會不一樣
# 例如灰階圖只有兩個維度，RGB 彩圖則有 3 個維度
# 假如今天我們把 BGR 3 個維度拆開來看會有甚麼不同的效果呢？

import cv2
import numpy as np

img_path = './data/lena.png'

# 以彩色圖片的方式載入
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

# 把圖片合併起來方便一起看
img_concat = np.hstack((img[:, :, 0], img[:, :, 1], img[:, :, 2]))

# 為了要不斷顯示圖片，所以使用一個迴圈
while True:
    # 顯示彩圖
    cv2.imshow('bgr', img)
    cv2.imshow('bgr_split', img_concat)

    # 直到按下 ESC 鍵才會自動關閉視窗結束程式
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        break