# 作業
# 思考一下我們前面有提到圖片是矩陣，但維度可能會不一樣
# 例如灰階圖只有兩個維度，RGB 彩圖則有 3 個維度
# 假如今天我們把 BGR 3 個維度拆開來看會有甚麼不同的效果呢？
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 以彩色圖片的方式載入
img_bgr = cv2.imread('./data/lena.png')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# 把圖片合併起來方便一起看
img_concat = np.hstack((img_rgb[:, :, 2], img_rgb[:, :, 1], img_rgb[:, :, 0]))

plt.title('RGB Image')
plt.imshow(img_rgb)
plt.show()
plt.title('Concat Image')
plt.imshow(img_concat, cmap='gray')
plt.show()