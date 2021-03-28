# 範例 
# 實作本篇提到的三大概念 
# 翻轉：實作上下翻轉
# 縮放：實作鄰近差值
# 平移：建立 Translation Transformation Matrix 來做平移

import cv2
import matplotlib.pyplot as plt
import time
import numpy as np

img = cv2.imread('./data/lena.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 上下翻轉圖片
# 垂直翻轉 (vertical)
# ::-1代表從 end 走到 start (倒序),同樣的方式對 x 軸處理就會變成水平翻轉
img_vflip = img_rgb[::-1, :, :]

# 縮放圖片 
# 放大 
# 我們先透過縮小圖片去壓縮原有圖片保有的資訊，再放大比較不同方法之間的速度與圖片品質

# 將圖片縮小成原本的 20%
img_scale_20 = cv2.resize(img_rgb, None, fx=0.2, fy=0.2)
# 將圖片放大為"小圖片"的 8 倍大 = 原圖的 1.6 倍大
img_scale_80 = cv2.resize(img_scale_20, None, fx=8, fy=8)

# 計算花費時間
start_time = time.time()
# 最近鄰插值
img_nearest_scale = cv2.resize(img_scale_20, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_NEAREST)
print('img nearest scale spend {} seconds'.format(time.time() - start_time))

# 平移幾何轉換
# 設定 translation transformation matrix
# x 平移 100 pixel; y 平移 50 pixel
matrix = np.array([[1, 0, 100], [0, 1, 50]], dtype=np.float32)
img_transformation = cv2.warpAffine(img_rgb, matrix, (img_rgb.shape[1], img_rgb.shape[0]))

plt.subplot(231)
plt.title('RGB Image')
plt.imshow(img_rgb)
plt.subplot(232)
plt.title('Vertical Flip Image')
plt.imshow(img_vflip)
plt.subplot(233)
plt.title('20% Scale Image')
plt.imshow(img_scale_20)
plt.subplot(234)
plt.title('80% Scale Image')
plt.imshow(img_scale_80)
plt.subplot(235)
plt.title('NEAREST Scale Image')
plt.imshow(img_nearest_scale)
plt.subplot(236)
plt.title('Transformation Image')
plt.imshow(img_transformation)
plt.show()