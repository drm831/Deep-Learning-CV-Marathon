# 作業
# 實作本篇提到的三大概念 
# 翻轉：實作上下左右的翻轉
# 縮放：比較鄰近差值與雙立方插值 (或雙線性插值) 的圖片品質
# 平移：建立 Translation Transformation Matrix 來做平移

import cv2
import matplotlib.pyplot as plt
import time
import numpy as np

img = cv2.imread('./data/lena.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 上下左右翻轉圖片
# 水平翻轉 (horizontal)
img_hflip = cv2.flip(img_rgb, 1)
# 垂直翻轉 (vertical)
img_vflip = cv2.flip(img_hflip, 0)
# 水平 + 垂直翻轉
img_hvflip = cv2.flip(img_vflip, -1)

# 縮放圖片
# 放大
# 我們先透過縮小圖片去壓縮原有圖片保有的資訊，再放大比較不同方法之間的速度與圖片品質
# 將圖片縮小成原本的 20%
img_scale_20 = cv2.resize(img_rgb, None, fx=0.2, fy=0.2)
# 將圖片放大為"小圖片"的 8 倍大 = 原圖的 1.6 倍大
img_scale_80 = cv2.resize(img_scale_20, None, fx=8, fy=8)

# 鄰近差值 scale + 計算花費時間
start_time = time.time()
img_nearest_scale = cv2.resize(img_scale_20, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_NEAREST)
print('img NEAREST scale spend {} seconds'.format(time.time() - start_time))

# 雙立方差補 scale + 計算花費時間
start_time = time.time()
img_cubic_scale = cv2.resize(img_scale_20, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)
print('INTER CUBIC scale spend {} seconds'.format(time.time() - start_time))

# 平移幾何轉換
# 設定 translation transformation matrix
# x 平移 50 pixel; y 平移 100 pixel
M = np.array([[1, 0, 50], [0, 1, 100]], dtype=np.float32)
shift_img = cv2.warpAffine(img_rgb, M, (img_rgb.shape[1], img_rgb.shape[0]))

plt.subplot(341)
plt.title('RGB Image')
plt.imshow(img_rgb)
plt.subplot(342)
plt.title('Horizontal Flip Image')
plt.imshow(img_hflip)
plt.subplot(343)
plt.title('Vertical Flip Image')
plt.imshow(img_vflip)
plt.subplot(344)
plt.title('Horizontal & Vertical Flip Image')
plt.imshow(img_hvflip)
plt.subplot(345)
plt.title('20% Scale Image')
plt.imshow(img_scale_20)
plt.subplot(346)
plt.title('80% Scale Image')
plt.imshow(img_scale_80)
plt.subplot(347)
plt.title('NEAREST Scale Image')
plt.imshow(img_nearest_scale)
plt.subplot(348)
plt.title('CUBIC Scale Image')
plt.imshow(img_cubic_scale)
plt.subplot(349)
plt.title('Transformation Image')
plt.imshow(shift_img)
plt.show()