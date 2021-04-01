# 作業 
# 實作模糊與邊緣檢測
# 透過 Gaussian Filter 實作模糊操作
# 透過 Sobel Filter 實作邊緣檢測

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./data/lena.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 邊緣檢測
# 比較 Sobel 如果在 uint8 的情況下做會 overflow 的狀況
# 轉為灰階圖片
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# 對 x 方向以包含負數的資料格式 (cv2.CV_16S) 進行 Sobel 邊緣檢測
img_sobel_x = cv2.Sobel(img_gray, cv2.CV_16S, dx=1, dy=0, ksize=3)
# 對 x 方向依照比例縮放到所有數值都是非負整數
img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
# 對 x 方向直接以非負整數的資料格式 (uint8) 進行 Sobel 邊緣檢測
img_sobel_x_uint8 = cv2.Sobel(img_sobel_x, -1, dx=1, dy=0, ksize=3)

# 比較一次與兩次計算偏微分的結果
img_gray_copy = img_gray.copy()
# 求一次導數取得邊緣檢測結果
img_sobel_x_copy = cv2.Sobel(img_gray_copy, cv2.CV_16S, dx=1, dy=0, ksize=3)
img_sobel_x_copy = cv2.convertScaleAbs(img_sobel_x_copy)
# 求二次導數取得邊緣檢測結果
img_sobel_xx = cv2.Sobel(img_gray_copy, cv2.CV_16S, dx=2, dy=0, ksize=3)
img_sobel_xx = cv2.convertScaleAbs(img_sobel_xx)

plt.subplot(231)
plt.title('RGB Image')
plt.imshow(img_rgb)
plt.subplot(232)
plt.title('Gray Image')
plt.imshow(img_gray, cmap='gray')
plt.subplot(233)
plt.title('Sobel X Image')
plt.imshow(img_sobel_x, cmap='gray')
plt.subplot(234)
plt.title('Sobel X Unit8 Image')
plt.imshow(img_sobel_x_uint8, cmap='gray')
plt.subplot(235)
plt.title('Sobel X Image')
plt.imshow(img_sobel_x_copy, cmap='gray')
plt.subplot(236)
plt.title('Sobel XX Image')
plt.imshow(img_sobel_xx, cmap='gray')
plt.show()