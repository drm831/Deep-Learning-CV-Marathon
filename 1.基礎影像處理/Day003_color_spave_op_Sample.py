# 範例
# 實作各種不一樣的方式來調整飽和 / 對比 / 明亮
# 改變到 HSL color space 來調整飽和度
# 對灰階圖實作直方圖均衡
# alpha/ beta 調整對比 / 明亮

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./data/lena.png')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 改變飽和度
# 轉換成 HLS color space, 改變 s channel 的值
# 為了要改變飽和度，我們先把 color space 轉成 HSL 格式 (OpenCV 表示順序是 HLS)
img_hls = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HLS)
change_percentage = 0.2

# 針對飽和度的值做改變，超過界線 0~1 的都會 bound
# 在 HLS color space 減少飽和度
img_hls_down = img_hls.astype('float32')
# 轉成小數點並調降兩成飽和度
img_hls_down[..., -1] = img_hls_down[..., -1] / 255 - change_percentage
img_hls_down[img_hls_down[..., -1] < 0] = 0
img_hls_down[..., -1] = img_hls_down[..., -1] * 255
img_hls_down = img_hls_down.astype('uint8')

# 在 HLS color space 增加飽和度
img_hls_up = img_hls.astype('float32')
img_hls_up[..., -1] = img_hls_up[..., -1] / 255 + change_percentage
img_hls_up[img_hls_up[..., -1] > 1] = 1
img_hls_up[..., -1] = img_hls_up[..., -1] * 255
img_hls_up = img_hls_up.astype('uint8')

# 轉換
img_hls_down = cv2.cvtColor(img_hls_down, cv2.COLOR_HLS2BGR)
img_hls_up = cv2.cvtColor(img_hls_up, cv2.COLOR_HLS2BGR)

# 直方圖均衡
# 轉為灰階圖片
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# 灰階圖片直方圖均衡
img_gray_equal = cv2.equalizeHist(img_gray)

# 調整對比 / 明亮
# alpha: 控制對比度 (1.0~3.0)
# beta: 控制明亮度 (0~255)
add_contrast = cv2.convertScaleAbs(img_rgb, alpha=2.0, beta=0)
add_lighness = cv2.convertScaleAbs(img_rgb, alpha=1.0, beta=50)

plt.subplot(241)
plt.title('RGB Image')
plt.imshow(img_rgb)
plt.subplot(242)
plt.title('HLS Image')
plt.imshow(img_hls)
plt.subplot(243)
plt.title('HLS Down Image')
plt.imshow(img_hls_down)
plt.subplot(244)
plt.title('HLS Up Image')
plt.imshow(img_hls_up)
plt.subplot(245)
plt.title('Gray Image')
plt.imshow(img_gray, cmap='gray')
plt.subplot(246)
plt.title('Gray Equal Image')
plt.imshow(img_gray_equal, cmap='gray')
plt.subplot(247)
plt.title('Add Contrast Image')
plt.imshow(add_contrast)
plt.subplot(248)
plt.title('Add Contrast Image')
plt.imshow(add_lighness)
plt.show()