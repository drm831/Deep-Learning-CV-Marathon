# 作業
# 實作各種不一樣的方式來調整飽和 / 對比 / 明亮
# 改變 color space 來調整飽和度
# 實作直方圖均衡
# alpha/ beta 調整對比 / 明亮

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('./data/lena.png')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 改變飽和度
# 轉換成 HSV color space, 改變 s channel 的值
# 為了要改變飽和度，我們先把 color space 轉成 HSV 格式
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
change_percentage = 0.2

# 針對飽和度的值做改變，超過界線 0~1 的都會 bound
# 在 HSV color space 減少飽和度
img_hsv_down = img_hsv.astype('float32')
img_hsv_down[..., 1] = img_hsv_down[..., 1] / 255 - change_percentage
img_hsv_down[img_hsv_down[..., 1] < 0] = 0
img_hsv_down[..., 1] = img_hsv_down[..., 1] * 255
img_hsv_down = img_hsv_down.astype('uint8')

# 在 HSV color space 增加飽和度
img_hsv_up = img_hsv.astype('float32')
img_hsv_up[..., 1] = img_hsv_up[..., 1] / 255 + change_percentage
img_hsv_up[img_hsv_up[..., 1] > 1] = 1
img_hsv_up[..., 1] = img_hsv_up[..., 1] * 255
img_hsv_up = img_hsv_up.astype('uint8')

# 轉換 color space 回 RGB
img_hsv_down = cv2.cvtColor(img_hsv_down, cv2.COLOR_HSV2BGR)
img_hsv_up = cv2.cvtColor(img_hsv_up, cv2.COLOR_HSV2BGR)

# 直方圖均衡
# case 1: 把彩圖拆開對每個 channel 個別做直方圖均衡再組合起來
# case 2: 轉換 color space 到 HSV 之後對其中一個 channel 做直方圖均衡

# case 1
# 每個 channel 個別做直方圖均衡
equalHist_by_channel = [img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]]
equalHist_by_channel = [cv2.equalizeHist(i) for i in equalHist_by_channel]
# 組合經過直方圖均衡的每個 channel
img_rgb_equal = np.stack(equalHist_by_channel, axis=-1)

# # case 2 - 轉換 color space 後只對其中一個 channel 做直方圖均衡
img_hsv[..., -1] = cv2.equalizeHist(img_hsv[..., -1])
img_hsv_equal = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

# 調整對比 / 明亮
# alpha: 控制對比度 (1.0~3.0)
# beta: 控制明亮度 (0~255)
add_contrast = cv2.convertScaleAbs(img_rgb, alpha=2.0, beta=0)
add_lighness = cv2.convertScaleAbs(img_rgb, alpha=1.0, beta=50)

plt.subplot(241)
plt.title('RGB Image')
plt.imshow(img_rgb)
plt.subplot(242)
plt.title('HSV Image')
plt.imshow(img_hsv)
plt.subplot(243)
plt.title('HSV Down Image')
plt.imshow(img_hsv_down)
plt.subplot(244)
plt.title('HSV Up Image')
plt.imshow(img_hsv_up)
plt.subplot(245)
plt.title('RGB Equal Image')
plt.imshow(img_rgb_equal)
plt.subplot(246)
plt.title('HSV Equal Image')
plt.imshow(img_hsv_equal)
plt.subplot(247)
plt.title('Add Contrast Image')
plt.imshow(add_contrast)
plt.subplot(248)
plt.title('Add Contrast Image')
plt.imshow(add_lighness)
plt.show()