# 範例 
# Hint: 人物原始邊框座標 (60, 40), (420, 510)
# 請根據 Lena 圖做以下處理
# 對明亮度做直方圖均衡處理
# 水平鏡像 + 縮放處理 (0.5 倍)
# 畫出人物矩形邊框

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./data/lena.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Hint: 矩形
img_copy = img_rgb.copy()
img_rectangle = cv2.rectangle(img_copy, (60, 40), (420 , 510), (255, 0, 0), 3)              

# Hint: 線
img_copy = img_rgb.copy()
img_line = cv2.line(img_copy, (60, 40), (420 , 510), (255, 0, 0), 3)

# Hint: 文字
img_copy = img_rgb.copy()
img_text = cv2.putText(img_copy, '(60, 40)', (60, 40), 0, 1, (255, 0, 0), 2)

# 假設我們希望先對圖片做以下幾點預處理，請印出最後結果
# Hint: 注意先後順序，人物原始邊框座標 (60, 40), (420, 510)
# 對明亮度做直方圖均衡處理
# 水平鏡像 + 縮放處理 (0.5 倍)
# 畫出人物矩形邊框

# 對明亮度做直方圖均衡處理
# 原始 BGR 圖片轉 HSV 圖片
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
# 對明亮度做直方圖均衡 -> 對 HSV 的 V 做直方圖均衡
img_hsv[..., -1] = cv2.equalizeHist(img_hsv[..., -1])
# 將圖片轉回 BGR
img_equalizeHist = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
# 水平鏡像 + 縮放處理 (0.5 倍)
# 水平鏡像 (圖片)
img_resize = img_equalizeHist[:, ::-1, :]
# 縮放處理
img_resize = cv2.resize(img_resize, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
# 畫出人物矩形邊框
img_copy = img_resize.copy()
img_resize_rectangle = cv2.rectangle(img_copy, (45, 19), (228, 255),(255, 0, 0), 2)

plt.subplot(241)
plt.title('RGB Image')
plt.imshow(img_rgb)
plt.subplot(242)
plt.title('Rectangle Image')
plt.imshow(img_rectangle)
plt.subplot(243)
plt.title('Line Image')
plt.imshow(img_line)
plt.subplot(244)
plt.title('Text Image')
plt.imshow(img_text)
plt.subplot(245)
plt.title('EqualizeHist Image')
plt.imshow(img_equalizeHist)
plt.subplot(246)
plt.title('Resize Image')
plt.imshow(img_resize)
plt.subplot(247)
plt.title('Resize Rectangle Image')
plt.imshow(img_resize_rectangle)
plt.show()