# 範例
# 取得 SIFT 特徵
# 轉成灰階圖片

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./data/lena.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_copy = img_rgb.copy()
# 轉灰階圖片
img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
# 使用 SIFT 來抽取特徵
# 建立 SIFT 物件
sift = cv2.SIFT_create()
# 抽取關鍵點
keypoints = sift.detect(img_gray, None)
img_sift = cv2.drawKeypoints(img_gray, keypoints, img_rgb)

plt.subplot(121)
plt.title('Gray Image')
plt.imshow(img_gray, cmap='gray')
plt.subplot(122)
plt.title('Sift Image')
plt.imshow(img_sift)
plt.show()