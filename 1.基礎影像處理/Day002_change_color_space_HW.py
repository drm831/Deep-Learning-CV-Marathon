# 作業
# 改變圖片的 color space (HLS, LAB) 並呈現, 包含 RGB, HSV, HSL, LAB
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./data/lena.png')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

plt.subplot(141)
plt.title('RGB Image')
plt.imshow(img_rgb)
plt.subplot(142)
plt.title('HSV Image')
plt.imshow(img_hsv)
plt.subplot(143)
plt.title('HLS Image')
plt.imshow(img_hls)
plt.subplot(144)
plt.title('LAB Image')
plt.imshow(img_lab)
plt.show()