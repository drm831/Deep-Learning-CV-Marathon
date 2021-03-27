# 改變圖片的 color space (HSV) 並呈現
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./data/lena.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

plt.subplot(121)
plt.title('RGB Image')
plt.imshow(img_rgb)
plt.subplot(122)
plt.title('HSV Image')
plt.imshow(img_hsv)
plt.show()