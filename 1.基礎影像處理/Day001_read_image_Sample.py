# 以彩色圖片顯示
# 以灰階圖片顯示
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./data/lena.png')
# 以彩色圖片的方式載入
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 以灰階圖片的方式載入
img_gray = cv2.cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.title('RGB Image')
plt.imshow(img_rgb)
plt.show()
plt.title('Gray Image')
plt.imshow(img_gray, cmap='gray')
plt.show()