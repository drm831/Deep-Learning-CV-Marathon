# 以彩色圖片顯示
# 以灰階圖片顯示

import cv2
import matplotlib.pyplot as plt

# 以彩色圖片的方式載入
def show_rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(121)
    plt.title('RGB Image')
    plt.imshow(img)

# 以灰階圖片的方式載入
def show_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.subplot(122)
    plt.title('Gray Image')
    plt.imshow(img, cmap='gray')
    plt.show()

if __name__ == '__main__':
    img = cv2.imread('./data/lena.png')
    show_rgb(img)
    show_gray(img)