# 改變圖片的 color space (HSV) 並呈現

import cv2
import matplotlib.pyplot as plt

def show_rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(121)
    plt.title('RGB Image')
    plt.imshow(img)

def show_hsv(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    plt.subplot(122)
    plt.title('HSV Image')
    plt.imshow(img)
    plt.show()
    plt.close()

if __name__ == '__main__':
    img = cv2.imread('./data/lena.png')
    show_rgb(img)
    show_hsv(img)