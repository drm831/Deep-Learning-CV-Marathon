# 作業
# 思考一下我們前面有提到圖片是矩陣，但維度可能會不一樣
# 例如灰階圖只有兩個維度，RGB 彩圖則有 3 個維度
# 假如今天我們把 BGR 3 個維度拆開來看會有甚麼不同的效果呢？
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 以彩色圖片的方式載入
def show_rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(121)
    plt.title('RGB Image')
    plt.imshow(img)

def show_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.subplot(122)
    plt.title('Gray Image')
    plt.imshow(img, cmap='gray')
    plt.show()

# 把圖片合併起來方便一起看
def show_concat(img):
    img = np.hstack((img[:, :, 2], img[:, :, 1], img[:, :, 0]))
    plt.title('Concat Image')
    plt.imshow(img, cmap='gray')
    plt.show()

if __name__ == '__main__':
    img = cv2.imread('./data/lena.png')
    show_rgb(img)
    show_gray(img)
    show_concat(img)