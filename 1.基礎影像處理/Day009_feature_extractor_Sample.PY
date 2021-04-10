# 範例
# 取得 SIFT 特徵
# 轉成灰階圖片

import cv2
import matplotlib.pyplot as plt

def show_rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(131)
    plt.title('RGB Image')
    plt.imshow(img)

# 轉灰階圖片
def show__gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.subplot(132)
    plt.title('Gray Image')
    plt.imshow(img, cmap='gray')
def show_sift(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用 SIFT 來抽取特徵
    # 建立 SIFT 物件
    sift = cv2.SIFT_create()
    # 抽取關鍵點
    keypoints = sift.detect(img, None)
    img_sift = cv2.drawKeypoints(img, keypoints, img)
    plt.subplot(133)
    plt.title('Sift Image')
    plt.imshow(img_sift)
    plt.show()

if __name__ == '__main__':
    img = cv2.imread('./data/lena.png')
    show_rgb(img)
    show__gray(img)
    show_sift(img)