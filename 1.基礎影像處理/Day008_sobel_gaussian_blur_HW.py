# 作業 
# 實作模糊與邊緣檢測
# 透過 Gaussian Filter 實作模糊操作
# 透過 Sobel Filter 實作邊緣檢測

import cv2
import matplotlib.pyplot as plt

def show_rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(231)
    plt.title('RGB Image')
    plt.imshow(img)

# 邊緣檢測
# 比較 Sobel 如果在 uint8 的情況下做會 overflow 的狀況
# 轉為灰階圖片
def show__gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.subplot(232)
    plt.title('Gray Image')
    plt.imshow(img, cmap='gray')

def show_sobel_x(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 對 x 方向以包含負數的資料格式 (cv2.CV_16S) 進行 Sobel 邊緣檢測
    img_sobel_x = cv2.Sobel(img, cv2.CV_16S, dx=1, dy=0, ksize=3)
    # 對 x 方向依照比例縮放到所有數值都是非負整數
    img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
    plt.subplot(233)
    plt.title('Sobel X Image')
    plt.imshow(img_sobel_x, cmap='gray')

def show_sobel_x_uint8(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 對 x 方向以包含負數的資料格式 (cv2.CV_16S) 進行 Sobel 邊緣檢測
    img_sobel_x = cv2.Sobel(img, cv2.CV_16S, dx=1, dy=0, ksize=3)
    # 對 x 方向依照比例縮放到所有數值都是非負整數
    img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
    # 對 x 方向直接以非負整數的資料格式 (uint8) 進行 Sobel 邊緣檢測
    img_sobel_x_uint8 = cv2.Sobel(img_sobel_x, -1, dx=1, dy=0, ksize=3)
    plt.subplot(234)
    plt.title('Sobel X Unit8 Image')
    plt.imshow(img_sobel_x_uint8, cmap='gray')

def show_sobel_x_copy(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 比較一次與兩次計算偏微分的結果
    # 求一次導數取得邊緣檢測結果
    img_sobel_x_copy = cv2.Sobel(img, cv2.CV_16S, dx=1, dy=0, ksize=3)
    img_sobel_x_copy = cv2.convertScaleAbs(img_sobel_x_copy)
    plt.subplot(235)
    plt.title('Sobel X Image')
    plt.imshow(img_sobel_x_copy, cmap='gray')

def show_sobel_xx(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 比較一次與兩次計算偏微分的結果
    # 求一次導數取得邊緣檢測結果
    img_sobel_x_copy = cv2.Sobel(img, cv2.CV_16S, dx=1, dy=0, ksize=3)
    img_sobel_x_copy = cv2.convertScaleAbs(img_sobel_x_copy)
    # 求二次導數取得邊緣檢測結果
    img_sobel_xx = cv2.Sobel(img, cv2.CV_16S, dx=2, dy=0, ksize=3)
    img_sobel_xx = cv2.convertScaleAbs(img_sobel_xx)
    plt.subplot(236)
    plt.title('Sobel XX Image')
    plt.imshow(img_sobel_xx, cmap='gray')
    plt.show()

if __name__ == '__main__':
    img = cv2.imread('./data/lena.png')
    show_rgb(img)
    show__gray(img)
    show_sobel_x(img)
    show_sobel_x_uint8(img)
    show_sobel_x_copy(img)
    show_sobel_xx(img)