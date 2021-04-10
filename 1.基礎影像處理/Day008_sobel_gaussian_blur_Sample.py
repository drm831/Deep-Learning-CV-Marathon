# 範例
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

# 模糊
def show_blur(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 重複多次 Gaussian 模糊的操作來加深模糊的程度
    img_blur = cv2.GaussianBlur(img, (5,5), 0)
    img_blur = cv2.GaussianBlur(img_blur, (5,5), 0)
    img_blur = cv2.GaussianBlur(img_blur, (5,5), 0)
    plt.subplot(232)
    plt.title('Blur Image')
    plt.imshow(img_blur)

def show_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.subplot(233)
    plt.title('Gray Image')
    plt.imshow(img, cmap='gray')

# 邊緣檢測
# 組合 x-axis, y-axis 的影像合成
def show_sobel_x(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 對 x 方向做 Sobel 邊緣檢測
    img_sobel_x = cv2.Sobel(img, cv2.CV_16S, dx=1, dy=0, ksize=3)
    img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
    plt.subplot(234)
    plt.title('Sobel X Image')
    plt.imshow(img_sobel_x, cmap='gray')

def show_sobel_y(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    # 對 y 方向做 Sobel 邊緣檢測
    img_sobel_y = cv2.Sobel(img, cv2.CV_16S, dx=0, dy=1, ksize=3)
    img_sobel_y = cv2.convertScaleAbs(img_sobel_y)
    plt.subplot(235)
    plt.title('Sobel Y Image')
    plt.imshow(img_sobel_y, cmap='gray')

def show_sobel_combine(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 對 x 方向做 Sobel 邊緣檢測
    img_sobel_x = cv2.Sobel(img, cv2.CV_16S, dx=1, dy=0, ksize=3)
    img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
    # 對 y 方向做 Sobel 邊緣檢測
    img_sobel_y = cv2.Sobel(img, cv2.CV_16S, dx=0, dy=1, ksize=3)
    img_sobel_y = cv2.convertScaleAbs(img_sobel_y)
    # x, y 方向的邊緣檢測後的圖各以一半的全重進行合成
    img_sobel_combine = cv2.addWeighted(img_sobel_x, 0.5, img_sobel_y, 0.5, 0)
    plt.subplot(236)
    plt.title('Sobel Combine Image')
    plt.imshow(img_sobel_combine, cmap='gray')
    plt.show()

if __name__ == '__main__':
    img = cv2.imread('./data/lena.png')
    show_rgb(img)
    show_blur(img)
    show_gray(img)
    show_sobel_x(img)
    show_sobel_y(img)
    show_sobel_combine(img)