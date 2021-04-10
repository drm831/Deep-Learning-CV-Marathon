# 作業
# 實作本篇提到的三大概念 
# 翻轉：實作上下左右的翻轉
# 縮放：比較鄰近差值與雙立方插值 (或雙線性插值) 的圖片品質
# 平移：建立 Translation Transformation Matrix 來做平移

import cv2
import matplotlib.pyplot as plt
import time
import numpy as np

def show_rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(341)
    plt.title('RGB Image')
    plt.imshow(img)

# 上下左右翻轉圖片
def show_horizontal_flip(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 水平翻轉 (horizontal)
    img = cv2.flip(img, 1)
    plt.subplot(342)
    plt.title('Horizontal Flip Image')
    plt.imshow(img)

def show_vertical_flip(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.flip(img, 1)
    # 垂直翻轉 (vertical)
    img = cv2.flip(img, 0)
    plt.subplot(343)
    plt.title('Vertical Flip Image')
    plt.imshow(img)

def show_horizontal_and_vertical_flip(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.flip(img, 1)
    img = cv2.flip(img, 0)
    # 水平 + 垂直翻轉
    img = cv2.flip(img, -1)
    plt.subplot(344)
    plt.title('Horizontal & Vertical Flip Image')
    plt.imshow(img)

# 縮放圖片
# 放大
# 我們先透過縮小圖片去壓縮原有圖片保有的資訊，再放大比較不同方法之間的速度與圖片品質
def show_scale_20(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 將圖片縮小成原本的 20%
    img = cv2.resize(img, None, fx=0.2, fy=0.2)
    plt.subplot(345)
    plt.title('20% Scale Image')
    plt.imshow(img)

def show_scale_80(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 將圖片放大為"小圖片"的 8 倍大 = 原圖的 1.6 倍大
    img = cv2.resize(img, None, fx=8, fy=8)
    plt.subplot(346)
    plt.title('80% Scale Image')
    plt.imshow(img)

def show_nearest_scale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, None, fx=0.2, fy=0.2)
    # 鄰近差值 scale + 計算花費時間
    start_time = time.time()
    img = cv2.resize(img, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_NEAREST)
    print('img NEAREST scale spend {} seconds'.format(time.time() - start_time))
    plt.subplot(347)
    plt.title('NEAREST Scale Image')
    plt.imshow(img)

def show_cubic_scale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, None, fx=0.2, fy=0.2)
    # 雙立方差補 scale + 計算花費時間
    start_time = time.time()
    img = cv2.resize(img, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)
    print('img CUBIC scale spend {} seconds'.format(time.time() - start_time))
    plt.subplot(348)
    plt.title('CUBIC Scale Image')
    plt.imshow(img)

# 平移幾何轉換
# 設定 translation transformation matrix
def show_transformation(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # x 平移 50 pixel; y 平移 100 pixel
    M = np.array([[1, 0, 50], [0, 1, 100]], dtype=np.float32)
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    plt.subplot(349)
    plt.title('Transformation Image')
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    img = cv2.imread('./data/lena.png')
    show_rgb(img)
    show_horizontal_flip(img)
    show_vertical_flip(img)
    show_horizontal_and_vertical_flip(img)
    show_scale_20(img)
    show_scale_80(img)
    show_nearest_scale(img)
    show_cubic_scale(img)
    show_transformation(img)