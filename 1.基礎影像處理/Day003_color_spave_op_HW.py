# 作業
# 實作各種不一樣的方式來調整飽和 / 對比 / 明亮
# 改變 color space 來調整飽和度
# 實作直方圖均衡
# alpha/ beta 調整對比 / 明亮

import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(241)
    plt.title('RGB Image')
    plt.imshow(img)

# # 改變飽和度
# # 轉換成 HSV color space, 改變 s channel 的值
# # 為了要改變飽和度，我們先把 color space 轉成 HSV 格式
def show_hsv(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    plt.subplot(242)
    plt.title('HSV Image')
    plt.imshow(img)

def show_hsv_down(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    change_percentage = 0.2
    # 針對飽和度的值做改變，超過界線 0~1 的都會 bound
    # 在 HSV color space 減少飽和度
    img = img.astype('float32')
    img[..., 1] = img[..., 1] / 255 - change_percentage
    img[img[..., 1] < 0] = 0
    img[..., 1] = img[..., 1] * 255
    img = img.astype('uint8')
    # 轉換 color space 回 RGB
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    plt.subplot(243)
    plt.title('HSV Down Image')
    plt.imshow(img)

def show_hsv_up(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    change_percentage = 0.2
    # 在 HSV color space 增加飽和度
    img = img.astype('float32')
    img[..., 1] = img[..., 1] / 255 + change_percentage
    img[img[..., 1] > 1] = 1
    img[..., 1] = img[..., 1] * 255
    img = img.astype('uint8')
    # 轉換 color space 回 RGB
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    plt.subplot(244)
    plt.title('HSV Up Image')
    plt.imshow(img)

# # 直方圖均衡
# # case 1: 把彩圖拆開對每個 channel 個別做直方圖均衡再組合起來
# # case 2: 轉換 color space 到 HSV 之後對其中一個 channel 做直方圖均衡

# # case 1
# # 每個 channel 個別做直方圖均衡
def show_rgb_equal(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = [img[..., 0], img[..., 1], img[..., 2]]
    img = [cv2.equalizeHist(i) for i in img]
    # 組合經過直方圖均衡的每個 channel
    img = np.stack(img, axis=-1)
    plt.subplot(245)
    plt.title('RGB Equal Image')
    plt.imshow(img)

# case 2 - 轉換 color space 後只對其中一個 channel 做直方圖均衡
def show_hsv_equal(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[..., -1] = cv2.equalizeHist(img[..., -1])
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    plt.subplot(246)
    plt.title('HSV Equal Image')
    plt.imshow(img)

# 調整對比 / 明亮
# alpha: 控制對比度 (1.0~3.0)
def show_add_contrast(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.convertScaleAbs(img, alpha=2.0, beta=0)
    plt.subplot(247)
    plt.title('Add Contrast Image')
    plt.imshow(img)

# beta: 控制明亮度 (0~255)
def show_add_lighness(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.convertScaleAbs(img, alpha=1.0, beta=50)
    plt.subplot(248)
    plt.title('Add Contrast Image')
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    img = cv2.imread('./data/lena.png')
    show_rgb(img)
    show_hsv(img)
    show_hsv_down(img)
    show_hsv_up(img)
    show_rgb_equal(img)
    show_hsv_equal(img)
    show_add_contrast(img)
    show_add_lighness(img)