# 範例
# 實作各種不一樣的方式來調整飽和 / 對比 / 明亮
# 改變到 HSL color space 來調整飽和度
# 對灰階圖實作直方圖均衡
# alpha/ beta 調整對比 / 明亮

import cv2
import matplotlib.pyplot as plt

def show_rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(241)
    plt.title('RGB Image')
    plt.imshow(img)

def show_hls(img):    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    plt.subplot(242)
    plt.title('HLS Image')
    plt.imshow(img)

# 改變飽和度
# 轉換成 HLS color space, 改變 s channel 的值
# 為了要改變飽和度，我們先把 color space 轉成 HSL 格式 (OpenCV 表示順序是 HLS)
def show_hls_down(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    change_percentage = 0.2
    # 針對飽和度的值做改變，超過界線 0~1 的都會 bound
    # 在 HLS color space 減少飽和度
    img = img.astype('float32')
    # 轉成小數點並調降兩成飽和度
    img[..., -1] = img[..., -1] / 255 - change_percentage
    img[img[..., -1] < 0] = 0
    img[..., -1] = img[..., -1] * 255
    img = img.astype('uint8')
    # 轉換
    img = cv2.cvtColor(img, cv2.COLOR_HLS2BGR)
    plt.subplot(243)
    plt.title('HLS Down Image')
    plt.imshow(img)

def show_hls_up(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    change_percentage = 0.2
    # 在 HLS color space 增加飽和度
    img = img.astype('float32')
    img[..., -1] = img[..., -1] / 255 + change_percentage
    img[img[..., -1] > 1] = 1
    img[..., -1] = img[..., -1] * 255
    img = img.astype('uint8')
    # 轉換
    img = cv2.cvtColor(img, cv2.COLOR_HLS2BGR)
    plt.subplot(244)
    plt.title('HLS Up Image')
    plt.imshow(img)

# 直方圖均衡
# 轉為灰階圖片
def show_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.subplot(245)
    plt.title('Gray Image')
    plt.imshow(img, cmap='gray')

# 灰階圖片直方圖均衡
def show_gray_equal(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_equalizeHist = cv2.equalizeHist(img)
    plt.subplot(246)
    plt.title('Gray Equal Image')
    plt.imshow(img_equalizeHist, cmap='gray')

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
    plt.title('Add Lighness Image')
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    img = cv2.imread('./data/lena.png')
    show_rgb(img)
    show_hls(img)
    show_hls_down(img)
    show_hls_up(img)
    show_gray(img)
    show_gray_equal(img)
    show_add_contrast(img)
    show_add_lighness(img)