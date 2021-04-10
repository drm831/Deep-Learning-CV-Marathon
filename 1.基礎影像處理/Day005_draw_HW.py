# 範例 
# Hint: 人物原始邊框座標 (60, 40), (420, 510)
# 請根據 Lena 圖做以下處理
# 對明亮度做直方圖均衡處理
# 水平鏡像 + 縮放處理 (0.5 倍)
# 畫出人物矩形邊框

import cv2
import matplotlib.pyplot as plt

def show_rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(241)
    plt.title('RGB Image')
    plt.imshow(img)

# Hint: 矩形
def show_rectangle(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.rectangle(img, (60, 40), (420 , 510), (255, 0, 0), 3)
    plt.subplot(242)
    plt.title('Rectangle Image')
    plt.imshow(img)

# Hint: 線
def show__line(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.line(img, (60, 40), (420 , 510), (255, 0, 0), 3)
    plt.subplot(243)
    plt.title('Line Image')
    plt.imshow(img)

# Hint: 文字
def show_text(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.putText(img, '(60, 40)', (60, 40), 0, 1, (255, 0, 0), 2)
    plt.subplot(244)
    plt.title('Text Image')
    plt.imshow(img)

# 假設我們希望先對圖片做以下幾點預處理，請印出最後結果
# Hint: 注意先後順序，人物原始邊框座標 (60, 40), (420, 510)
# 對明亮度做直方圖均衡處理
# 水平鏡像 + 縮放處理 (0.5 倍)
# 畫出人物矩形邊框

# 對明亮度做直方圖均衡處理
# 原始 BGR 圖片轉 HSV 圖片
def show_equalizeHist(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 對明亮度做直方圖均衡 -> 對 HSV 的 V 做直方圖均衡
    img[..., -1] = cv2.equalizeHist(img[..., -1])
    # 將圖片轉回 BGR
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    plt.subplot(245)
    plt.title('EqualizeHist Image')
    plt.imshow(img)

# 水平鏡像 + 縮放處理 (0.5 倍)
# 水平鏡像 (圖片)
def show_resize(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[..., -1] = cv2.equalizeHist(img[..., -1])
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img = img[:, ::-1, :]
    # 縮放處理
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    plt.subplot(246)
    plt.title('Resize Image')
    plt.imshow(img)

    # 畫出人物矩形邊框
def show_resize_rectangle(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[..., -1] = cv2.equalizeHist(img[..., -1])
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img = img[:, ::-1, :]
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    img = cv2.rectangle(img, (45, 19), (228, 255),(255, 0, 0), 2)
    plt.subplot(247)
    plt.title('Resize Rectangle Image')
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    img = cv2.imread('./data/lena.png')
    show_rgb(img)
    show_rectangle(img)
    show__line(img)
    show_text(img)
    show_equalizeHist(img)
    show_resize(img)
    show_resize_rectangle(img)