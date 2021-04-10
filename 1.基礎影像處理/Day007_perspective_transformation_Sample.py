# 範例 
# 根據以下的參考點，嘗試做透視變換

# point1 = np.array([[60, 40], [420, 40], [420, 510], [60, 510]], dtype=np.float32)
# point2 = np.array([[0, 80], [w, 120], [w, 430], [0, 470]], dtype=np.float32)

import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(121)
    plt.title('RGB Image')
    plt.imshow(img)

def show_perspective(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 透視轉換
    h, w = img.shape[:2]
    # 設定四對點，並取得 perspective 矩陣
    point1 = np.array([[60, 40], [420, 40], [420, 510], [60, 510]], dtype=np.float32)
    point2 = np.array([[0, 80], [w, 120], [w, 430], [0, 470]], dtype=np.float32)
    img_perspective_transform = cv2.getPerspectiveTransform(point1, point2)
    # perspective 轉換
    img = cv2.warpPerspective(img, img_perspective_transform, (w, h))
    plt.subplot(122)
    plt.title('Perspective Transform Image')
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    img = cv2.imread('./data/lena.png')
    show_rgb(img)
    show_perspective(img)