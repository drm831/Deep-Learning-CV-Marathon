# 作業 
# 練習以旋轉變換 + 平移變換來實現仿射變換
# > 旋轉 45 度 + 縮放 0.5 倍 + 平移 (x+100, y-50)
# 直接建構仿射矩陣 (圖片至少包含 3 個點以上的 pair)
# cv2.getAffineTransform (譬如：圖片1的點，圖片2的點，input 應該包含兩張對應圖片的點)

import cv2
import matplotlib.pyplot as plt
import time
import numpy as np

def show_rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(131)
    plt.title('RGB Image')
    plt.imshow(img)

# Affine Transformation - Case 2: any three point
# 給定兩兩一對，共三對的點
# 這邊我們先用手動設定三對點，一般情況下會有點的資料或是透過介面手動標記三個點
def show_label(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rows, cols = img.shape[:2]
    point_1 = np.array([[50,50], [300,100], [200,300]], dtype=np.float32)
    point_2 = np.array([[80,80], [330,150], [300,300]], dtype=np.float32)
    print(point_1, point_2)
    # 取得 affine 矩陣並做 affine 操作
    M_affine = cv2.getAffineTransform(point_1, point_2)
    img_affine = cv2.warpAffine(img, M_affine, (cols, rows))
    # 在圖片上標記點
    img_copy = img.copy()
    for idx, pts in enumerate(point_1):
        pts = tuple(map(int, pts))
        cv2.circle(img_copy, pts, 3, (0, 255, 0), -1)
        cv2.putText(img_copy, str(idx), (pts[0]+5, pts[1]+5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    plt.subplot(132)
    plt.title('Label Image')
    plt.imshow(img_copy)

def show_affine(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rows, cols = img.shape[:2]
    point_1 = np.array([[50,50], [300,100], [200,300]], dtype=np.float32)
    point_2 = np.array([[80,80], [330,150], [300,300]], dtype=np.float32)
    print(point_1, point_2)
    # 取得 affine 矩陣並做 affine 操作
    M_affine = cv2.getAffineTransform(point_1, point_2)
    img_affine = cv2.warpAffine(img, M_affine, (cols, rows))
    # 在圖片上標記點
    img_copy = img.copy()
    for idx, pts in enumerate(point_1):
        pts = tuple(map(int, pts))
        cv2.circle(img_copy, pts, 3, (0, 255, 0), -1)
        cv2.putText(img_copy, str(idx), (pts[0]+5, pts[1]+5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

    for idx, pts in enumerate(point_2):
        pts = tuple(map(int, pts))
        cv2.circle(img_affine, pts, 3, (0, 255, 0), -1)
        cv2.putText(img_affine, str(idx), (pts[0]+5, pts[1]+5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    plt.subplot(133)
    plt.title('Affine Label Image')
    plt.imshow(img_affine)
    plt.show()

if __name__ == '__main__':
    img = cv2.imread('./data/lena.png')
    show_rgb(img)
    show_label(img)
    show_affine(img)