# 範例
# 透過 SIFT 特徵實作 Brute-Force Matching

import cv2
import matplotlib.pyplot as plt

# 以灰階方式讀入圖片
img_query = cv2.imread('./data/box.png', 0)
img_train = cv2.imread('data/box_in_scene.png', 0)

# 建立 SIFT 物件
sift = cv2.SIFT_create()
# 偵測並計算 SIFT 特徵 (keypoints 關鍵點, descriptor 128 維敘述子)
kp1, des1 = sift.detectAndCompute(img_query, None)
kp2, des2 = sift.detectAndCompute(img_train, None)

# 基於 SIFT 特徵的暴力比對
# D.Lowe ratio test
# knn 比對
# 建立 Brute-Force Matching 物件
bf = cv2.BFMatcher(cv2.NORM_L2)
# 以 knn 方式暴力比對特徵
matches = bf.knnMatch(des1, des2, k=2)
# 透過 D.Lowe ratio test 排除不適合的配對
candidate = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        candidate.append([m])
# 顯示配對結果
img_match = cv2.drawMatchesKnn(img_query, kp1, img_train, kp2, candidate, None, flags=2)

plt.title('Match Image')
plt.imshow(img_match)
plt.show()