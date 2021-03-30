# 範例 
# 根據以下的參考點，嘗試做透視變換

# point1 = np.array([[60, 40], [420, 40], [420, 510], [60, 510]], dtype=np.float32)
# point2 = np.array([[0, 80], [w, 120], [w, 430], [0, 470]], dtype=np.float32)

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('./data/lena.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 透視轉換
img_perspective = img_rgb.copy()
h, w = img_rgb.shape[:2]

# 設定四對點，並取得 perspective 矩陣
point1 = np.array([[60, 40], [420, 40], [420, 510], [60, 510]], dtype=np.float32)
point2 = np.array([[0, 80], [w, 120], [w, 430], [0, 470]], dtype=np.float32)

img_perspective_transform = cv2.getPerspectiveTransform(point1, point2)

# perspective 轉換
img_perspective = cv2.warpPerspective(img_rgb, img_perspective_transform, (w, h))

plt.subplot(131)
plt.title('RGB Image')
plt.imshow(img_rgb)
plt.subplot(132)
plt.title('Perspective Transform Image')
plt.imshow(img_perspective)
plt.show()
