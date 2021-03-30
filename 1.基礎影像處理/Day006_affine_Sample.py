# 範例 
# 練習以旋轉變換 + 平移變換來實現仿射變換
# 旋轉 45 度 + 縮放 0.5 倍 + 平移 (x+100, y-50)
# 透過旋轉矩陣處理

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('./data/lena.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Affine Transformation - Case 1: 旋轉 45度 --> 縮小0.5 > 平移 (x+100, y-50)
rows, cols = img_rgb.shape[:2]

# 取得旋轉矩陣
# getRotationMatrix2D(center, angle, scale)
img_rotate = cv2.getRotationMatrix2D((cols//2, rows//2), 45, 0.5)
print('Rotation Matrix')
print(img_rotate)
print()

# 取得平移矩陣
img_translation = np.array([[1, 0 , 100], [0, 1, 100]], dtype=np.float32)
print('Translation Matrix')
print(img_translation)

# 旋轉
img_rotate = cv2.warpAffine(img_rgb, img_rotate, (cols, rows))
# 平移
img_translation = cv2.warpAffine(img_rgb, img_translation, (cols, rows))

plt.subplot(131)
plt.title('RGB Image')
plt.imshow(img_rgb)
plt.subplot(132)
plt.title('Rotate Image')
plt.imshow(img_rotate)
plt.subplot(133)
plt.title('Translation Image')
plt.imshow(img_translation)
plt.show()