# 作業
# 改變圖片的 color space (HLS, LAB) 並呈現, 包含 RGB, HSV, HSL, LAB
import cv2
import matplotlib.pyplot as plt

def show_rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(141)
    plt.title('RGB Image')
    plt.imshow(img)

def show_hsv(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    plt.subplot(142)
    plt.title('HSV Image')
    plt.imshow(img)

def show_hls(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    plt.subplot(143)
    plt.title('HLS Image')
    plt.imshow(img)

def show_lab(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    plt.subplot(144)
    plt.title('LAB Image')
    plt.imshow(img)
    plt.show()
    plt.close()

if __name__ == '__main__':
    img = cv2.imread('./data/lena.png')
    show_rgb(img)
    show_hsv(img)
    show_hls(img)
    show_lab(img)