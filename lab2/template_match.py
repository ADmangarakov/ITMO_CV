import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from skimage.metrics import structural_similarity as ssim

IMG_NAME ="img/cup.jpg"
SUB_IMG_NAME = "img/small_cup.jpg"


def find_window(full_img, sub_img):
    full_w, full_h = full_img.shape[:2]
    sub_w, sub_h = sub_img.shape[:2]
    print(full_w, full_h)
    print(sub_w, sub_h)
    win_pos = (0, 0)
    winW = 0
    found = False
    while winW < full_w - sub_w and not found:
        winH = 0
        while winH < full_h - sub_h:
            window = full_img[winW:winW + sub_w, winH:winH + sub_h]
            if ssim(sub_img, window) > 0.80:
                found = True
                print("found", ssim(sub_img, window))
                win_pos = (winH, winW)
                break
            winH += 1
        winW += 1
    return win_pos


full_image = cv2.imread(IMG_NAME, 0)
sub_image = cv2.imread(SUB_IMG_NAME, 0)
fig, ax = plt.subplots()

ax.imshow(img.imread(IMG_NAME))
win_pos = find_window(full_image, sub_image)
sub_w, sub_h = sub_image.shape[:2]
# Create a Rectangle patch
rect = patches.Rectangle(win_pos, sub_h, sub_w, linewidth=1, edgecolor='r', facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

plt.show()
