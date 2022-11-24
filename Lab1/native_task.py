import cv2 as cv
import numpy as np
import time

KERNEL_SIZE = (25, 25)
C = 7

img = cv.imread('puchkov.jpeg', cv.IMREAD_GRAYSCALE)
M, N = img.shape
K, L = KERNEL_SIZE


pdimg = np.pad(img, K // 2, 'minimum')
print(pdimg)
h_k = K // 2
start = time.time()
cp_img = pdimg.copy()
for i in range(h_k - 1, M + h_k):
    for j in range(h_k - 1, N + h_k):
        m = pdimg[i - h_k + 1:i + h_k, j - h_k + 1:j + h_k].mean() - C
        if pdimg[i, j] < m:
            cp_img[i, j] = 0
        else:
            cp_img[i, j] = 255
end = time.time()
print("Elapsed = %s" % (end - start))

cv.imshow('test', cp_img)
cv.waitKey(0)
cv.destroyAllWindows()
