import cv2 as cv
import numpy as np
from numba import njit
import time

KERNEL_SIZE = (11, 11)
C = 2

img = cv.imread('take_the_frog.jpg', cv.IMREAD_GRAYSCALE)
M, N = img.shape
K, L = KERNEL_SIZE

pdimg = np.pad(img, K // 2, 'minimum')
print(pdimg)
h_k = K // 2


@njit
def adapt_bin(img_to_bin, dest):
    for i in range(h_k - 1, M + h_k):
        for j in range(h_k - 1, N + h_k):
            m = img_to_bin[i - h_k + 1:i + h_k, j - h_k + 1:j + h_k].mean() - C
            if img_to_bin[i, j] < m:
                dest[i, j] = 0
            else:
                dest[i, j] = 255
    return dest


# compile
start = time.time()
pdimg = adapt_bin(pdimg, pdimg.copy())
end = time.time()
print("Elapsed (before compilation) = %s" % (end - start))
start = time.time()
pdimg = adapt_bin(pdimg, pdimg.copy())
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
print(pdimg.shape)
print(M, N)
pdimg = pdimg[int((pdimg.shape[0] - M) / 2):M + int((pdimg.shape[0] - M) / 2),
        int((pdimg.shape[1] - N) / 2):N + int((pdimg.shape[1] - N) / 2)]
cv.imshow('test', pdimg)
cv.imshow('diff', abs(cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2) - pdimg))
cv.imshow('lib', cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2))
cv.imwrite('result.jpg', pdimg)
cv.waitKey(0)
cv.destroyAllWindows()
