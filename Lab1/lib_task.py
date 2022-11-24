import cv2 as cv
import time

img = cv.imread('take_the_frog.jpg', cv.IMREAD_GRAYSCALE)
print(img.shape)
start = time.time()
th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
end = time.time()
print("Elapsed = %s" % (end - start))
title = 'Adaptive thresh mean'

cv.imshow(title, th2)
cv.waitKey(0)
cv.destroyAllWindows()
