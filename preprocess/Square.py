import cv2
import numpy as np

height = 128
width = 128

img = np.zeros((height, width, 3), np.uint8)
img.fill(255)  # white

#cv2.circle(img, (64, 64), 40, (0, 0, 255), -1)
cv2.rectangle(img, (32, 32), (96, 96), (0, 0, 255), -1)

cv2.imwrite("image/rectangle.png", img)
