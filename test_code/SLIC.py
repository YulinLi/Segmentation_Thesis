import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from skimage.filters import sobel
from skimage.segmentation import slic
from skimage.util import img_as_float

src = cv2.imread(
    'C:\\Users\\Mislab\\Desktop\\Research\\dataset\\LFW\\train\\image/Aaron_Eckhart_0001.jpg')

#src = cv2.GaussianBlur(src, (5, 5), 0)
cv2.imshow('ex', src)
cv2.waitKey()
segments_slic = slic(src, n_segments=250, compactness=10, sigma=1)
img = color.label2rgb(segments_slic, src, kind='avg')
cv2.imshow('ex', img)
cv2.waitKey()
segments_slic_1 = slic(img, n_segments=100, compactness=10, sigma=1)
img_1 = color.label2rgb(segments_slic_1, img, kind='avg')
cv2.imshow('ex', img_1)
cv2.waitKey()
segments_slic_2 = slic(img_1, n_segments=50, compactness=10, sigma=1)
img_2 = color.label2rgb(segments_slic_2, img_1, kind='avg')
cv2.imshow('ex', img_2)
cv2.waitKey()
segments_slic_3 = slic(img_2, n_segments=10, compactness=10, sigma=1)
img_3 = color.label2rgb(segments_slic_3, img_2, kind='avg')
cv2.imshow('ex', img_3)
cv2.waitKey()
segments_slic_4 = slic(img_3, n_segments=5, compactness=10, sigma=1)
img_4 = color.label2rgb(segments_slic_4, img_3, kind='avg')
cv2.imshow('ex', img_4)
cv2.waitKey()
