import cv2
import numpy as np
import os
from os.path import isfile, join

pathIn = 'C:/Users/Mislab/Desktop/Research/dataset/image/'
pathOut = 'C:/Users/Mislab/Desktop/Research/dataset/Flower/test/image/'

files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
# for sorting the file names properly
files.sort(key=lambda x: x[5:-4])
# batch image
for i in range(150):
    filename = pathIn + files[i]
    print(filename)
    img = cv2.imread(filename)
    w = 128
    dim = (w, w)
    red = (0, 0, 255)
    line_width = 1

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    """cv2.line(resized, (0, 0), (w, w), red, line_width)
    cv2.line(resized, (int(w/4), 0), (int(w*3/4), w), red, line_width)
    cv2.line(resized, (int(w/2), 0), (int(w/2), w), red, line_width)
    cv2.line(resized, (int(w*3/4), 0), (int(w/4), w), red, line_width)
    cv2.line(resized, (w, 0), (0, w), red, line_width)
    cv2.line(resized, (w, int(w/4)), (0, int(w*3/4)), red, line_width)
    cv2.line(resized, (0, int(w/2)), (w, int(w/2)), red, line_width)
    cv2.line(resized, (w, int(w*3/4)), (0, int(w/4)), red, line_width)"""

    cv2.imwrite(pathOut + files[i], resized)
