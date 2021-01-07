import cv2
import numpy as np
import os
from os.path import isfile, join
#import util.Snake_tool as tool

pathIn = 'C:/Users/Mislab/Desktop/Research/dataset/lfw_/'
pathOut = 'C:/Users/Mislab/Desktop/Research/dataset/LFW_total/'

if not os.path.exists(pathOut):
    os.makedirs(pathOut)

i = 0

for f in os.listdir(pathIn):
    temp_dir = pathIn + f + '/'
    for files in os.listdir(temp_dir):
        if isfile(join(temp_dir, files)):
            filename = temp_dir + files
            img = cv2.imread(filename)
            w = 128
            dim = (w, w)
            red = (0, 0, 255)
            line_width = 1
            i += 1
            r = w/2
            theta = 2*np.pi/16
            center = (r, r)
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            """cv2.line(resized, tool.P2C(r, 0*theta, center),
                     tool.P2C(r, 8*theta, center), red, line_width)
            cv2.line(resized, tool.P2C(r, 1*theta, center),
                     tool.P2C(r, 9*theta, center), red, line_width)
            cv2.line(resized, tool.P2C(r, 2*theta, center),
                     tool.P2C(r, 10*theta, center), red, line_width)
            cv2.line(resized, tool.P2C(r, 3*theta, center),
                     tool.P2C(r, 11*theta, center), red, line_width)
            cv2.line(resized, tool.P2C(r, 4*theta, center),
                     tool.P2C(r, 12*theta, center), red, line_width)
            cv2.line(resized, tool.P2C(r, 5*theta, center),
                     tool.P2C(r, 13*theta, center), red, line_width)
            cv2.line(resized, tool.P2C(r, 6*theta, center),
                     tool.P2C(r, 14*theta, center), red, line_width)
            cv2.line(resized, tool.P2C(r, 7*theta, center),
                     tool.P2C(r, 15*theta, center), red, line_width)"""
            cv2.imwrite(pathOut + files, resized)
