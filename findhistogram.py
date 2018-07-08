import numpy as np
import glob
import os
from PIL import Image
import cv2
import colorsys
import matplotlib.pyplot as plt
import timeit

import operator



def findmaxregion(frame):

    frame1 = frame[:, :, 0]
    frame2 = frame[:, :, 1]
    frame3 = frame[:, :, 2]
    hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
    hist3 = cv2.calcHist([frame3], [0], None, [256], [0, 256])
    list_of_hist = [hist1,hist2,hist3]

    low = np.array([0,0,0])
    high = np.array([0,0,0])
    for item in range(3):
        max_region = 0
        max_region_sum = 0
        plus = 0
        ratio = 0.0
        while(ratio<=0.8):
         for i in range(0,256-70-plus,5):
            current_region_sum = 0
            for j in range(70+plus):
               current_region_sum = current_region_sum + list_of_hist[item][i+j]
            if current_region_sum > max_region_sum:
                max_region_sum = current_region_sum
                max_region = i
         low[item] = max_region
         high[item] = max_region+70+plus
         ratio = float(max_region_sum)/float(frame1.shape[0]*frame1.shape[1])
         plus = plus+10
    return low,high

def point2lineDis(x1,y1,x2,y2,x0,y0):
    D = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)/((y2-y1)**2 +(x2-x1)**2)**0.5
    return D

def point2pointDis(x1,y1,x2,y2):
    D = ((x1-x2)**2 + (y1-y2)**2)**0.5
    return D

def linefilter(l1,l2,D):
    lines = np.array([[0,0,0,0]])
    for x1,y1,x2,y2 in l1:
        for x_1,y_1,x_2,y_2 in l2:
            x0 = (x_1 + x_2)/2
            y0 = (y_1 + y_2)/2
            Dis = point2lineDis(x1,y1,x2,y2,x0,y0)
            if Dis <= D:
              lines = np.append(lines,[[x_1,y_1,x_2,y_2]],axis=0)
    lines = np.delete(lines,0,0)
    return lines




