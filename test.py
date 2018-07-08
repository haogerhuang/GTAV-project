import numpy as np
import glob
import os
from PIL import Image
import cv2
import colorsys
from findhistogram import findmaxregion, point2lineDis, point2pointDis, linefilter
from PIL import ImageGrab


def roi(img,vertices):
  mask = np.zeros_like(img)
  cv2.fillPoly(mask, vertices, 255)
  masked = cv2.bitwise_and(img,mask)
  return masked

def find_thres(hsv):
  hsv1 = hsv[hsv.shape[0] / 2:hsv.shape[0], :]
  ret, mask = cv2.threshold(hsv1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  return ret

def im_process(mask,scale,scale2):
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, scale))
  kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale2, scale2))
  open1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=kernel)
  open2 = cv2.morphologyEx(open1, cv2.MORPH_GRADIENT, kernel=kernel1)
  #open2 = cv2.Laplacian(open1, cv2.CV_64F)
  #open2 = cv2.Canny(open1,100,300)
  return open2

def draw_lines(frame,lines):
  if lines[0] is not None:
    for x1, y1, x2, y2 in lines[0]:
      if y1 > 360 or y2 > 360:
        if float(x1 - x2) != 0:
          if abs(float(y1 - y2) / float(x1 - x2)) > 0.2:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)
  return frame




while(True):


  frame = np.array(ImageGrab.grab(bbox=(0,0,800,600)))
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  hsv = cv2.pyrDown(hsv)
  hsv_2 = hsv[:,:,2]
  thres = find_thres(hsv_2)
  ret, mask = cv2.threshold(hsv_2, thres, 255, cv2.THRESH_BINARY)
  mask = cv2.pyrUp(mask)
  mask1 = im_process(mask,5,5)
  
  lines = cv2.HoughLinesP(mask1,1,np.pi/180,100,minLineLength=30,maxLineGap=1)
  frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
  frame = draw_lines(frame,lines)
  cv2.imshow('a',mask)
  cv2.waitKey(10)



  
  

  
