import numpy as np
import cv2
from PIL import ImageGrab
import glob
import matplotlib.pyplot as plt
import math
from presskey import PressKey, W, A ,S, D
from findlane import  smooth_max, get_curve, std, gaussian, gaussian_updata, cen_mass


def get_img2hsv():
    image = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    return image


def observation(image, row):
    curve = get_curve(image, row)
    x = np.array([i for i in range(0, 800, 1)])
    avg = cen_mass(curve[0:400])
    sigma = std(curve[0:400], avg)
    g1 = gaussian(x, avg, sigma)
    return g1, avg, sigma, curve


def main():
    iter = 0
    while(True):
        image = get_img2hsv()
        g1, avg, sigma, curve = observation(image,300)
        g1, avg, sigma, curve2 = observation(image,400)
        g1, avg, sigma, curve3 = observation(image,500)
        x = np.array([i for i in range(0, 800, 1)])
        if iter > 0:
            new_avg, new_sigma = gaussian_updata(avg, sigma, pre_avg, pre_sigma)
            g2 = gaussian(x, new_avg, new_sigma)
            pre_avg = new_avg
            pre_sigma = new_sigma + 50
        else:
            g2 = gaussian(x, avg, sigma + 50)
            pre_avg = avg
            pre_sigma = sigma + 50

        plt.plot(curve, 'r')
        plt.plot(curve2, 'b')
        plt.plot(curve3, 'g')
        #plt.plot(g2, 'g')
        plt.xlim([0, 800])
        plt.ion()
        plt.pause(0.001)
        plt.clf()
        cv2.imshow('a', image[:,:,2])
        cv2.waitKey(1)
main()
