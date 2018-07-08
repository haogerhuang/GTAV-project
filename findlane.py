import numpy as np
import cv2
from PIL import ImageGrab
import glob
import matplotlib.pyplot as plt
import math



def smooth_max(curve, filter_size):
    smooth_curve = np.zeros_like(curve)
    if filter_size % 2 == 0:
        print error
    else:
        for i,j in enumerate(curve):
            if i >= filter_size/2 and i < (curve.shape[0] - filter_size/2 ):
                filt = 1
                while filt <= filter_size/2:
                    smooth_curve[i] = curve[i]
                    if smooth_curve[i] < curve[i + filt]:
                        smooth_curve[i] = curve[i + filt]
                    if smooth_curve[i] < curve[i - filt]:
                        smooth_curve[i] = curve[i - filt]
                    filt = filt + 1

        return smooth_curve



def std(x, mu):
    index = np.array([i for i in range(0, x.size)])
    return (np.sum(np.multiply(np.square(index - mu), x.astype(float)))/np.sum(x))**0.5


def gaussian(x, mu, sigma):
    pi = math.pi
    gauss = 1/((2 * pi * sigma**2)**0.5)*(np.exp(-(x.astype(float)-mu)**2/(2*sigma**2)))
    return gauss

def gaussian_updata(mu, sigma, mu2, sigma2):
    new_mu =  1/(sigma**2 + sigma2**2) * (sigma2**2 * mu + sigma**2 * mu2)
    new_sigma = (1/(1/sigma**2 + 1/sigma2**2))**0.5
    return new_mu, new_sigma


def get_curve(i,row):
    i2 = i[:, :, 0]
    i3 = i[:, :, 1]
    i = cv2.cvtColor(i, cv2.COLOR_RGB2HSV)
    i4 = i[:, :, 2]
    curve = np.multiply(i2[row, :].astype(float) / 255, i3[row, :].astype(float) / 255)
    curve = np.multiply(curve, i4[row, :].astype(float)/255)
    curve = smooth_max(curve, 21)
    curve = np.convolve(curve, [0.02 for i in range(50)], mode='same')
    return curve

def cen_mass(x):
    total = 0
    for i,j in enumerate(x):
        total = total + i*j
    return float(total)/np.sum([x])


def cen_size(x, avg):
    total = 0
    num = 0
    for i,j in enumerate(x):
        total = total + j*(i-avg)**2
        num = num + (i - avg)**2
    avg = float(total)/float(num)
    return avg

def fit(x, curve, avg):
    if np.sum(np.multiply(gaussian(x, avg, 30), curve)) >= np.sum(np.multiply(gaussian(x, avg + 10, 30), curve))\
        and np.sum(np.multiply(gaussian(x, avg, 30), curve)) >= np.sum(np.multiply(gaussian(x, avg - 10, 30), curve)):
        return avg
    elif np.sum(np.multiply(gaussian(x, avg, 30), curve)) < np.sum(np.multiply(gaussian(x, avg + 10, 30), curve)):
        return fit(x, curve, avg + 10)
    else:
        return fit(x, curve, avg - 10)

def check_training_data(training_data,name):
    iter = 0
    for im, j in training_data:
        im = cv2.resize(im, (800, 600))

        curve = get_curve(im, 500)
        curve2 = get_curve(im, 450)
        curve3 = get_curve(im, 350)

        x = np.array([i for i in range(0, 800, 1)])

        avg = cen_mass(curve[0:400])
        sigma = std(curve[0:400], avg)
        g1 = gaussian(x, avg, sigma)

        if iter > 0:
            new_avg, new_sigma = gaussian_updata(avg, sigma, pre_avg, pre_sigma)
            g2 = gaussian(x, new_avg, new_sigma)
            pre_avg = new_avg
            pre_sigma = new_sigma + 50
        else:
            g2 = gaussian(x, avg, sigma + 50)
            pre_avg = avg
            pre_sigma = sigma + 50

        #plt.plot(curve, 'r')
        #plt.plot(curve2, 'g')
        #plt.plot(curve3, 'b')
        #plt.plot(curve4, 'k')
        plt.plot(curve3, 'y')

        plt.xlim([0, 800])
        plt.ion()
        plt.pause(0.1)
        plt.clf()
        cv2.line(im, (0, 350),(250, 350), (255, 255, 255), thickness=1)
        cv2.line(im, (250, 350), (500, 350), (255, 0, 0), thickness=1)
        cv2.line(im, (500, 350), (799, 350), (255, 0, 255), thickness=1)
        cv2.imshow('a', im)
        cv2.waitKey(10)


        iter = iter + 1

training_data = np.load('data11.npy')
check_training_data(training_data, 'a')
