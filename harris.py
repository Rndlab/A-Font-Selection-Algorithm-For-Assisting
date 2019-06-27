# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt

def getCorner(url):
    img = cv2.imread(url, 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cornersNum = np.zeros(10)
    lastNum = 0
    rank = 0
    for i in range(0, 10):
        level = 0.99 - i*0.03
        corners = cv2.goodFeaturesToTrack(gray, 100, level, 10)
        corners = np.int0(corners)
        cornersNum[i] = len(corners)-lastNum
        lastNum = len(corners)
        print "角点数量"+str(len(corners))+"角点质量"+ str(level)

    for i in range(0, 10):
        rank += (1-i/20)*cornersNum[i]

    for i in corners:
        x, y = i.ravel()
       cv2.rectangle(img, (x-20, y-20), (x+20, y+20), (0, 255, 0), 2)
    return rank