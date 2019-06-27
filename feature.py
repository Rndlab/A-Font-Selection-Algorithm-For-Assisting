# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt


def getFeatures(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 5, 0.01, 40)
    corners = np.int0(corners)
#     for i in corners:
    # x, y = i.ravel()
    # cv2.rectangle(img, (x-40, y-40), (x+40, y+40), (0, 255, 0), 2)
    # plt.imshow(img), plt.show()

    return corners

k = 1
for i in range(1, 6):
    filename = "Tss\\" + str(i)+".jpg"
    img = cv2.imread(filename)
    positions = getFeatures(img)
    for j in positions:
        x, y = j.ravel()
        imgfeature = img[y-40:y+40, x-40:x+40]
        filename = "Tss\\"+str(i)+"_"+str(k)+".jpg"
        cv2.imwrite(filename, imgfeature, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        if k == 5:
            k = 0
        k += 1
