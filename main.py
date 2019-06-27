#!/usr/local/bin/python

import cv2
import numpy
from sklearn import cross_validation
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.externals import joblib

MIN_DESCRIPTOR = 10  
TRAINING_SIZE = 2


def findDescriptor(img):
    contour = []
    _, contour, hierarchy = cv2.findContours(
        img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
        contour)
    contour_array = contour[0][:, 0, :]
    contour_complex = numpy.empty(contour_array.shape[:-1], dtype=complex)
    contour_complex.real = contour_array[:, 0]
    contour_complex.imag = contour_array[:, 1]
    fourier_result = numpy.fft.fft(contour_complex)
    return fourier_result


def truncate_descriptor(descriptors, degree):
    descriptors = numpy.fft.fftshift(descriptors)
    center_index = len(descriptors) / 2
    descriptors = descriptors[
        center_index - degree / 2:center_index + degree / 2]
    descriptors = numpy.fft.ifftshift(descriptors)
    return descriptors


def reconstruct(descriptors, degree):
    descriptor_in_use = truncate_descriptor(descriptors, degree)
    contour_reconstruct = numpy.fft.ifft(descriptor_in_use)
    contour_reconstruct = numpy.array(
        [contour_reconstruct.real, contour_reconstruct.imag])
    contour_reconstruct = numpy.transpose(contour_reconstruct)
    contour_reconstruct = numpy.expand_dims(contour_reconstruct, axis=1)
    if contour_reconstruct.min() < 0:
        contour_reconstruct -= contour_reconstruct.min()
    contour_reconstruct *= 800 / contour_reconstruct.max()
    contour_reconstruct = contour_reconstruct.astype(numpy.int32, copy=False)
    black = numpy.zeros((800, 800), numpy.uint8)
    cv2.drawContours(black, contour_reconstruct, -1, 255, thickness=5)
    return descriptor_in_use




def sample_generater(sample1):
    response = numpy.array([0, 1])
    response = numpy.tile(response, TRAINING_SIZE / 2)
    response = response.astype(numpy.float32)
    training_set = numpy.empty(
        [TRAINING_SIZE, MIN_DESCRIPTOR], dtype=numpy.float32)
    # assign descriptors with noise to our training_set
    for i in range(0, TRAINING_SIZE - 1, 2):
        descriptors_sample1 = findDescriptor(sample1)
        descriptors_sample1 = truncate_descriptor(
            descriptors_sample1,
            MIN_DESCRIPTOR)

    return training_set, response



def getFourier(sample1):
    _, sample1 = cv2.threshold(sample1, 127, 255, cv2.THRESH_BINARY_INV)
    training_set, response = sample_generater()
    fourier_result = findDescriptor(sample1)
    contour_reconstruct = reconstruct(fourier_result, MIN_DESCRIPTOR)

    return contour_reconstruct

def getFeatures (filename):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 3, 0.7, 40)
    corners = numpy.int0(corners)
    descriptor = numpy.zeros([3, 10])
    j = 0
    for i in corners:
        x, y = i.ravel()
        imgfeature = gray[x-20:x+20, y-20:y+20]
        descriptor[j] = getFourier(imgfeature)

        j += 1

    return descriptor

for i in range(1, 6):
    for j in range(1, 6):
        filename = "Bbs/" + str(i)+"_"+str(j)+".jpg"
        sample = cv2.imread(filename, 0)
        print(filename)
        if i == 1 and j == 1:
            array = getFourier(sample)
            print(array)
        else:
            print(getFourier(sample))
            array = numpy.row_stack((array, getFourier(sample)))
Y = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5
     ]

clf = svm.SVC(gamma=0.01)
clf.fit(array, Y)
joblib.dump(clf, "Bbs.m")

# sample = cv2.imread('qw.jpg', 0)
# X_test = getFourier(sample)


# desc = getFeatures('wdj.jpg')
# print desc[1]
# predicted = clf.predict(X_test)


# print(predicted)




