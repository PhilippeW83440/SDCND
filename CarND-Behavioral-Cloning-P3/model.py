#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 19:42:48 2017

@author: philippew
"""

import csv
import cv2
import numpy as np

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

del lines[0]
        
images = []
measurements = []

for line in lines:
    #print(line)
    source_path = line[0]
    filename = source_path.split('/')[-1]
    local_path = "./data/IMG/" + filename
    #print(local_path)
    image = cv2.imread(local_path)
    images.append(image)
    measurement = float(line[3])
    #print(measurement)
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)
print(X_train.shape)
print(y_train.shape)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Convolution2D, MaxPooling2D, Dropout, Lambda

model = Sequential()
# Normalizing lambda layer
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# First convolutional layer
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
# Second convolutional layer
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
# Fully-connected layers
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7, verbose=2)
model.save('model.h5')
