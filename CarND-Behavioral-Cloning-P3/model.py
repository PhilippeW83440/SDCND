#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 19:42:48 2017

@author: philippew
"""

import csv
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

# np.random.seed(5)

# Settings

N_AUG = 6 # 1 image => 6 images (center, left, right) * flip
BATCH_SIZE = 32
PATIENCE = 3
NB_EPOCHS = 40
ANGLE_CORRECTION = [0.0, 0.2, -0.2] # center, left, right
CROP_TOP = 70
CROP_BOTTOM = 25


def load_data(directory, logfile, samples):
  with open(directory+logfile) as csvfile:
      next(csvfile, None) # ignore 1st line (headers)
      reader = csv.reader(csvfile)
      for line in reader:
          # Filter out 90% of 0 angles as they are overrepresented
          angle = float(line[3])
          if angle == 0 and np.random.uniform() <= 0.90:
              continue
          line.append(directory)
          #print(line)
          samples.append(line)

          
# Use a generator to avoid any memory contraints when dealing with huge datasets          
def generator(samples, batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]            
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                # Deal with center, left, right cameras
                filedir = batch_sample[-1]
                for i in range(3):
                    filename = batch_sample[i].strip()
                    #print(filedir+filename)
                    image = cv2.imread(filedir+filename)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # TEST PWE
                    angle = float(batch_sample[3]) + ANGLE_CORRECTION[i]
                    images.append(image) 
                    angles.append(angle)
                    
                    # Data augmentation via flip
                    flipped_image = cv2.flip(image, 1)
                    images.append(flipped_image)
                    angles.append(-angle)
                    
                    # Data augmentation via random brighteness
                    #brightened_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
                    #random_bright = .25+np.random.uniform()
                    #brightened_image[:,:,2] = brightened_image[:,:,2]*random_bright
                    #brightened_image = cv2.cvtColor(brightened_image,cv2.COLOR_HSV2RGB)
            X_samples = np.array(images)
            y_samples = np.array(angles)
            yield shuffle(X_samples, y_samples)

                     
samples = []
#load_data('./driving_data/data/', 'driving_log.csv', samples)
load_data('./driving_data/track1_drive/', 'driving_log.csv', samples)
load_data('./driving_data/track1_recovery/', 'driving_log.csv', samples)

load_data('./driving_data/track2_lap1/', 'driving_log.csv', samples)
load_data('./driving_data/track2_lap2/', 'driving_log.csv', samples)
load_data('./driving_data/track2_lap3/', 'driving_log.csv', samples)

train_samples, valid_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=BATCH_SIZE)
valid_generator = generator(valid_samples, batch_size=BATCH_SIZE)


from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU, SReLU
#from keras.layers.pooling import MaxPooling2D

# Model based on Nvidia's end-to-end architecture
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((CROP_TOP, CROP_BOTTOM), (0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='elu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='elu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='elu'))
model.add(Convolution2D(64,3,3, activation='elu'))
model.add(Convolution2D(64,3,3, activation='elu'))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense( 1))
#model.add(Dense( 1, activation='tanh')) # make sure final result is in between -1 and 1

model.compile(optimizer='adam', loss='mse')

model.summary()


filepath="model.epoch{epoch:02d}-valloss{val_loss:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1)

history_object = model.fit_generator(train_generator, 
                    samples_per_epoch = N_AUG * len(train_samples), 
                    validation_data = valid_generator, 
                    nb_val_samples = N_AUG * len(valid_samples), 
                    nb_epoch = NB_EPOCHS,
                    verbose = 1,
                    callbacks = [checkpoint, early_stopping])

#model.save('model.h5')

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('history.png')
plt.show()
