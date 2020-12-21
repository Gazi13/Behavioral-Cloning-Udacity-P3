import numpy as np
import cv2
import csv
import math
import sklearn
import matplotlib.pyplot as plt
from math import ceil

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Cropping2D
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda

from keras import __version__ as keras_version
from utils import *
print(str(keras_version).encode('utf8'))


## Parameters
model_path='modelv8/modelv8.h5'
check_points_path = 'modelv8/modelv8-{epoch:03d}.h5'
batch_size= 16
epoch_num = 5


## Read Lines from a csv file
lines = []

with open("/opt/carnd_p3/data/data/driving_log.csv") as cvsFile:
    next(cvsFile)
    reader = csv.reader(cvsFile)
    for line in reader:
        lines.append(line)


## Split Data        
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

## Data generator
train_generator = generator(train_samples, batch_size=batch_size,is_training=True)
validation_generator = generator(validation_samples, batch_size=batch_size,is_training=False)


##---------------------------------------------------------------------------------
"""
## Continue from Checkpoint
model_path = "/home/workspace/CarND-Behavioral-Cloning-P3/modelv7/modelv7-005.h5"
# load retinanet model
model = load_model(model_path)
"""
##---------------------------------------------------------------------------------

## Build Model
model = Sequential()
model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(INPUT_SHAPE)))#160,320,3
#model.add(Cropping2D(cropping=((20,5), (0,0)), input_shape=(INPUT_SHAPE)))
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.summary()


## Save the check point 
checkpoint = ModelCheckpoint(check_points_path,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='auto')

## Compile and train
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
            steps_per_epoch=ceil(len(train_samples)/batch_size),
            validation_data=validation_generator,
            validation_steps=ceil(len(validation_samples)/batch_size),
            callbacks=[checkpoint],
            epochs=epoch_num,
            verbose=1)

## Save model
model.save(model_path)



### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig("plot.png")
#plt.show()
