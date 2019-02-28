from __future__ import print_function

import keras

from keras import backend as K

from keras.models import Sequential, Model

from keras.layers import Dense, Activation

from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, merge, add

import tensorflow as tf

from keras.models import load_model

from keras import optimizers

from keras import losses

from keras.optimizers import SGD, Adam

from keras.callbacks import ModelCheckpoint



import os, glob, sys, threading

import scipy.io

from scipy import ndimage, misc

import numpy as np

import re

import math




input_img = Input(shape=(192,192,3))



model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)

model = Activation('relu')(model)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)

model = Activation('relu')(model)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)

model = Activation('relu')(model)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)

model = Activation('relu')(model)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)

model = Activation('relu')(model)



model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)

model = Activation('relu')(model)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)

model = Activation('relu')(model)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)

model = Activation('relu')(model)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)

model = Activation('relu')(model)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)

model = Activation('relu')(model)



model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)

model = Activation('relu')(model)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)

model = Activation('relu')(model)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)

model = Activation('relu')(model)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)

model = Activation('relu')(model)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)

model = Activation('relu')(model)



model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)

model = Activation('relu')(model)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)

model = Activation('relu')(model)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)

model = Activation('relu')(model)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)

model = Activation('relu')(model)

model = Conv2D(1, (3, 3), padding='same', kernel_initializer='he_normal')(model)

res_img = model



output_img = merge.add([res_img, input_img])



model = Model(input_img, output_img)


model.summary()

model.save('vgg19.hdf5')