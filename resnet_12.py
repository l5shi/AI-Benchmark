import keras

from keras.layers import Dense, Conv2D, BatchNormalization, Activation

from keras.layers import AveragePooling2D, Input, Flatten, merge

from keras import backend as K

from keras.models import Model

import os





def convolution_block(x, filters, size, strides=(1,1), padding='same', act=True):

    x = Conv2D(filters, (size,size), strides=strides, padding=padding)(x)

    x = BatchNormalization()(x)

    if act == True:

        x = Activation('relu')(x)

    return x



def residual_block(blockInput, num_filters=64):

    x = convolution_block(blockInput, num_filters, 3)

    x = convolution_block(x, num_filters, 3, act=False)

    x = merge.add([x, blockInput])

    return x



inp=Input((128, 192 ,3))

x=convolution_block(inp, 64, 9)

x=residual_block(x)

x=residual_block(x)

x=residual_block(x)

x=residual_block(x)

x=convolution_block(x, 64, 3)

x=convolution_block(x, 64, 3)

#n1_out=convolution_block(x, 3, 9)

out=Conv2D(3, (9,9), activation='tanh', padding='same')(x)

model = Model(inp, output=out, name='Resnet-12')

model.summary()

model.save('resnet_12.hdf5')