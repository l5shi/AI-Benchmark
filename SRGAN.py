import os

import sys

import numpy as np

import tensorflow as tf

from keras.models import Sequential, Model

from keras.layers import Input, Activation, Add

from keras.layers import BatchNormalization, LeakyReLU, PReLU, Conv2D, Dense

from keras.layers import UpSampling2D, Lambda

from keras.optimizers import Adam

from keras.applications import VGG19

from keras.applications.vgg19 import preprocess_input

from keras.utils.data_utils import OrderedEnqueuer

from keras import backend as K

from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback




def build_generator(lr_input_size = (512, 512, 3), residual_blocks=16):

    """

    Build the generator network according to description in the paper.



    :param optimizer: Keras optimizer to use for network

    :param int residual_blocks: How many residual blocks to use

    :return: the compiled model

    """

    channels=3

    upscaling_factor=4

    # def SubpixelConv2D( name, scale=2):

    #     """

    #     Keras layer to do subpixel convolution.

    #     NOTE: Tensorflow backend only. Uses tf.depth_to_space

        

    #     :param scale: upsampling scale compared to input_shape. Default=2

    #     :return:

    #     """



    #     def subpixel_shape(input_shape):

    #         dims = [input_shape[0],

    #                 None if input_shape[1] is None else input_shape[1] * scale,

    #                 None if input_shape[2] is None else input_shape[2] * scale,

    #                 int(input_shape[3] / (scale ** 2))]

    #         output_shape = tuple(dims)

    #         return output_shape



    #     def subpixel(x):

    #         return tf.depth_to_space(x, scale)



    #     return Lambda(subpixel, output_shape=subpixel_shape, name=name)


    def residual_block(input):

        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(input)

        x = BatchNormalization(momentum=0.8)(x)

        x = PReLU(shared_axes=[1,2])(x)            

        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)

        x = BatchNormalization(momentum=0.8)(x)

        x = Add()([x, input])

        return x



    def upsample(x, number):

        x = Conv2D(256, kernel_size=3, strides=1, padding='same', name='upSampleConv2d_'+str(number))(x)

        #x = SubpixelConv2D('upSampleSubPixel_'+str(number), 2)(x)

        x = PReLU(shared_axes=[1,2], name='upSamplePReLU_'+str(number))(x)

        return x



    # Input low resolution image

    lr_input = Input(lr_input_size)



    # Pre-residual

    x_start = Conv2D(64, kernel_size=9, strides=1, padding='same')(lr_input)

    x_start = PReLU(shared_axes=[1,2])(x_start)



    # Residual blocks

    r = residual_block(x_start)

    for _ in range(residual_blocks - 1):

        r = residual_block(r)



    # Post-residual block

    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)

    x = BatchNormalization(momentum=0.8)(x)

    x = Add()([x, x_start])

    

    # Upsampling depending on factor

    x = upsample(x, 1)

    if upscaling_factor > 2:

        x = upsample(x, 2)

    if upscaling_factor > 4:

        x = upsample(x, 3)

    

    # Generate high resolution output

    # tanh activation, see: 

    # https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b

    hr_output = Conv2D(

        channels, 

        kernel_size=9, 

        strides=1, 

        padding='same', 

        activation='tanh'

    )(x)



    # Create model and compile

    model = Model(inputs=lr_input, outputs=hr_output)        

    return model    

if __name__ == "__main__":

    model = build_generator()
    model.summary()
    model.save('SRGAN.hdf5')