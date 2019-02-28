# -*- coding: utf-8 -*-

"""
   Inception V3 model
"""
from keras.models import Model
from keras.layers import Flatten, Dense, Input, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Dense, GlobalAveragePooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.layers.core import Dropout, Lambda
# from keras.engine.topology import merge
from keras.layers import merge
from keras.layers.core import Activation
from keras.regularizers import l2
import warnings
from keras.applications.inception_v3 import InceptionV3

def create_InceptionV3(input_shape):
    model = InceptionV3(include_top=True, weights=None, input_tensor=None, input_shape=input_shape, pooling=None, classes=1000)
    return model


if __name__ == "__main__":
	model = create_InceptionV3((346, 346, 3))
	model.summary()
	model.save('Inception_v3.hdf5')