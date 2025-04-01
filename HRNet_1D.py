from __future__ import print_function
import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import merge, Input,UpSampling2D, Conv2D,concatenate,ZeroPadding2D, Conv2DTranspose,\
    AveragePooling2D,add, SeparableConv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.layers.noise import GaussianDropout
from keras.models import Model
import numpy as np
from model_logic2 import *
g_init = tf.random_normal_initializer(1.,0.02)    ##生成标准正态分布的随机数
from keras import layers
from keras.applications.imagenet_utils import (decode_predictions,
                                               preprocess_input)
from keras.preprocessing import image
from keras.utils.data_utils import get_file
s=17                                                          #kernel value
def conv2_x(input_tensor, filters, stage, block, w_init):      #structure of Conv_Block
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    con = SeparableConv2D(filters, (s, 1), strides=(1, 1), activation=None,
                            name=conv_name_base + '2a', kernel_initializer=w_init,
                            bias_initializer='zeros', padding='same')(input_tensor)
    bn = BatchNormalization(gamma_initializer=g_init,name=bn_name_base + '2a')(con)
    ac = LeakyReLU(alpha=0.2)(bn)

    con = SeparableConv2D(filters, (s, 1), strides=(1, 1), activation=None,
                            name=conv_name_base + '2b', kernel_initializer=w_init,
                            bias_initializer='zeros', padding='same')(ac)
    bn = BatchNormalization(gamma_initializer=g_init,name=bn_name_base + '2b')(con)
    conv2_x_add = layers.add([bn, input_tensor])

    ac = LeakyReLU(alpha=0.2)(conv2_x_add)
    ac = Dropout(0.4)(ac)
    return ac

def ResNet50(img_rows, img_cols, color_type=1, num_class=1):
    w_init = 'glorot_normal'
    filters = 64

    global bn_axis
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
        img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    else:
        bn_axis = 1
        img_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')

    x = SeparableConv2D(filters, (s, 1), strides=(1, 1), activation=None, kernel_initializer=w_init,
                                  bias_initializer='zeros', padding='same')(img_input)
    x = BatchNormalization(name='bn_conv1')(x)
    x = LeakyReLU(alpha=0.2)(x)

    x1 = conv2_x(x, filters, 2, 'b', w_init)
    x2 = conv2_x(x1, filters, 2, 'c', w_init)
    x3 = conv2_x(x2, filters, 2, 'd', w_init)
    add1 = layers.add([x3, x])
    add1 = LeakyReLU(alpha=0.2)(add1)
    add1 = Dropout(0.4)(add1)
    x4 = conv2_x(add1, filters, 2, 'f', w_init)
    x5 = conv2_x(x4, filters, 2, 'g', w_init)
    x6 = conv2_x(x5, filters, 2, 'h', w_init)
    add2 = layers.add([x6, add1])
    add2 = LeakyReLU(alpha=0.2)(add2)
    add2 = Dropout(0.4)(add2)
    x7 = conv2_x(add2 , filters, 2, 'i', w_init)
    x8 = conv2_x(x7, filters, 2, 'k', w_init)
    x9 = conv2_x(x8, filters, 2, 'n', w_init)
    x10 = conv2_x(x9, filters, 2, 'm', w_init)
    add3 = layers.add([x10, add2])
    add3 = LeakyReLU(alpha=0.2)(add3)
    add3 = Dropout(0.4)(add3)
    x11 = SeparableConv2D(16, (s, 1), strides=(1, 1), activation=None, kernel_initializer=w_init,
                        bias_initializer='zeros', padding='same')(add3)
    x11 = BatchNormalization(name='bn_conv2')(x11)
    x11 = LeakyReLU(alpha=0.2)(x11)
    x12 = SeparableConv2D(4, (s, 1), strides=(1, 1), activation=None, kernel_initializer=w_init,
                        bias_initializer='zeros', padding='same')(x11)
    x12 = BatchNormalization(name='bn_conv3')(x12)
    x12 = LeakyReLU(alpha=0.2)(x12)

    x13 = SeparableConv2D(num_class, (1, 1), strides=(1, 1), activation='relu', name='output',
                        kernel_initializer = w_init, bias_initializer='zeros', padding='same',
                        kernel_regularizer=l2(1e-3))(x12)

    model = Model(input=[img_input], output=x13)
    return model
