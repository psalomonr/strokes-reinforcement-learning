#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:06:10 2020

@author: patrisaru
"""

from tensorflow.keras.layers import Input, Reshape, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Activation, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import keras
import tensorflow as tf


from tensorflow.keras.layers import Layer


from tensorflow.keras.utils  import get_custom_objects

class SubpixelConv2D(Layer):
    """ Subpixel Conv2D Layer
    upsampling a layer from (h, w, c) to (h*r, w*r, c/(r*r)),
    where r is the scaling factor, default to 4
    # Arguments
    upsampling_factor: the scaling factor
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        the second and the third dimension increased by a factor of
        `upsampling_factor`; the last layer decreased by a factor of
        `upsampling_factor^2`.
    # References
        Real-Time Single Image and Video Super-Resolution Using an Efficient
        Sub-Pixel Convolutional Neural Network Shi et Al. https://arxiv.org/abs/1609.05158
    """

    def __init__(self, upsampling_factor=4, **kwargs):
        super(SubpixelConv2D, self).__init__(**kwargs)
        self.upsampling_factor = upsampling_factor

    def build(self, input_shape):
        last_dim = input_shape[-1]
        factor = self.upsampling_factor * self.upsampling_factor
        if last_dim % (factor) != 0:
            raise ValueError('Channel ' + str(last_dim) + ' should be of '
                             'integer times of upsampling_factor^2: ' +
                             str(factor) + '.')

    def call(self, inputs, **kwargs):
        return tf.nn.depth_to_space( inputs, self.upsampling_factor )
        #return Lambda(lambda x:tf.nn.depth_to_space( inputs, self.upsampling_factor ))

    def get_config(self):
        config = { 'upsampling_factor': self.upsampling_factor, }
        base_config = super(SubpixelConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        factor = self.upsampling_factor * self.upsampling_factor
        input_shape_1 = None
        if input_shape[1] is not None:
            input_shape_1 = input_shape[1] * self.upsampling_factor
        input_shape_2 = None
        if input_shape[2] is not None:
            input_shape_2 = input_shape[2] * self.upsampling_factor
        dims = [ input_shape[0],
                 input_shape_1,
                 input_shape_2,
                 int(input_shape[3]/factor)
               ]
        return tuple( dims )

get_custom_objects().update({'SubpixelConv2D': SubpixelConv2D})




    

def model():
    ip = Input(shape=(10,))
    x = Dense(512, activation='relu', name='fc1')(ip)
    x = Dense(1024, activation='relu', name='fc2')(x)
    x = Dense(2048, activation='relu', name='fc3')(x)
    x = Dense(4096, activation='relu', name='fc4')(x)
    x = Reshape((-1,16,16))(x)
    x = Conv2D(32, (3,3), activation='relu',strides=(1, 1), padding = "same")(x)
    x = SubpixelConv2D(upsampling_factor=2)(Conv2D(32, (3,3), strides=(1, 1), padding = "same")(x))
    x = Conv2D(16, (3,3), activation='relu', strides=(1, 1), padding = "same")(x)
    x = SubpixelConv2D(upsampling_factor=2)(Conv2D(16, (3,3), strides=(1, 1), padding = "same")(x))
    x = Conv2D(8, (3,3), activation='relu', strides=(1, 1), padding = "same")(x)
    x = SubpixelConv2D(upsampling_factor=2)(Conv2D(4, (3,3), strides=(1, 1), padding = "same")(x))
    x = Flatten()(x)
    x = Activation('sigmoid')(x)
    x = Reshape((128,128))(x)
    
    model = Model(ip, x)
    
    model.compile(loss = "mean_squared_error", optimizer=Adam(), metrics=['mae'])
    return model

