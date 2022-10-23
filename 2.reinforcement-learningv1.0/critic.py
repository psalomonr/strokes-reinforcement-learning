#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:20:35 2020

@author: patrisaru
"""

import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, Sequential
from tensorflow.keras import Input, layers, Model
from tensorflow.keras.layers import Input, Reshape, Flatten
from tensorflow.keras.layers import Input, Activation, Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import backend as K


from tensorflow.keras import models
import numpy as np

from stroke import *
from rpm import rpm
from wgan import *


#CRÍTIC
class BasicBlockCritic(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlockCritic, self).__init__()
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 =  TReLU()
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.relu2 =  TReLU()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x:x

    def call(self, inputs, training=True):
        out = self.conv1(inputs)
        out = self.bn1(out,training=training)
        out = self.conv2(out)
        identity = self.downsample(inputs)
        output = layers.add([out, identity])
        output = self.relu2(output)
        return output
    
    
class ResNetCritic(keras.Model):
    def __init__(self, layer_dims, num_classes=1): # [2, 2, 2, 2]
        super(ResNetCritic, self).__init__()
        self.x = layers.Conv2D(64, (3, 3), strides=2, padding='same')
        self.trelu =  TReLU()
        self.layer1 = self.build_resblock(64,  layer_dims[0], stride=2)
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)
        self.avgpool = AveragePooling2D(pool_size=(4, 4))
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=True):
        x = self.x(inputs,training=training)
        x = self.trelu(x,training=training)
        x = self.layer1(x,training=training)
        x = self.layer2(x,training=training)
        x = self.layer3(x,training=training)
        x = self.layer4(x,training=training)
        x = self.avgpool(x)
        x = self.fc(x)
        return x
    
    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlockCritic(filter_num, stride))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlockCritic(filter_num, stride=1))
        return res_blocks
    
    
    def build_graph(self, input_shape): 
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = keras.Input(shape=input_shape_nobatch)
        
        _ = self.call(inputs)
        