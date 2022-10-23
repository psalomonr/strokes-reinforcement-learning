from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Reshape, Flatten, ZeroPadding2D, LeakyReLU
from tensorflow.keras.layers import Input, Conv2D, Activation, Dense, AveragePooling2D, Dropout, BatchNormalization
import tensorflow.keras
import tensorflow as tf


import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, Sequential
from tensorflow.keras import Input, layers, Model
from tensorflow.keras.layers import Input, Reshape, Flatten
from tensorflow.keras.layers import Input, Activation, Dense
from tensorflow.keras.layers import Input, Conv2D, Activation, Dense, AveragePooling2D, Dropout
import tensorflow_addons as tfa


def soft_update(target, source, tau=0.001):
    for target_param, param in zip(target.variables, source.variables):
        new_value = target_param *  (1.0 - tau) + param * tau        
        target_param.assign(new_value)
        

class TReLU(keras.layers.Layer):
    def __init__(self):
        super(TReLU, self).__init__()
        self.alpha = tf.zeros_initializer()
        self.alpha = tf.Variable(
            initial_value=self.alpha(shape=(1), dtype="float32"), trainable=True
        )

    def call(self, x):
        return tf.keras.activations.relu(x - self.alpha) + self.alpha


def Discriminator():
    ip = Input(shape=(128,128, 6))
    x =  Conv2D(16, kernel_size=(5, 5), strides=(2, 2), padding = "same", name='fc1')(ip)
    x = TReLU()(x)
    #x = layers.Activation('relu')(x)
    x =  Conv2D(32, kernel_size=(5, 5), strides=(2, 2),   padding = "same", name='fc2')(x)
    x = TReLU()(x)
    #x = layers.Activation('relu')(x)
    x =  Conv2D(64, kernel_size=(5, 5), strides=(2, 2),   padding = "same", name='fc3')(x)
    x = TReLU()(x)
   # x = layers.Activation('relu')(x)
    x =  Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding = "same", name='fc4')(x)
    x = TReLU()(x)
   # x = layers.Activation('relu')(x)
    x =  Conv2D(1, kernel_size=(5, 5), strides=(2, 2),   padding = "same", name='fc5')(x)
    x = TReLU()(x)
   # x = layers.Activation('relu')(x)
    x = AveragePooling2D(pool_size=(4, 4))(x)
   # x = layers.GlobalAveragePooling2D()(x)
    x = Reshape((-1,1))(x)
    model = Model(ip, x)
    return model

dim = 128
LAMBDA = 10 # Gradient penalty lambda hyperparameter

netD= Discriminator()
target_netD = Discriminator()

import numpy as np

def cal_gradient_penalty(netD, real_data, fake_data, batch_size):

    alpha = np.random.rand(96,1)
    alpha = np.resize(alpha, (96, 98304))
    alpha = np.resize(alpha, (96, 128, 128, 6))
    

    #diff = fake_data - real_data
    interpolated = alpha * real_data + ((1 - alpha) * fake_data)


    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
        pred = netD(interpolated, training=True)
        
        # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
    grads = tf.reshape(grads, (96,-1))
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
    gp = tf.reduce_mean((norm - 1.0) ** 2) * 10 #LAMDA

     
    return gp
    


def cal_reward(fake_data, real_data):

    real_data = tf.transpose(real_data, perm = [0, 3, 2, 1])
    fake_data = tf.transpose(fake_data, perm = [0, 3, 2, 1])

    var =  target_netD(tf.concat([real_data, fake_data], 3), training=False)

    return tf.reshape(var, [96,1])


tf.config.run_functions_eagerly(True)
@tf.function
def update(fake_data, real_data):
    
    fake = tf.concat([real_data, fake_data], 1)
    
    real = tf.concat([real_data, real_data], 1)
    
    
    optimizerD = tensorflow.keras.optimizers.Adam(lr=3e-4,beta_1=0.5, beta_2=0.999)
   
    with tf.GradientTape() as tape:
        real = tf.transpose(real, perm = [0, 3, 2, 1])

        D_real = netD(real, training=True)
 
        D_real = tf.reshape(D_real, [96,1])
 
        fake = tf.transpose(fake, perm = [0, 3, 2, 1])

        D_fake = netD(fake, training=True)
        D_fake = tf.reshape(D_fake, [96,1])


        gradient_penalty = cal_gradient_penalty(netD, real, fake, real.shape[0])
    #optimizerD.zero_grad()
        D_cost = tf.math.reduce_mean(D_fake) - tf.math.reduce_mean(D_real) + gradient_penalty
    grads2 = tape.gradient(D_cost, netD.trainable_variables)
    optimizerD.apply_gradients(zip(grads2, netD.trainable_variables))
    
    soft_update(target_netD, netD, 0.001)
  
   
    return tf.math.reduce_mean(D_fake), tf.math.reduce_mean(D_real), gradient_penalty

