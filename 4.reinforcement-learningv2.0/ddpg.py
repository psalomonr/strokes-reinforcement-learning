import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, Sequential
from tensorflow.keras import Input, layers, Model
from tensorflow.keras.layers import Input, Reshape, Flatten
from tensorflow.keras.layers import Input, Activation, Dense
from tensorflow.keras import backend as K
 
from tensorflow.keras import models
import numpy as np
from stroke import *
from rpm import rpm
from wgan import *
from actor import *
from critic import *
from train import *



coord = np.zeros([1, 2, 128, 128])
for i in range(128):
    for j in range(128):
        coord[0, 0, i, j] = i / 127.
        coord[0, 1, i, j] = j / 127.
#tf.keras.backend.clear_session()
Decoder = model()

def ssim_loss(y_true, y_pred):
       return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

def soft_update(target, source, tau=0.001):
    for target_param, param in zip(target, source):
        target_param.assign( target_param * (1.0 - tau) + param * tau)
        


def hard_update(target, source):
    for target_param, param in zip(target.variables, source.variables):
        target_param.assign(param)

tf.keras.backend.clear_session()
def decode(x, canvas): # b * (10 + 3)
    
    x = tf.reshape(x, (-1, 10 + 3))


    Decoder = load_model('eric_model_good.h5', custom_objects={"ssim_loss": ssim_loss} )
    stroke = 1 - Decoder(x[:, :10])
    tf.keras.backend.clear_session()

    stroke = tf.reshape(stroke, (-1,128, 128, 1))
    color_stroke = stroke * tf.reshape(x[:, -3:], (-1,1,1,3))
    stroke = tf.transpose(stroke, perm=[0, 3, 1, 2])
    color_stroke = tf.transpose(color_stroke, perm=[0, 3, 1, 2])
    stroke =  tf.reshape(stroke, (-1, 5, 1, 128, 128))
    color_stroke = tf.reshape(color_stroke, (-1, 5, 3, 128, 128))
    
    for i in range(5): #esto es la k que describen en el paper
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
   
    return canvas

def cal_trans(s, t):
    return (s.transpose(0, 3) * t).transpose(0, 3)

@tf.function
def update_target(target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))
    
class DDPG(object):
    def __init__(self, batch_size=64, env_batch=1, max_step=40, \
                 tau=0.001, discount=0.9, rmsize=800, \
                 writer=None, resume=None, output_path=None):

        self.max_step = max_step
        self.env_batch = env_batch
        self.batch_size = batch_size  
        
        
        self.actor_optim  = tf.keras.optimizers.Adam(lr=1e-3)
        self.critic_optim  = tf.keras.optimizers.Adam(lr=3e-4)


        self.actor = ResNet([2, 2, 2, 2])
        self.actor.build(input_shape=(20, 128, 128, 9))

        
        self.actor_target = ResNet([2, 2, 2, 2])
        self.actor_target.build(input_shape=(20, 128, 128, 9))
  
        
        
        self.critic = ResNetCritic([2, 2, 2, 2])
        self.critic.build(input_shape=(20, 128, 128, 12))
        # add the last canvas for better prediction
        self.critic_target =  ResNetCritic([2, 2, 2, 2])
        self.critic_target.build(input_shape=(20, 128, 128, 12))

        
        if (resume != None):
            self.load_weights(resume)           

        # Create replay buffer
        self.memory = rpm(rmsize * max_step)

        # Hyper-parameters
        self.tau = tau
        self.discount = discount
        
        self.state = [None] * self.env_batch # Most recent state
        self.action = [None] * self.env_batch # Most recent action


    def play(self, state, target=False, trainning=True):

        state = tf.concat((tf.cast(state[:, :6], tf.float32)/ 255, tf.cast(state[:, 6:7], tf.float32) / self.max_step, np.resize(coord, (state.shape[0],2,128,128))), axis=1)

        if target:
            state = tf.transpose(state, perm = [0, 3, 2, 1])
            state = self.actor_target(state, training=trainning)
            return state
        else:
            state = tf.transpose(state, perm = [0, 3, 2, 1])
            state = self.actor(state, training=trainning)
            return state
 
    
    def update_gan(self, state):
        canvas = state[:, :3]
        gt = state[:, 3 : 6]
        fake, real, penal = update(tf.cast(canvas, tf.float32) / 255, tf.cast(gt, tf.float32) / 255)


    def evaluate(self, state, action, target=False):
        T = state[:, 6 : 7]
        gt = tf.cast(state[:, 3 : 6], tf.float32) / 255
        canvas0 = tf.cast(state[:, :3], tf.float32)/ 255
        canvas1 = decode(action, canvas0)
        gan_reward = cal_reward(canvas1, gt) - cal_reward(canvas0, gt)
       
        coord_ = np.resize(coord, (state.shape[0],2,128,128))
        merged_state = tf.concat([canvas0, canvas1, gt, tf.cast((T + 1), tf.float32) / self.max_step, coord_], 1)
        if target:
            merged_state = tf.transpose(merged_state, perm = [0, 3, 2, 1])
            Q = self.critic_target(merged_state,training=True)

            return (Q + gan_reward), gan_reward

        else:
            merged_state = tf.transpose(merged_state, perm = [0, 3, 2, 1])
            Q = self.critic(merged_state, training=True)
            return (Q + gan_reward), gan_reward
   
    tf.config.run_functions_eagerly(True)
    @tf.function
    def update_policy(self,lr):
        
        self.actor_optim  = tf.keras.optimizers.Adam(lr=lr[1])
        self.critic_optim  = tf.keras.optimizers.Adam(lr=lr[0])
        
        
        state, action, reward, \
            next_state, terminal = self.memory.sample_batch(self.batch_size)
       

        self.update_gan(next_state)
        
        with tf.GradientTape() as tape:
            
            
            next_action = self.play(next_state, True, True)
   
           
            target_q, _ = self.evaluate(next_state, next_action, True)
            target_q = self.discount * tf.reshape((1 - tf.cast(terminal,tf.float32)),[-1,1]) * target_q
            
            cur_q, step_reward = self.evaluate(state, action)
            target_q += step_reward
            critic_loss = tf.math.reduce_mean(tf.math.square(cur_q - target_q))
            
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optim.apply_gradients(zip(grads, self.critic.trainable_variables))
        
        
        with tf.GradientTape() as tape:
        
            action = self.play(state,False,True)
 
            pre_q, _ = self.evaluate(state, action)

            actor_loss = tf.reduce_mean(-pre_q) 
   

        grads2 = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optim.apply_gradients(zip(grads2, self.actor.trainable_variables))
         
        soft_update(self.actor_target.variables, self.actor.variables)
        soft_update(self.critic_target.variables, self.critic.variables)

        return -actor_loss,critic_loss

    def observe(self, reward, state, done, step):
        s0 = self.state
        a = self.action
        r = reward
        s1 = state
        d = done.astype('float32')
        for i in range(self.env_batch):
            self.memory.append([s0[i], a[i], r[i], s1[i], d[i]])
        self.state = state

    def noise_action(self, noise_factor, state, action):
        noise = np.zeros(action.shape)
        for i in range(self.env_batch):
            action[i] = action[i] + np.random.normal(0, self.noise_level[i], action.shape[1:]).astype('float32')
        return np.clip(action.astype('float32'), 0, 1)
    
    def select_action(self, state, return_fix=False, noise_factor=0):
        action = self.play(state,False,False)  
        if noise_factor > 0:        
            action = self.noise_action(noise_factor, state, action)

        self.action = action
        if return_fix:
            return action
        return self.action

    def reset(self, obs, factor):
        self.state = obs
        self.noise_level = np.random.uniform(0, factor, self.env_batch)
        
    def save_model(self):
        self.actor.save_weights('actor_weights.h5')
        

