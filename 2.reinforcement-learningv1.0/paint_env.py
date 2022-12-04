import cv2
import numpy as np
from multi import *
import tensorflow as tf

class fastenv():
    def __init__(self, 
                 max_episode_length=10, env_batch=64, \
                 writer=None):
        self.max_episode_length = max_episode_length
        self.env_batch = env_batch
        self.env = Paint(self.env_batch, self.max_episode_length)
        self.env.load_data()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.writer = writer
        self.test = False
        self.log = 0
        
    def save_image(self, log, step):
        for i in range(self.env_batch):
            if self.env.imgid[i] == 10:
                canvas = tf.transpose(self.env.canvas[i], perm=[2,1,0])
                canvas = cv2.cvtColor((canvas.numpy()), cv2.COLOR_BGR2RGB)
                cv2.imwrite('{}_canvas_{}.png'.format(str(self.env.imgid[i]), str(step)), canvas)
                
                  
    
    def step(self, action):
        ob, r, d, _ = self.env.step(action)
        if d[0]:
            if not self.test:
                self.dist = self.get_dist()

        return ob, r, d, _

    
    def get_dist(self):
        return np.mean(np.mean(np.mean((((tf.cast(self.env.gt, tf.float32) - tf.cast(self.env.canvas, tf.float32)) / 255) ** 2), axis = 1), axis = 1), axis = 1)
       
        
    def reset(self, test=False, episode=0):
        self.test = test
        ob = self.env.reset(self.test, episode * self.env_batch)
        return ob