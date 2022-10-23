

from stroke import *
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import cv2
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import keras.backend as K
from ddpg import decode




width = 128
convas_area = width * width

img_train = []
img_test = []
train_num = 0
test_num = 0


class Paint:
    def __init__(self, batch_size, max_step):
        self.batch_size = batch_size
        self.max_step = max_step
        self.action_space = (13)
        self.observation_space = (self.batch_size, width, width, 7)
        self.test = False

    #función para leer el conjunto de datos de entrenamiento (CelebA) 
    def load_data(self):
        global train_num, test_num
        for i in range(200000):
            img_id = '%06d' % (i + 1)
            try:
                img = cv2.imread('//home/patsalrub/DRL/data/img_align_celeba/' + img_id + '.jpg', cv2.IMREAD_UNCHANGED)               
                img = cv2.resize(img, (width, width))
                if i > 2000:
                    train_num += 1
                    img_train.append(img)
                else:
                    test_num += 1
                    img_test.append(img)
            finally:
                if (i + 1) % 10000 == 0:                    
                    print('loaded {} images'.format(i + 1))

        print('finish loading data, {} training images, {} testing images'.format(str(train_num), str(test_num)))
       
        
    def pre_data(self, id, test):

        if test:
            img = img_test[id]

        else:
            img = img_train[id]
        img = np.asarray(img)

        return np.transpose(img, (2, 1, 0))
 
    def reset(self, test=False, begin_num=False):
        self.test = test
        self.imgid = [0] * self.batch_size
        self.gt = np.zeros([self.batch_size, 3, width, width], dtype='uint8')
        for i in range(self.batch_size):
            if test:
                id = (i + begin_num)  % test_num
            else:
                id = np.random.randint(train_num)
            self.imgid[i] = id
            self.gt[i] = self.pre_data(id, test)
        self.tot_reward = np.mean(np.mean(np.mean((((self.gt / 255) ** 2)), axis = 1), axis = 1), axis = 1)
        self.stepnum = 0
        self.canvas = np.zeros([self.batch_size, 3, width, width], dtype='uint8')
        self.lastdis = self.ini_dis = self.cal_dis()
        return self.observation()
    
    
    def observation(self):
        # canvas B * 3 * width * width
        # gt B * 3 * width * width
        # T B * 1 * width * width
        ob = []
        T = np.ones([self.batch_size, 1, width, width], dtype='uint8') * self.stepnum
        return np.concatenate((self.canvas, self.gt, T), 1) # canvas, img, T
    
    
    def cal_trans(self, s, t):
        return (s.transpose(0, 3) * t).transpose(0, 3)
    
    
    def step(self, action):
        self.canvas = tf.cast((decode(action, self.canvas / 255) * 255), tf.uint8)
        self.stepnum += 1
        ob = self.observation()
        done = (self.stepnum == self.max_step)
        reward = self.cal_reward() # np.array([0.] * self.batch_size)
        return ob, reward, np.array([done] * self.batch_size), None


    def cal_dis(self):
        return np.mean(np.mean(np.mean(((self.canvas - self.gt) / 255) ** 2, axis = 1), axis = 1), axis = 1)
    
    
    def cal_reward(self):
        dis = self.cal_dis()
        
        reward = ( self.lastdis - dis) / (self.ini_dis + 1e-8)
       
        self.lastdis = dis
        return reward