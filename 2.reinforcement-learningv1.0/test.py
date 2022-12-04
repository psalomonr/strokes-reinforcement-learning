from actor import *
from stroke import *
import argparse
import cv2
import os

import tensorflow as tf

width = 128

parser = argparse.ArgumentParser(description='Learning to Paint')
parser.add_argument('--max_step', default=40, type=int, help='max length for episode')

parser.add_argument('--img', default='image/test.png', type=str, help='test image')
parser.add_argument('--imgid', default=0, type=int, help='set begin number for generated image')
parser.add_argument('--divide', default=4, type=int, help='divide the target image to get better resolution')
args = parser.parse_args()

canvas_cnt = args.divide * args.divide
T = np.ones([1, 1, width, width], dtype='float32')
img = cv2.imread(args.img, cv2.IMREAD_COLOR)
origin_shape = (img.shape[1], img.shape[0])

coord = np.zeros([1, 2, 128, 128])
for i in range(128):
    for j in range(128):
        coord[0, 0, i, j] = i / 127.
        coord[0, 1, i, j] = j / 127.


def decode(x, canvas): # b * (10 + 3)
    


    x = tf.reshape(x, (-1, 10 + 3))


    Decoder = model()
    Decoder.trainable = False 
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    Decoder.compile(loss = "mean_squared_error", optimizer=opt, metrics=['mae'])
    Decoder.load_weights("my_model_weights_blanco&negro.h5")
      
    stroke = 1 - Decoder(x[:, :10])
   
    

    stroke = tf.reshape(stroke, (-1,128, 128, 1))
    color_stroke = stroke * tf.reshape(x[:, -3:], (-1,1,1,3))
    stroke = tf.transpose(stroke, perm=[0, 3, 1, 2])
    color_stroke = tf.transpose(color_stroke, perm=[0, 3, 1, 2])
    stroke =  tf.reshape(stroke, (-1, 5, 1, 128, 128))
    color_stroke = tf.reshape(color_stroke, (-1, 5, 3, 128, 128))
    
    res = []
    for i in range(5):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
        res.append(canvas)
    return canvas, res


def small2large(x):
    x = x.reshape(args.divide, args.divide, width, width, -1)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(args.divide * width, args.divide * width, -1)
    return x

def large2small(x):
    x = x.reshape(args.divide, width, args.divide, width, 3)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(canvas_cnt, width, width, 3)
    return x

def smooth(img):
    def smooth_pix(img, tx, ty):
        if tx == args.divide * width - 1 or ty == args.divide * width - 1 or tx == 0 or ty == 0: 
            return img
        img[tx, ty] = (img[tx, ty] + img[tx + 1, ty] + img[tx, ty + 1] + img[tx - 1, ty] + img[tx, ty - 1] + img[tx + 1, ty - 1] + img[tx - 1, ty + 1] + img[tx - 1, ty - 1] + img[tx + 1, ty + 1]) / 9
        return img

    for p in range(args.divide):
        for q in range(args.divide):
            x = p * width
            y = q * width
            for k in range(width):
                img = smooth_pix(img, x + k, y + width - 1)
                if q != args.divide - 1:
                    img = smooth_pix(img, x + k, y + width)
            for k in range(width):
                img = smooth_pix(img, x + width - 1, y + k)
                if p != args.divide - 1:
                    img = smooth_pix(img, x + width, y + k)
    return img

def save_img(res, imgid, divide=False):
    output = res.numpy() # d * d, 3, width, width    
    output = np.transpose(output, (0, 2, 3, 1))
    if divide:
        output = small2large(output)
        output = smooth(output)
    else:
        output = output[0]
    output = (output * 255).astype('uint8')
    output = cv2.resize(output, origin_shape)
    cv2.imwrite('output/generated' + str(imgid) + '.png', output)


actor = ResNet([2, 2, 2, 2])
actor.build(input_shape=(96, 128, 128, 9))
actor.load_weights("actor_weights-48.h5")

canvas = np.zeros([1, 3, width, width])

patch_img = cv2.resize(img, (width * args.divide, width * args.divide))
patch_img = large2small(patch_img)
patch_img = np.transpose(patch_img, (0, 3, 1, 2))
patch_img = tf.cast(patch_img, tf.float32) / 255.

img = cv2.resize(img, (width, width))
img = img.reshape(1, width, width, 3)
img = np.transpose(img, (0, 3, 1, 2))
img = tf.cast(img, tf.float32)/ 255.

os.system('mkdir output')

if args.divide != 1:
    args.max_step = args.max_step // 2
    for i in range(args.max_step):
        stepnum = T * i / args.max_step
        entrada = np.concatenate([canvas, img, stepnum, coord], 1)
        entrada = tf.transpose(entrada, perm = [0, 3, 2, 1])
        actions = actor(entrada)
        canvas, res = decode(actions, canvas)
        print('canvas step {}, L2Loss = {}'.format(i, np.mean((canvas - img) ** 2)))
        for j in range(5):
            save_img(res[j], args.imgid)
            args.imgid += 1
    if args.divide != 1:
        canvas = canvas[0].numpy()
        canvas = np.transpose(canvas, (1, 2, 0))    
        canvas = cv2.resize(canvas, (width * args.divide, width * args.divide))
        canvas = large2small(canvas)
        canvas = np.transpose(canvas, (0, 3, 1, 2))
        canvas = tf.cast(canvas, tf.float32)
        coord = np.resize(coord,(canvas_cnt, 2, width, width) )
        T = np.resize(T, (canvas_cnt, 1, width, width) )
        for i in range(args.max_step):
            stepnum = T * i / args.max_step
            entrada = np.concatenate([canvas, patch_img, stepnum, coord], 1)
            entrada = tf.transpose(entrada, perm = [0, 3, 2, 1])
            actions = actor(entrada)
            
            canvas, res = decode(actions, canvas)
            print('divided canvas step {}, L2Loss = {}'.format(i, np.mean((canvas - patch_img) ** 2)))
            for j in range(5):
                save_img(res[j], args.imgid, True)
                args.imgid += 1