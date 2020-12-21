## Source : https://github.com/naokishibuya/car-behavioral-cloning/blob/master/utils.py


import os
import cv2
import sklearn
import math
import csv
import numpy as np
from scipy.ndimage import rotate
from scipy.stats import bernoulli


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

#IMG_file="/home/workspace/CarND-Behavioral-Cloning-P3/data/IMG/"
IMG_file="/opt/carnd_p3/data/data/IMG/"


def random_rotation(image, steering_angle, rotation_amount=15):

    angle = np.random.uniform(-rotation_amount, rotation_amount + 1)
    rad = (np.pi / 180.0) * angle
    return rotate(image, angle, reshape=False), steering_angle + (-1) * rad

def random_translate(image, steering_angle, range_x=100, range_y=10):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def flipper(image,angle):
    
    return cv2.flip(image,1),(-angle)#np.fliplr(image)

def preproces(image):
    
    image = image[60:-25, :, :]
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    
    return image

def augumentation(image,angle,aug_num):
    
    image = preproces(image)
    if aug_num == 0:
        #image = preproces(image)
        return image,angle
        
    if aug_num == 1:
        image,angle = random_rotation(image, angle, rotation_amount=10)
        
    if aug_num == 2:
        image, angle = flipper(image, angle)
        
    if aug_num == 3:
        image, angle = random_translate(image, angle, range_x=100, range_y=10)
        
    if aug_num == 4:
        image = random_shadow(image)
        image = random_brightness(image)
        image, angle = flipper(image, angle)
    
    

    
    return image,angle


    
def generator(samples, batch_size=16,is_training=True):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, (batch_size)):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name_center = IMG_file+batch_sample[0].split('\\')[-1]
                name_left = IMG_file+batch_sample[1].split('\\')[-1]
                name_right = IMG_file+batch_sample[2].split('\\')[-1]
                
                center_image =  cv2.imread(name_center)
                left_image = cv2.imread(name_left)
                right_image = cv2.imread(name_right)
                
                
                center_angle = float(batch_sample[3])
                left_angle = center_angle+0.009#*(1.005)
                right_angle = center_angle-0.009#*(0.995)
                
                if not is_training:
                    
                    center_image1,center_angle1 = augumentation(center_image,center_angle,0)
                    left_image1,left_angle1 = augumentation(center_image,center_angle,0)
                    right_image1,right_angle1 = augumentation(center_image,center_angle,0)
                    
                    images.append(center_image1)
                    images.append(left_image1)
                    images.append(right_image1)
                
                    angles.append(center_angle1)
                    angles.append(left_angle1)
                    angles.append(right_angle1)
                
                else:
            
                    for i in range(5):

                        center_image1,center_angle1 = augumentation(center_image,center_angle,i)
                        left_image1,left_angle1 = augumentation(center_image,center_angle,i)
                        right_image1,right_angle1 = augumentation(center_image,center_angle,i)

                        #center_image1 = preproces(center_image1)
                        #left_image1 = preproces(left_image1)
                        #ight_image1 = preproces(right_image1)

                        images.append(center_image1)
                        images.append(left_image1)
                        images.append(right_image1)

                        angles.append(center_angle1)
                        angles.append(left_angle1)
                        angles.append(right_angle1)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
    