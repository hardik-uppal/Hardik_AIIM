# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 23:32:17 2019

@author: hardi
"""

import imgaug.augmenters as iaa
import cv2
import numpy as np

out_dir = 'D:/TEST_FOLDER/imgaug/'
image_dir = 'D:/EURECOM_Kinect_Face_Dataset_crop/RGB/0001/rgb_0001_s1_LightOn.bmp'

image = cv2.imread(image_dir)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_batch = np.expand_dims(image,axis=0)
#aug_img = iaa.Fliplr(1)(images=image_batch)

augseq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.CoarseDropout(p=0.1, size_percent=0.1),
    iaa.Affine(rotate=-45)
])



#aug_img = iaa.Affine(rotate=-45)(images=image_batch)


aug_img = augseq(images=image_batch)
i=0
cv2.imwrite(out_dir+'example_{}.jpg'.format(i),aug_img[i])


x_batch_rgb, y_batch_rgb
flip_img = iaa.Fliplr(1)(images=x_batch_rgb*255)
rot_img = iaa.Affine(rotate=(-30, 30))(images=x_batch_rgb*255)

shear_aug = iaa.Affine(shear=(-16, 16))(images=x_batch_rgb*255)
trans_aug = iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)})(images=x_batch_rgb*255)
y_batch_rgb
norm_img  = normalized(x_batch_rgb*255)

y_batch_total = np.tile(y_batch_rgb,(2,1))


for i,aug_img in enumerate(x_batch_rgb*255):
    image = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_dir+'example_{}.jpg'.format(i+10),image)
