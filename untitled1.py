# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 05:32:27 2019

@author: hardi
"""
import numpy as np


import matplotlib.pyplot as plt


from model_2_mutimodal_attention_CurtinFaces_dataaug import VGGFace_multimodal ## our model
from keract import get_activations,display_activations,display_heatmaps
from model_vgg_face import VGG16 ##VGG model
import os

from keras import optimizers
import cv2

from vis.utils import utils

import matplotlib.cm as cm
from vis.visualization import visualize_cam
from keras import activations


image_rgb = 'D:/CurtinFaces_crop/RGB/test2/01/69.jpg'

image_rgb_2 = 'D:/CurtinFaces_crop/RGB/test2/01/59.jpg'

image_depth = 'D:/CurtinFaces_crop/normalized/DEPTH/test2/01/69.jpg'
img1 = utils.load_img(image_rgb, target_size=(224, 224))
img2 = utils.load_img(image_rgb_2, target_size=(224, 224))

f, ax = plt.subplots(1, 2)
ax[0].imshow(img1)
ax[1].imshow(img2)
##define the model
model_vgg_multimodal = VGGFace_multimodal(input_shape=(224,224,3), n_class=52)
model_vgg_multimodal.load_weights('D:/tutorial/rgb+depth+thermal/CurtinFaces/dataaug_vgg_multimodal_dropout-0.5_3fc_batch30/weights-best.h5')
model_vgg_multimodal.compile(optimizer=optimizers.Adam(lr=0.01), loss=['categorical_crossentropy'], metrics=['accuracy'])
model_vgg_multimodal.summary()



layer_idx = utils.find_layer_idx(model_vgg_multimodal, 'output')
# Swap softmax with linear
model_vgg_multimodal.layers[layer_idx].activation = activations.linear
model_vgg_multimodal = utils.apply_modifications(model_vgg_multimodal)









##get activations
#layer_name = 'multiply_4'#'conv2d_2'
#activation_att = get_activations(model_vgg_multimodal, [image_arr_rgb,image_arr_depth],layer_name)


#
#visualize_cam(model, layer_idx, filter_indices, seed_input, penultimate_layer_idx=None, \
#    backprop_modifier=None, grad_modifier=None)



for modifier in [None, 'guided', 'relu']:
    plt.figure()
    f, ax = plt.subplots(1, 2)
    plt.suptitle("vanilla" if modifier is None else modifier)
    for i, img in enumerate([img1, img2]):    
        # 20 is the imagenet index corresponding to `ouzel`
        grads = visualize_cam(model_vgg_multimodal, layer_idx, filter_indices=20, 
                              seed_input=img, backprop_modifier=modifier)        
        # Lets overlay the heatmap onto original image.    
        jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
        ax[i].imshow(overlay(jet_heatmap, img))
