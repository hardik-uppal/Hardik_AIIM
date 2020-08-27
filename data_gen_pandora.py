# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:37:58 2019

@author: hardi
"""

import re
import os
import numpy as np
import cv2
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(100, 100), grayscale = True)
    img = img_to_array(img)
     
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(img, alpha=10), cv2.COLORMAP_JET)
    image_depth = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
    image_depth = img_to_array(image_depth)
    return image_depth

def training_generator_with_files(args, data_list_file='data/validation_0_1_2.npy'):
    data = np.load(data_list_file, allow_pickle=True)

#    train_data_files = list(chunks(list(data[::, 0]), args.batch_size))
#    train_label = list(chunks(list(data[::, 1]), args.batch_size))
    train_data_files_1 = list(chunks(list(data[:,0]), args.batch_size))####change 20 to args.batch_size
    train_data_files_2 = list(chunks(list(data[:,1]), args.batch_size))
    train_label = list(chunks(list(data[::, 2]), args.batch_size))
    '''
    'leftEar' : labels[list(range(0,6))]
    'rightEar': labels[list(range(27,33))]
    'Chin' : labels[list(range(6,27))]
    'leftEye' : labels[list(range(60,68))+[96,]]
    'leftEyeBrow' : labels[list(range(33,42))]
    'rightEye' : labels[list(range(68,76))+[97,]]
    'rightEarBrow' : labels[list(range(42,51))]
    'Nose' : labels[list(range(51,60))]
    'Lips' : labels[list(range(76,96))]
    '''
    i=0
    data_chunk, label_chunk = [], []
    while i < args.batch_size:
#    for i in range(0,20):
        try:
            # print([os.path.join('data/WFLW_normal/train', x) for x in data_chunk])
#            data = [np.array(cv2.imread(os.path.join('D:/Scaled_landmark_datasets/WFLW_normal/train', x))) for x in data_chunk]
            data_chunk_1 =  train_data_files_1[i]
            data_chunk_2 =  train_data_files_2[i]
            label_chunk = train_label[i]
#            print(data_chunk, label_chunk)
            ########### From images
            image_1_address = [os.path.join('D:/face_dataset_depth_16bit/face_dataset_16/', x) for x in data_chunk_1]
            image_2_address = [os.path.join('D:/face_dataset_depth_16bit/face_dataset_16/', x) for x in data_chunk_2]
            data_1 = [np.array(preprocess_image(x)) for x in image_1_address]
            data_2 = [np.array(preprocess_image(x)) for x in image_2_address]
            
            
            ############# From feature files
#            data = [np.array(np.loadtxt(os.path.join('D:/Net_features/WFLW/train/ResNet_50_conv4_1_blk_', x.split('.')[0]+'.txt'))) for x in data_chunk]
            

            labels_source = np.array(label_chunk)
            yield [[np.array(data_1),np.array(data_2)], labels_source]
        except ValueError and IndexError:
            data_chunk = train_data_files_1.pop(0)
            data_chunk = train_data_files_2.pop(0)
            label_chunk = train_label.pop(0)
            train_data_files_1.append([data_chunk])
            train_data_files_2.append([data_chunk])
            train_label.append([label_chunk])
            
def test_generator_with_files(args, data_list_file='data/test_0_1_2.npy'):
    data = np.load(data_list_file, allow_pickle=True)

#    train_data_files = list(chunks(list(data[::, 0]), args.batch_size))
#    train_label = list(chunks(list(data[::, 1]), args.batch_size))
    train_data_files_1 = list(chunks(list(data[:,0]), args.batch_size))####change 20 to args.batch_size
    train_data_files_2 = list(chunks(list(data[:,1]), args.batch_size))
    train_label = list(chunks(list(data[::, 2]), args.batch_size))
    '''
    'leftEar' : labels[list(range(0,6))]
    'rightEar': labels[list(range(27,33))]
    'Chin' : labels[list(range(6,27))]
    'leftEye' : labels[list(range(60,68))+[96,]]
    'leftEyeBrow' : labels[list(range(33,42))]
    'rightEye' : labels[list(range(68,76))+[97,]]
    'rightEarBrow' : labels[list(range(42,51))]
    'Nose' : labels[list(range(51,60))]
    'Lips' : labels[list(range(76,96))]
    '''
    i=0
    data_chunk, label_chunk = [], []
    while i < args.batch_size:
#    for i in range(0,20):
        try:
            # print([os.path.join('data/WFLW_normal/train', x) for x in data_chunk])
#            data = [np.array(cv2.imread(os.path.join('D:/Scaled_landmark_datasets/WFLW_normal/train', x))) for x in data_chunk]
            data_chunk_1 =  train_data_files_1[i]
            data_chunk_2 =  train_data_files_2[i]
            label_chunk = train_label[i]
#            print(data_chunk, label_chunk)
            ########### From images
            data_1 = [np.array(preprocess_image(os.path.join('D:/face_dataset_depth_16bit/face_dataset_16/', x))) for x in data_chunk_1]
            data_2 = [np.array(preprocess_image(os.path.join('D:/face_dataset_depth_16bit/face_dataset_16/', x))) for x in data_chunk_2]
            
            
            ############# From feature files
#            data = [np.array(np.loadtxt(os.path.join('D:/Net_features/WFLW/train/ResNet_50_conv4_1_blk_', x.split('.')[0]+'.txt'))) for x in data_chunk]
            

            labels_source = np.array(label_chunk)
            yield [[np.array(data_1),np.array(data_2)], labels_source]
        except ValueError and IndexError:
            data_chunk = train_data_files_1.pop(0)
            data_chunk = train_data_files_2.pop(0)
            label_chunk = train_label.pop(0)
            train_data_files_1.append([data_chunk])
            train_data_files_2.append([data_chunk])
            train_label.append([label_chunk])
