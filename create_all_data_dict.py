# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 13:25:03 2019

@author: hardi
"""


import os
from shutil import copyfile
import cv2
import csv
import numpy as np
from tqdm import tqdm
import math



#ROOT_DIR = 'D:/EURECOM_Kinect_Face_Dataset/EURECOM_Kinect_Face_Dataset/'
#ROOT_DIR = 'D:/face_dataset_RGB/face_dataset_RGB/'
ROOT_DIR = 'D:/face_dataset_depth_16bit/face_dataset_16/'
OUT_DIR = 'D:/RGB_D_Dataset/train/depth/'
######eurecom
label_list = os.listdir(ROOT_DIR)


for subject in tqdm(label_list):
    subject_path = os.path.join(ROOT_DIR,subject)
    destination_dir = os.path.join(OUT_DIR,subject)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    for session in range(1,3):
        image_dir = subject_path + '/s{}/RGB/'.format(session)#### change dir for depth and rgb
        for image in os.listdir(image_dir):
#            print(image)
#            break
            if image != 'Thumbs.db':
                source = os.path.join(image_dir,image)
#                destination_dir = os.path.join(OUT_DIR,'train')
#                destination_dir = os.path.join(OUT_DIR,subject)
                destination = os.path.join(destination_dir,image)

                copyfile(source, destination)
            
  ########################
## PANDORA          
#
#image_dir = "D:/face_dataset_depth_16bit/face_dataset_16/001/frame_000000_face_depth.png"
#image = cv2.imread(image_dir,0)
#depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(image, alpha=20), cv2.COLORMAP_JET)
#
  
#cv2.imwrite('D:/face_dataset_depth_16bit/example_d.jpg',image_depth)


folder_list = os.listdir(ROOT_DIR)

for folder in tqdm(folder_list):
    image_dir = os.path.join(ROOT_DIR, folder)
    subject = math.ceil(int(folder)/5) + 52
    run = int(folder) - (subject) * 5
    destination_dir = os.path.join(OUT_DIR,str(subject))
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for img_file in os.listdir(image_dir):
        if img_file not in ['angles.txt']:
            image_path = os.path.join(image_dir, img_file)
            destination = os.path.join(destination_dir, folder+'_'+img_file)
            image = cv2.imread(image_path,0)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(image, alpha=10), cv2.COLORMAP_JET)
            image_depth = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(destination,image_depth)

#            copyfile(image_path, destination)
############################# IIIT RGBD
ROOT_DIR = 'D:/IIITD-RGBD/'
OUT_DIR = 'D:/RGB_D_Dataset_new/'


folder_list = os.listdir(ROOT_DIR)

for fold in tqdm(folder_list):
    fold_dir = os.path.join(ROOT_DIR, fold)
    out_dir_fold = os.path.join(OUT_DIR, fold)
    for test_train in os.listdir(fold_dir):
        test_train_dir = os.path.join(fold_dir, test_train)
        
        for subject in os.listdir(test_train_dir):
            
            subject_new = int(subject) + 72
            
            if test_train == 'testing':
                dest_test_train = os.path.join(out_dir_fold,'test')
            elif test_train == 'training':
                dest_test_train = os.path.join(out_dir_fold,'train')
            
            subject_dir = os.path.join(test_train_dir,subject)
            
#            destination_dir = os.path.join(OUT_DIR,str(subject_new))
#            if not os.path.exists(destination_dir):
#                os.makedirs(destination_dir)
            
            for rgb_d in os.listdir(subject_dir):
                final_dir = os.path.join(subject_dir,rgb_d)
            
                if rgb_d == 'Depth':
                    dest_rgb_d = os.path.join(dest_test_train,'depth')
                elif rgb_d == 'RGB':
                    dest_rgb_d = os.path.join(dest_test_train,'RGB') 
                
                
                destination_dir = os.path.join(dest_rgb_d,str(subject_new))
                if not os.path.exists(destination_dir):
                    os.makedirs(destination_dir)                
                
                for image in os.listdir(final_dir):
                    
                
                
                    source = os.path.join(final_dir,image)
                    destination = os.path.join(destination_dir,image)
                    copyfile(source, destination)
            
#        destination_dir = os.path.join(OUT_DIR,str(subject_new))
#        if not os.path.exists(destination_dir):
#            os.makedirs(destination_dir)
    
        for img_file in os.listdir(image_dir):
            if img_file not in ['angles.txt']:
                image_path = os.path.join(image_dir, img_file)
                destination = os.path.join(destination_dir, folder+'_'+img_file)
                image = cv2.imread(image_path,0)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(image, alpha=10), cv2.COLORMAP_JET)
                image_depth = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(destination,image_depth)            
            
            
            