# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 23:59:51 2019

@author: hardi

script for changing eurocom to keras data generator format

"""

import os
from shutil import copyfile
import cv2
import csv
import numpy as np
from tqdm import tqdm



ROOT_DIR = 'D:/EURECOM_Kinect_Face_Dataset/EURECOM_Kinect_Face_Dataset/'
OUT_DIR = 'D:/EURECOM_Kinect_Face_Dataset/depth/'

label_list = os.listdir(ROOT_DIR)
#
#
#for subject in label_list:
#    subject_path = os.path.join(ROOT_DIR,subject)
#    destination_dir = os.path.join(OUT_DIR,subject)
#    if not os.path.exists(destination_dir):
#        os.makedirs(destination_dir)
#    for i in range(1,3):
#        image_dir = subject_path + '/s{}/RGB'.format(i)
#        for image in os.listdir(image_dir):
#            if image != 'Thumbs.db':
#                source = os.path.join(image_dir,image)
#                source_image= cv2.imread(source)
#                destination = os.path.join(destination_dir,image)
##                print(source, destination)
#                cv2.imwrite(destination,source_image)
##                copyfile(source,destination)
#        
#


for subject in tqdm(label_list):
    subject_path = os.path.join(ROOT_DIR,subject)
#    destination_dir = os.path.join(OUT_DIR,subject)
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    for session in range(1,3):
        image_dir = subject_path + '/s{}/Depth/DepthBMP'.format(session)#### change dir for depth and rgb
        for image in os.listdir(image_dir):
#            print(image)
#            break
            if image != 'Thumbs.db':
                if image in ['depth_{}_s{}_OcclusionEyes.bmp'.format(subject,session),'depth_{}_s{}_OpenMouth.bmp'.format(subject,session)]:### val set
                    source = os.path.join(image_dir,image)
                    destination_dir = os.path.join(OUT_DIR,'val')
                    destination_dir = os.path.join(destination_dir,subject)
                    destination = os.path.join(destination_dir,image)
                    if not os.path.exists(destination_dir):
                        os.makedirs(destination_dir)
    
                    copyfile(source,destination)
                    
#                elif image in ['depth_{}_s{}_LightOn.bmp'.format(subject,session),'depth_{}_s{}_Smile.bmp'.format(subject,session)]:### test set
#                    
#                    source = os.path.join(image_dir,image)
#                    destination_dir = os.path.join(OUT_DIR,'test')
#                    destination_dir = os.path.join(destination_dir,subject)
#                    destination = os.path.join(destination_dir,image)
#                    if not os.path.exists(destination_dir):
#                        os.makedirs(destination_dir)
#    #                source_image= cv2.imread(source)
#    #                destination = os.path.join(destination_dir,image)
#    #                print(source, destination)
#                    copyfile(source,destination)
                else:### train set
                    
                    source = os.path.join(image_dir,image)
                    destination_dir = os.path.join(OUT_DIR,'train')
                    destination_dir = os.path.join(destination_dir,subject)
                    destination = os.path.join(destination_dir,image)
                    if not os.path.exists(destination_dir):
                        os.makedirs(destination_dir)
    #                source_image= cv2.imread(source)
    #                destination = os.path.join(destination_dir,image)
    #                print(source, destination)
                    copyfile(source,destination)
    

def extract_depth(path):
    with open(path) as f:
        file = csv.reader(f, delimiter='\t')
        pts = [(row[:-1]) for row in file]
    pts = np.asarray(pts).astype(int)
    return pts  

for subject in label_list:
    subject_path = os.path.join(ROOT_DIR,subject)
    destination_dir = os.path.join(OUT_DIR,subject)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    for i in range(1,3):
        image_dir = subject_path + '/s{}/Depth/DepthBMP'.format(i)
        for image in os.listdir(image_dir):
#            if image != 'Thumbs.db':
            source = os.path.join(image_dir,image)
#            depth_pts = 
            destination = os.path.join(destination_dir,image)

            copyfile(source,destination)
            
        

      
            
        
    
        

    
