# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 23:59:51 2019

@author: hardi

script for changing eurocom to keras data generator format

"""

import os
from shutil import copyfile
#import cv2
#import csv
import numpy as np
from tqdm import tqdm






#### change dir for depth and rgb

RGB_DIR = 'D:/CurtinFaces_crop/RGB/test/'

RGB_OUT_DIR = 'D:/CurtinFaces_crop/RGB/test'


DEPTH_DIR = 'D:/CurtinFaces_crop/normalized/DEPTH/test/'
DEPTH_OUT_DIR = 'D:/CurtinFaces_crop/normalized/DEPTH/test'

#read_dir = 'D:/CurtinFaces_processed/RGB/'

subj_list = os.listdir(RGB_DIR)
test1_list = []
test1_list.append([i for i in range(2,5)])
test1_list.append([i for i in range(12,18)])
test1_list.append([i for i in range(19,25)])
test1_list.append([i for i in range(26,32)])
test1_list.append([i for i in range(33,39)])
test1_list.append([i for i in range(40,46)])
test1_list.append([i for i in range(47,53)])
test1_list = [item for sublist in test1_list for item in sublist]



test2_list = []
test2_list.append([i for i in range(58,88)])
test2_list = [item for sublist in test2_list for item in sublist]


test3_list = []
test3_list.append([i for i in range(88,93)])
test3_list = [item for sublist in test3_list for item in sublist]

for subject in tqdm(subj_list):
    subject_dir_rgb = os.path.join(RGB_DIR,subject)
    subject_dir_depth = os.path.join(DEPTH_DIR,subject)
    


    for image in os.listdir(subject_dir_rgb):
#            print(image)
#            break
    
        if int(image.split('.')[0]) in test1_list:### test set
            source_rgb = os.path.join(subject_dir_rgb,image)
            
            dest_dir = RGB_OUT_DIR + '1/'
            destination_dir_rgb = os.path.join(dest_dir,subject)
            if not os.path.exists(destination_dir_rgb):
                os.makedirs(destination_dir_rgb)  
            
            destination_rgb =  os.path.join(destination_dir_rgb,image)  
            copyfile(source_rgb,destination_rgb)
            
            source_depth = os.path.join(subject_dir_depth,image)
            dest_dir = DEPTH_OUT_DIR + '1/'
            destination_dir_depth = os.path.join(dest_dir,subject)
            if not os.path.exists(destination_dir_depth):
                os.makedirs(destination_dir_depth)
            
            destination_depth =  os.path.join(destination_dir_depth,image)  
            copyfile(source_depth,destination_depth)
        
        elif int(image.split('.')[0]) in test2_list:### test set
            source_rgb = os.path.join(subject_dir_rgb,image)
            
            dest_dir = RGB_OUT_DIR + '2/'
            destination_dir_rgb = os.path.join(dest_dir,subject)
            if not os.path.exists(destination_dir_rgb):
                os.makedirs(destination_dir_rgb)  
            
            destination_rgb =  os.path.join(destination_dir_rgb,image)  
            copyfile(source_rgb,destination_rgb)
            
            source_depth = os.path.join(subject_dir_depth,image)
            dest_dir = DEPTH_OUT_DIR + '2/'
            destination_dir_depth = os.path.join(dest_dir,subject)
            if not os.path.exists(destination_dir_depth):
                os.makedirs(destination_dir_depth)
            
            destination_depth =  os.path.join(destination_dir_depth,image)  
            copyfile(source_depth,destination_depth)

        elif int(image.split('.')[0]) in test3_list:### test set
            source_rgb = os.path.join(subject_dir_rgb,image)
            
            dest_dir = RGB_OUT_DIR + '3/'
            destination_dir_rgb = os.path.join(dest_dir,subject)
            if not os.path.exists(destination_dir_rgb):
                os.makedirs(destination_dir_rgb)  
            
            destination_rgb =  os.path.join(destination_dir_rgb,image)  
            copyfile(source_rgb,destination_rgb)
            
            source_depth = os.path.join(subject_dir_depth,image)
            dest_dir = DEPTH_OUT_DIR + '3/'
            destination_dir_depth = os.path.join(dest_dir,subject)
            if not os.path.exists(destination_dir_depth):
                os.makedirs(destination_dir_depth)
            
            destination_depth =  os.path.join(destination_dir_depth,image)  
            copyfile(source_depth,destination_depth)
            


    

#def extract_depth(path):
#    with open(path) as f:
#        file = csv.reader(f, delimiter='\t')
#        pts = [(row[:-1]) for row in file]
#    pts = np.asarray(pts).astype(int)
#    return pts  
#
#for subject in label_list:
#    subject_path = os.path.join(ROOT_DIR,subject)
#    destination_dir = os.path.join(OUT_DIR,subject)
#    if not os.path.exists(destination_dir):
#        os.makedirs(destination_dir)
#    for i in range(1,3):
#        image_dir = subject_path + '/s{}/Depth/DepthBMP'.format(i)
#        for image in os.listdir(image_dir):
##            if image != 'Thumbs.db':
#            source = os.path.join(image_dir,image)
##            depth_pts = 
#            destination = os.path.join(destination_dir,image)
#
#            copyfile(source,destination)
            
        

      
            
        
    
        

    
