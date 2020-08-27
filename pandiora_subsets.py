# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 07:53:34 2019

@author: hardi
"""
import cv2
import os
import math
from random import sample,randint

#turn depth images to colour maps
image_dir = "D:/face_dataset_depth_16bit/face_dataset_16/001/frame_000000_face_depth.png"
image = cv2.imread(image_dir,0)
depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(image, alpha=10), cv2.COLORMAP_JET)
image_depth = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
cv2.imwrite('D:/face_dataset_depth_16bit/example_d.jpg',image_depth)


# create npy
#three subsets S1; S2; S3
S1, S2, S3 = {}
master_list = []

ROOT_DIR ='D:/face_dataset_RGB/face_dataset_RGB/'


folder_list = os.listdir(ROOT_DIR)

for folder in (folder_list):
    image_dir = os.path.join(ROOT_DIR, folder)
    subject = math.ceil(int(folder)/5)
    run = int(folder) - (subject) * 5
#    print(image_dir)
    for img_file in os.listdir(image_dir):
        if img_file not in ['angles.txt']:
            image_path = os.path.join(image_dir, img_file)
    #        print(img_file)
            master_list.append([image_path,subject, run])
            
for x in master_list:            
contained = [x for x in master_list if x not in master_list_depth]            

master_list_depth = []

ROOT_DIR_depth ='D:/face_dataset_depth_16bit/face_dataset_16/'


folder_list = os.listdir(ROOT_DIR_depth)

for folder in (folder_list):
    image_dir = os.path.join(ROOT_DIR_depth, folder)
    subject = math.ceil(int(folder)/5)
    run = int(folder) - (subject) * 5
#    print(image_dir)
    for img_file in os.listdir(image_dir):
        if img_file not in ['angles.txt']:
            image_path = os.path.join(image_dir, img_file)
    #        print(img_file)
            master_list_depth.append([image_path,subject, run])
        
subject_len = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0,20:0}
for image in master_list:
    subject = image[1]
#    print(subject_len.keys)
    subject_len[subject] =  subject_len[subject] + 1
    
    
subset_1 = []
subset_2 = []
subset_super = []

for i in range(0,1000000):
    image_1 = sample(master_list,1)[0]
    image_2 = sample(master_list,1)[0]
#    print(imag1e_1)
    if image_1[2]<=2 and image_2[2]<=2 :
        if image_1[1]==image_2[1]:
            subset_1.append([image_1,image_2,1])
        else:
            subset_1.append([image_1,image_2,0])
    elif image_1[2]>2 and image_2[2]>2 :
        if image_1[1]==image_2[1]:
            subset_2.append([image_1,image_2,1])
        else:
            subset_2.append([image_1,image_2,0])
    else:
        if image_1[1]==image_2[1]:
            subset_super.append([image_1,image_2,1])
        else:
            subset_super.append([image_1,image_2,0])
    
positive_pairs= []
negative_pairs= []


    
    
    
    

