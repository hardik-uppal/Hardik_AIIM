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
import dlib





#### change dir for depth and rgb

rgb_dir = 'D:/lock3d_protocol3/rgb/test_oc'
depth_dir = 'D:/lock3d_protocol3/depth/test_oc'#'D:/CurtinFaces_processed/DEPTH/'


out_depth_dir = 'D:/lock3d_protocol_crop/depth/test_oc'#'D:/CurtinFaces_crop/DEPTH/'
out_rgb_dir = 'D:/lock3d_protocol_crop/rgb/test_oc'#'D:/CurtinFaces_crop/DEPTH/'

if not os.path.exists(out_depth_dir):
    os.makedirs(out_depth_dir)

if not os.path.exists(out_rgb_dir):
    os.makedirs(out_rgb_dir)








subj_list = os.listdir(rgb_dir)
#### intitialization for Face crop
weights = 'mmod_human_face_detector.dat'
# initialize cnn based face detector with the weights
cnn_face_detector = dlib.cnn_face_detection_model_v1(weights)
noface_detetcted_depth =[]



def preprocess_image(image_path, image_depth, rgb_out_path, depth_out_path):
#    INPUT: image source path and image out path
#    RETURNS: X_min and Y_min coord, and 1,1 if no face id detected
    
    
#    start = time.time()
    
    image_read = cv2.imread(image_path)
    image_read = cv2.resize(image_read, (512, 424), interpolation=cv2.INTER_AREA)
    image = cv2.imread(image_depth)
    
#    print(image_read.shape)
#    
#    print(image.shape)
    # apply face detection (cnn)
    faces_cnn = cnn_face_detector(image_read, 1)
#    print(len(faces_cnn))
#    end = time.time()
#    print("CNN : ", format(end - start, '.2f'))
    if len(faces_cnn) != 0:
#        print('yes')
    # loop over detected faces
        for face in faces_cnn:
            x = face.rect.left()
            y = face.rect.top()
            w = face.rect.right() - x
            h = face.rect.bottom() - y
        
             # draw box over face
            cropped_image_rgb = image_read[int(y-(0.8*h)):int(y + (1.4*h)), int(x-(0.8*w)):int(x + (1.4*w))]
            cropped_image_depth = image[int(y-(0.8*h)):int(y + (1.4*h)), int(x-(0.8*w)):int(x + (1.4*w))]
    #        cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)  
            cv2.imwrite(rgb_out_path, cropped_image_rgb)
            cv2.imwrite(depth_out_path, cropped_image_depth)
        
            return int(x-(0.2*w)), int(y-(0.2*h))
    else:
        print('no face detetcted')
        noface_detetcted_depth.append(image_path)
        cv2.imwrite(rgb_out_path, image_read)
        cv2.imwrite(depth_out_path, image)
        return 1, 1


#######################Lock3dFaces
        
    
for subject in tqdm(subj_list):
    subj_dir_rgb = os.path.join(rgb_dir,subject)
    subj_dir_depth = os.path.join(depth_dir,subject)
    subj_dir_rgb_out = os.path.join(out_rgb_dir,subject)
    subj_dir_depth_out = os.path.join(out_depth_dir,subject)
    
    if not os.path.exists(subj_dir_rgb_out):
        os.makedirs(subj_dir_rgb_out)

    if not os.path.exists(subj_dir_depth_out):
        os.makedirs(subj_dir_depth_out)

    
    for file in os.listdir(subj_dir_rgb): 
        file_rgb = os.path.join(subj_dir_rgb,file)
        file_depth = os.path.join(subj_dir_depth,file)
        file_rgb_out = os.path.join(subj_dir_rgb_out,file)
        file_depth_out = os.path.join(subj_dir_depth_out,file)

        preprocess_image(file_rgb, file_depth, file_rgb_out, file_depth_out)








#######################curtinFaces
#for subject in tqdm(subj_list):
#    subject_dir = os.path.join(read_dir,subject)
#    subject_dir_depth = os.path.join(ROOT_DIR,subject)
##    destination_dir = os.path.join(OUT_DIR,subject)
##    if not os.path.exists(OUT_DIR):
##        os.makedirs(OUT_DIR)
#
#    for image in os.listdir(subject_dir_depth):
##            print(image)
##            break
#    
#        if image.split('.')[0] not in ['01','05','11','53','06','18','54','07','25','55','08','32','56','57','09','10','39','46']:### test set
#            source_read = os.path.join(subject_dir,image)
#            source = os.path.join(subject_dir_depth,image)
#            destination_dir = os.path.join(OUT_DIR,'test')
#            destination_dir = os.path.join(destination_dir,subject)
#            destination = os.path.join(destination_dir,image)
#            if not os.path.exists(destination_dir):
#                os.makedirs(destination_dir)
#
#            preprocess_image(source_read,source,destination)
#            
#
#        else:### train set
#            
#            source_read = os.path.join(subject_dir,image)
#            source = os.path.join(subject_dir_depth,image)
#            destination_dir = os.path.join(OUT_DIR,'train')
#            destination_dir = os.path.join(destination_dir,subject)
#            destination = os.path.join(destination_dir,image)
#            if not os.path.exists(destination_dir):
#                os.makedirs(destination_dir)
#
#            preprocess_image(source_read,source,destination)
#    

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
            
        

      
            
        
    
        

    
