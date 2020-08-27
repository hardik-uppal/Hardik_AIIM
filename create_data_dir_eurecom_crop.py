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



ROOT_DIR = 'D:/EURECOM_Kinect_Face_Dataset/EURECOM_Kinect_Face_Dataset/'
OUT_DIR = 'D:/EURECOM_Kinect_Face_Dataset_crop/'

weights = 'mmod_human_face_detector.dat'
# initialize cnn based face detector with the weights
cnn_face_detector = dlib.cnn_face_detection_model_v1(weights)
noface_detetcted_depth =[]
def preprocess_image(image_path, image_out_path):
#    INPUT: image source path and image out path
#    RETURNS: X_min and Y_min coord, and 1,1 if no face id detected
    
    
#    start = time.time()
        
    image = cv2.imread(image_path)
#    print('here')
    # apply face detection (cnn)
    faces_cnn = cnn_face_detector(image, 1)
#    print(len(faces_cnn))
#    end = time.time()
#    print("CNN : ", format(end - start, '.2f'))
    if len(faces_cnn) is not 0:
#        print('yes')
    # loop over detected faces
        for face in faces_cnn:
            x = face.rect.left()
            y = face.rect.top()
            w = face.rect.right() - x
            h = face.rect.bottom() - y
        
             # draw box over face
            y_min = int(y-(0.2*h))
            y_max = int(y + (1.2*h))
            x_min = int(x-(0.2*w))
            x_max = int(x + (1.2*w))
            cropped_image = image[y_min:y_max, x_min:x_max]
    #        cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)  
            cv2.imwrite(image_out_path, cropped_image)
        
            return y_min,y_max, x_min,x_max

    else:
        print('no face detetcted')
        noface_detetcted_depth.append(image_out_path)
        cv2.imwrite(image_out_path, image)
        return 1,1,1,1
    
    
    
    
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
        
        image_dir_rgb = subject_path + '/s{}/RGB/'.format(session)#### change dir for depth and rgb
        
        image_dir_depth = subject_path + '/s{}/Depth/DepthBMP'.format(session)#### change dir for depth and rgb
        for image in os.listdir(image_dir_rgb):
#            print(image)
#            break
            if image != 'Thumbs.db':
                
                source = os.path.join(image_dir_rgb,image)
                destination_dir = os.path.join(OUT_DIR,'RGB')
                destination_dir = os.path.join(destination_dir,subject)
                
                
                
                destination = os.path.join(destination_dir,image)
                if not os.path.exists(destination_dir):
                    os.makedirs(destination_dir)

                y_min,y_max, x_min,x_max = preprocess_image(source,destination)


    
        for image in os.listdir(image_dir_depth):
#            print(image)
#            break
            if image != 'Thumbs.db':
                
                source = os.path.join(image_dir_depth,image)
                destination_dir = os.path.join(OUT_DIR,'depth')
                destination_dir = os.path.join(destination_dir,subject)
                destination = os.path.join(destination_dir,image)
                if not os.path.exists(destination_dir):
                    os.makedirs(destination_dir)
                
                # crop depthg image
                image_depth = cv2.imread(source)    
                cropped_image = image_depth[y_min:y_max, x_min:x_max]
 
                cv2.imwrite(destination, cropped_image)
                if y_min+y_max+x_min+x_max == 4:
                    copyfile(source,destination)
                    

    
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
            
        

      
            
        
    
        

    
