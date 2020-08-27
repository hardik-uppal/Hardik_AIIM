# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 20:23:09 2019

@author: hardi
"""

import os
from tqdm import tqdm
from shutil import copyfile
import dlib
import cv2
import numpy as np
#eurecom_dir = 'D:/EURECOM_Kinect_Face_Dataset_crop/depth/'
iiitd_dir = 'D:/IIITD-RGBD/'
vap_dir = 'D:/vap-RGB-D/vap-RGBD/'

OUT_DIR = 'D:/RGB_D_Dataset_3small/'
############ IIITD RGBD




folder_list = os.listdir(iiitd_dir)

for fold in tqdm(folder_list):
    fold_dir = os.path.join(iiitd_dir, fold)
#    out_dir_fold = os.path.join(OUT_DIR, fold)
    for test_train in os.listdir(fold_dir):
        test_train_dir = os.path.join(fold_dir, test_train)
        
        for subject in os.listdir(test_train_dir):
            
            subject_new = int(subject) + 52
            
#            if test_train == 'testing':
#                dest_test_train = os.path.join(out_dir_fold,'test')
#            elif test_train == 'training':
#                dest_test_train = os.path.join(out_dir_fold,'train')
            
            
            subject_dir = os.path.join(test_train_dir,subject)
            
#            destination_dir = os.path.join(OUT_DIR,str(subject_new))
#            if not os.path.exists(destination_dir):
#                os.makedirs(destination_dir)
            
            for rgb_d in os.listdir(subject_dir):
                final_dir = os.path.join(subject_dir,rgb_d)
            
                if rgb_d == 'Depth':
                    dest_rgb_d = os.path.join(OUT_DIR,'depth')
                elif rgb_d == 'RGB':
                    dest_rgb_d = os.path.join(OUT_DIR,'RGB') 
                
                
                destination_dir = os.path.join(dest_rgb_d,str(subject_new))
                if not os.path.exists(destination_dir):
                    os.makedirs(destination_dir)                
                
                for image in os.listdir(final_dir):
                    
                
                
                    source = os.path.join(final_dir,image)
                    destination = os.path.join(destination_dir,image)
                    copyfile(source, destination)
                    
                    
                    
#################### VAP RGBD
                        

weights = 'mmod_human_face_detector.dat'
# initialize cnn based face detector with the weights
cnn_face_detector = dlib.cnn_face_detection_model_v1(weights)
noface_detetcted_depth =[]
def preprocess_image(image_path, image_out_path):
#    INPUT: image source path and image out path
#    RETURNS: X_min and Y_min coord, and 1,1 if no face id detected
    
    
#    start = time.time()
        
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640,480), interpolation = cv2.INTER_AREA)
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

subject_new=158
subject_list = os.listdir(vap_dir)    

for subject_dir in tqdm(subject_list):
    bb_dir = {}
    image_dir = os.path.join(vap_dir,subject_dir)
    subject_new = subject_new + 1
    print(subject_new)
    for images in os.listdir(image_dir):
        
        if (images.split('.')[-1] == 'bmp'):
            image_file_rgb = os.path.join(image_dir,images)
            dest_dir = os.path.join(OUT_DIR,'RGB')
            dest_subj_dir = os.path.join(dest_dir,str(subject_new))
            dest_file = os.path.join(dest_subj_dir,images.split('.')[0]+'.jpg')
            if not os.path.exists(dest_subj_dir):
                os.makedirs(dest_subj_dir) 
        
            y_min,y_max, x_min,x_max = preprocess_image(image_file_rgb,dest_file)
            bb_dir[images.split('_')[0]+'_'+images.split('_')[1]] =  y_min,y_max, x_min,x_max
        
        
        
        
        elif (images.split('.')[-1] == 'dat'):
            image_file_depth = os.path.join(image_dir,images)
            dest_dir = os.path.join(OUT_DIR,'depth')
            dest_subj_dir = os.path.join(dest_dir,str(subject_new))
            dest_file = os.path.join(dest_subj_dir,images.split('.')[0]+'.jpg')
            if not os.path.exists(dest_subj_dir):
                os.makedirs(dest_subj_dir) 
            
            image_depth = np.loadtxt(image_file_depth)
            y_min,y_max, x_min,x_max = bb_dir[images.split('_')[0]+'_'+images.split('_')[1]]
            
            
            lower_per = 12 
            upper_per = 27.1
            
#            image_depth = np.loadtxt(vap_dir)
            
            lower_limit = np.percentile(image_depth, lower_per)
            upper_limit = np.percentile(image_depth, upper_per)
            
            
            
            image_depth[image_depth > upper_limit] = upper_limit
            image_depth[image_depth < lower_limit] = lower_limit
            image_norm = cv2.normalize(image_depth,None, 0, 255, cv2.NORM_MINMAX)
            image_norm = image_norm[y_min:y_max,x_min:x_max]
            cv2.imwrite(dest_file,image_norm)
                
        
            
        

    