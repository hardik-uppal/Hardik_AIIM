# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import scipy.io
import cv2
import numpy as np
import os
from tqdm import tqdm
#
#filepath = "D:/CurtinFaces/01/01.mat" 
#mat = scipy.io.loadmat(filepath)
#image_comb = mat['d']
##image_rgb = image_comb[:,3:6]
##
##image_rgb = (np.rot90(np.reshape(image_rgb,(640,480,3)),3)).astype(np.uint16)
##image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
##cv2.imwrite('D:/CurtinFaces_processed/example.jpg',image_rgb)
##
#image_depth = image_comb[:,2]
#image_depth = (np.rot90(np.reshape(image_depth,(640,480,1)),3)).astype(np.uint16)
#depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(image_depth, alpha=0.09), cv2.COLORMAP_JET)
##image_depth = cv2.cvtColor(image_depth, cv2.COLOR_BGR2GRAY)
#cv2.imwrite('D:/CurtinFaces_processed/example_d.jpg',depth_colormap)

ROOT_DIR = 'D:/CurtinFaces/'
#OUT_rgb_DIR = 'D:/CurtinFaces_processed/RGB'
OUT_depth_DIR = 'D:/CurtinFaces_processed/DEPTH'
#if not os.path.exists(OUT_rgb_DIR):
#    os.makedirs(OUT_rgb_DIR)
if not os.path.exists(OUT_depth_DIR):
    os.makedirs(OUT_depth_DIR)

label_list = os.listdir(ROOT_DIR)


for label in tqdm(label_list):
    mat_filepath = os.path.join(ROOT_DIR,label)
#    out_rgb_filepath = os.path.join(OUT_rgb_DIR,label)
    out_depth_filepath = os.path.join(OUT_depth_DIR,label)
#    if not os.path.exists(out_rgb_filepath):
#        os.makedirs(out_rgb_filepath)
    if not os.path.exists(out_depth_filepath):
        os.makedirs(out_depth_filepath)
    for file in os.listdir(mat_filepath):
        mat_filename = os.path.join(mat_filepath,file)
#        out_rgb_filename = os.path.join(out_rgb_filepath,file.split('.')[0])
        out_depth_filename = os.path.join(out_depth_filepath,file.split('.')[0])
        
        mat = scipy.io.loadmat(mat_filename)
        image_comb = mat['d']
#        image_rgb = image_comb[:,3:6]
#        image_rgb = np.rot90(np.reshape(image_rgb,(640,480,3)),3).astype(np.uint16)
#        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
#        cv2.imwrite(out_rgb_filename+'.jpg',image_rgb)
        
        image_depth = image_comb[:,2]
        image_depth = np.rot90(np.reshape(image_depth,(640,480,1)),3).astype(np.uint16)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(image_depth, alpha=0.09), cv2.COLORMAP_JET)
#        image_depth = cv2.cvtColor(image_depth, cv2.COLOR_BGR2RGB)
        cv2.imwrite(out_depth_filename+'.jpg',depth_colormap)


OUT_depth_DIR = 'D:/CurtinFaces_crop/DEPTH/'
Process_out_dir = 'D:/CurtinFaces_crop/normalized/DEPTH/'
if not os.path.exists(Process_out_dir):
    os.makedirs(Process_out_dir)

for test_train in tqdm(os.listdir(OUT_depth_DIR)):
    dir_test_train = os.path.join(OUT_depth_DIR, test_train)
    dir_test_train_out = os.path.join(Process_out_dir, test_train)
    
    
    for subject in os.listdir(dir_test_train):
        subject_dir = os.path.join(dir_test_train,subject)
        subject_dir_out = os.path.join(dir_test_train_out,subject)
        if not os.path.exists(subject_dir_out):
            os.makedirs(subject_dir_out)
        
        for image_file in  os.listdir(subject_dir):
            image = os.path.join(subject_dir, image_file)
            image_out = os.path.join(subject_dir_out, image_file)
            lower_range = 100
            upper_range = 160
            image = cv2.imread(image)
            image_depth = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_depth[image_depth > upper_range] = upper_range
            image_depth[image_depth < lower_range] = lower_range
            norm_image = cv2.normalize(image_depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            cv2.imwrite(image_out,norm_image)
            


#
#hrr_file = "D:/HRRFD/01/test/01_009.png"
#image_read = cv2.imread(hrr_file)
