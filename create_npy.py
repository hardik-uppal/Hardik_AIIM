# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:09:51 2019

@author: hardi
"""

import numpy as np
import os





file_path = 'D:/Pandora/validation_0_1_2_3_4.txt'
def create_npy(file_path):
    
    training_data_scaled = []
    reject = []
    with open(file_path, "r") as f:
        for row in f:
#            print(row,int(row.split()[0].split('/')[0]))
            if (int(row.split()[0].split('/')[0]) <= 100 and int(row.split()[1].split('/')[0]) <= 100):
#                print('accepted: {}'.format(list(row.split())))
#                print('folder: ' , int(row.split()[0].split('/')[0]),int(row.split()[1].split('/')[0]))
                training_data_scaled.append(list(row.split()))
            else:
#                print('rejected: {}'.format(list(row.split())))
#                print('folder: ' ,int(row.split()[0].split('/')[0]),int(row.split()[1].split('/')[0]))
                reject.append(list(row.split()))
 
    np.save('data/{}'.format(file_path.split('/')[-1].split('.')[0]) ,np.array(training_data_scaled))


for file in os.listdir('D:/Pandora/'):
    file_path = os.path.join('D:/Pandora/',file)
    create_npy(file_path)


data = np.load('data/validation_0_1_2_3_4.npy')
for row in data:
    if int(row[1].split('/')[0])>100:
        print(row)