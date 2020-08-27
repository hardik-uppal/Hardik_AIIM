# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 00:20:57 2019

@author: hardi
"""

import os
import csv
import numpy as np

dir_name= './testCurtinFaces'

folder_list = os.listdir(dir_name)
test_acc_final = 0
final_val_acc = 0
dict_tes_acc = {}
dict_val_acc = {}
for folder in folder_list:
#    print(folder) ## variation type
    file_path = os.path.join(dir_name,str(folder))
    for file in os.listdir(file_path):
        if file == 'log_test.csv':
            logfile_test = os.path.join(file_path,file)
            with open(logfile_test) as f:
                content = f.read()
                
                test_acc = float(content.split()[-1])
            if test_acc > test_acc_final:
                test_acc_final = test_acc
                dict_tes_acc[folder] = test_acc_final
        if file == 'log.csv':
            content_log=[]
            logfile_test = os.path.join(file_path,file)
            with open(logfile_test) as f:
                for row in f:
                    content_log.append(row.split(','))
            if len(content_log) != 0:
                
                content_log = np.asarray(content_log[1:],dtype='float16')
                max_val_acc = np.max(content_log[:,3])
                if  max_val_acc > final_val_acc:
                    final_val_acc = max_val_acc
                    dict_val_acc[folder] = final_val_acc
                    
                    
print('dict_val_acc:' , dict_val_acc)
print('dict_tes_acc:' , dict_tes_acc)               
            
                
            
        
        
        
