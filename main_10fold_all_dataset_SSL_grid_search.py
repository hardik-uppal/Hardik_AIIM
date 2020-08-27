# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 23:57:24 2019

@author: Pritam
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import tensorflow as tf

## get filename during run time and set pwd manually
dirname, filename = os.path.split(os.path.abspath(__file__))
print("running: {}".format(filename) )
## change directory to the current directory where code is saved
os.chdir(dirname)

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from pathlib import Path

tf.logging.set_verbosity(tf.logging.ERROR)

#from SSL_Func import *
#from signal_transformation_task import *
from SSL_model import SSL_model_shallow, get_label, calculate_loss, get_weighted_loss, get_prediction, make_total_batch, unison_shuffled_copies
from SSL_model import fetch_all_loss, fetch_pred_labels, fetch_true_labels, get_results_ssl, write_result, write_summary
from supervised_of_selfsupervised import extract_feature, general_supervised_task_swell, general_supervised_task_wesad, general_supervised_task_dreamer, general_supervised_task_amigos

from data_preprocessing import extract_swell_dataset, extract_dreamer_dataset, extract_amigos_dataset, extract_wesad_dataset, load_data 
from data_preprocessing import swell_prepare_for_10fold, wesad_prepare_for_10fold, amigos_prepare_for_10fold, dreamer_prepare_for_10fold, current_time, makedirs, one_hot_encoding
from send_emails import send_email


#data_folder = Path("D:\\Code Repo\\Working dir\\Self supervised task\\NEW_WORK_DIR\\data_folder\\full_data_filtered\\")
extract_data = 0

""" for the first time run this """ 
#if extract_data == 1:
#    _       = extract_swell_dataset(overlap_pct= 1, window_size_sec= 10, data_save_path= data_folder, save= 1)
#    _       = extract_dreamer_dataset(overlap_pct= 1, window_size_sec= 10, data_save_path= data_folder, save= 1)
#    _       = extract_amigos_dataset(overlap_pct= 1, window_size_sec= 10, data_save_path= data_folder, save=1)
#    _       = extract_wesad_dataset(overlap_pct=1, window_size_sec=10, data_save_path= data_folder, save=1)

dirname = Path("/home/pritam/self_supervised_learning/")
swell_data              = load_data(dirname / "data_folder/full_data_filtered/swell_dict.npy")    
dreamer_data            = load_data(dirname / "data_folder/full_data_filtered/dreamer_dict.npy")    
amigos_data             = load_data(dirname / "data_folder/full_data_filtered/amigos_dict.npy")    
wesad_data              = load_data(dirname / "data_folder/full_data_filtered/wesad_dict.npy")    


swell_data              = swell_prepare_for_10fold(swell_data)  #person, y_input_stress, y_arousal, y_valence, 
wesad_data              = wesad_prepare_for_10fold(wesad_data) # person, y_stress
amigos_data             = amigos_prepare_for_10fold(amigos_data) # person, y_arousal, y_valence, y_dominance
dreamer_data            = dreamer_prepare_for_10fold(dreamer_data) # person, y_arousal, y_valence, y_dominance

#swell_data = swell_data[1:300]
#wesad_data = wesad_data[1:300]
#amigos_data = amigos_data[1:300]
#dreamer_data = dreamer_data[1:300]

total_fold = 10
kf = KFold(n_splits=total_fold, shuffle=True, random_state=True)

swell_train_index = []
swell_test_index = []
for train_index, test_index in kf.split(swell_data):
    temp = train_index
    swell_train_index.append(train_index)
    swell_test_index.append(test_index)
    
wesad_train_index = []
wesad_test_index = []
for train_index, test_index in kf.split(wesad_data):
    wesad_train_index.append(train_index)
    wesad_test_index.append(test_index)
    
amigos_train_index = []
amigos_test_index = []
for train_index, test_index in kf.split(amigos_data):
    amigos_train_index.append(train_index)
    amigos_test_index.append(test_index)
    
dreamer_train_index = []
dreamer_test_index = []
for train_index, test_index in kf.split(dreamer_data):
    dreamer_train_index.append(train_index)
    dreamer_test_index.append(test_index)


""" self supervised task start """


noise_amount = [25] # [15, 25]
scaling_factor = [0.8, 0.9, 1.1, 1.4]
permutation_pieces = [9, 8, 20]
time_warping_pieces = [9, 7]	
time_warping_stretch_factor = [1.05, 1.3]
time_warping_squeeze_factor = [1/e for e in time_warping_stretch_factor]

####################################

#selfsupevised_path = Path("D:\\Code Repo\\working dir\\Self supervised task\\NEW_WORK_DIR\\selfsupervised\\10fold\\")
selfsupevised_path = Path("/home/pritam/self_supervised_learning/10fold/")


no_of_task = ['original_signal', 'noised_signal', 'scaled_signal', 'negated_signal', 'flipped_signal', 'permuted_signal', 'time_warped_signal']
transform_task = [0, 1, 2, 3, 4, 5, 6]
single_batch_size = len(transform_task)
supervised_interval = 5

""" hyper parameter tuning conv layers"""

batchsize = 128  
actual_batch_size =  batchsize * single_batch_size
log_step = 100
epoch = 30
initial_learning_rate = 0.001
drop_rate = 0.6
regularizer = 1
L2 = 0.0001
lr_decay_steps = 10000
lr_decay_rate = 0.9
loss_coeff = [0.195, 0.195, 0.195, 0.0125, 0.0125, 0.195, 0.195]
window_size = 2560


graph = tf.Graph()
print('creating graph...')
with graph.as_default():
    
    input_tensor        = tf.compat.v1.placeholder(tf.float32, shape = (None, window_size, 1), name = "input")
    y                   = tf.compat.v1.placeholder(tf.float32, shape = (None, np.shape(transform_task)[0]), name = "output") 
    drop_out            = tf.compat.v1.placeholder_with_default(1.0, shape=(), name="Drop_out")
    isTrain             = tf.placeholder(tf.bool, name = 'isTrain')
    global_step         = tf.Variable(0, dtype=np.float32, trainable=False, name="steps")

    conv1, conv2, conv3, main_branch, task_0, task_1, task_2, task_3, task_4, task_5, task_6 = SSL_model_shallow(input_tensor, isTraining= isTrain, drop_rate= drop_out) #convpool doesn't work
    logits = [task_0, task_1, task_2, task_3, task_4, task_5, task_6]
    featureset_size = main_branch.get_shape()[1].value
    y_label = get_label(y= y, actual_batch_size= actual_batch_size)
    all_loss = calculate_loss(y_label, logits)
    output_loss = get_weighted_loss(loss_coeff, all_loss)  
    
    if regularizer:
        l2_loss = 0
        weights = []
        for v in tf.trainable_variables():
            weights.append(v)
            if 'kernel' in v.name:
                l2_loss += tf.nn.l2_loss(v)
        output_loss = output_loss + l2_loss * L2
        
    y_pred                = get_prediction(logits = logits)
    learning_rate         = tf.compat.v1.train.exponential_decay(initial_learning_rate, global_step, decay_steps=lr_decay_steps, decay_rate=lr_decay_rate, staircase=True)

    optimizer             = tf.compat.v1.train.AdamOptimizer(learning_rate) 
    
    with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
        train_op    = optimizer.minimize(output_loss, global_step, colocate_gradients_with_ops=True)
        
    with tf.variable_scope('Session_saver'):
        saver       = tf.compat.v1.train.Saver(max_to_keep=10)

    tf.compat.v1.summary.scalar('model/learning_rate', learning_rate)
    tf.compat.v1.summary.scalar('model/total_batch_output_loss', output_loss)
    
    summary_op      = tf.compat.v1.summary.merge_all()    
        
print('graph creation finished')


for noise_param in noise_amount:
    for scale_param in scaling_factor:
        for permu_param in permutation_pieces:
            for tw_piece_param in time_warping_pieces:
                for twsf_param in time_warping_stretch_factor:
                
                #for k in range(total_fold):
                    flag = str(noise_param) + ',' + str(scale_param) + ',' + str(permu_param) + ',' + str(tw_piece_param) + ',' + str(twsf_param)
                    k=4
                    ssl_logs = selfsupevised_path / "SSL_logs" / str(current_time() + "_ssl_logs")
                    super_logs = selfsupevised_path / "SSL_logs" / str(current_time() + "_super_logs")
                
                    tr_ssl_result_filename  = selfsupevised_path / "result"   / str("tr_signal_tf_" + str(k) +"_"  + current_time() + ".npy")
                    te_ssl_result_filename  = selfsupevised_path / "result"   / str("te_signal_tf_" + str(k) +"_"  + current_time() + ".npy")
                    tr_ssl_loss_filename    = selfsupevised_path / "loss"     / str("tr_signal_tf_" + str(k) +"_"  + current_time() + ".npy")
                    te_ssl_loss_filename    = selfsupevised_path / "loss"     / str("te_signal_tf_" + str(k) +"_"  + current_time() + ".npy")
                            
                
                    makedirs(ssl_logs)
                    
                    train_ECG   = np.vstack((swell_data[swell_train_index[k], 4:], amigos_data[amigos_train_index[k], 3:], dreamer_data[dreamer_train_index[k], 3:], wesad_data[wesad_train_index[k], 2:])) 
                    test_ECG    = np.vstack((swell_data[swell_test_index[k], 4:], amigos_data[amigos_test_index[k], 3:], dreamer_data[dreamer_test_index[k], 3:], wesad_data[wesad_test_index[k], 2:])) 
                    train_ECG   = shuffle(train_ECG)
                
                    train_swell_input_stress, test_swell_input_stress = one_hot_encoding(arr = swell_data[:, 1], tr_index = swell_train_index[k], te_index = swell_test_index[k])
                    train_swell_arousal, test_swell_arousal           = one_hot_encoding(arr = swell_data[:, 2], tr_index = swell_train_index[k], te_index = swell_test_index[k])
                    train_swell_valence, test_swell_valence           = one_hot_encoding(arr = swell_data[:, 3], tr_index = swell_train_index[k], te_index = swell_test_index[k])
                    
                    train_dreamer_arousal, test_dreamer_arousal       = one_hot_encoding(arr = dreamer_data[:, 1], tr_index = dreamer_train_index[k], te_index = dreamer_test_index[k])
                    train_dreamer_valence, test_dreamer_valence       = one_hot_encoding(arr = dreamer_data[:, 2], tr_index = dreamer_train_index[k], te_index = dreamer_test_index[k])
                    
                    train_amigos_arousal, test_amigos_arousal         = one_hot_encoding(arr = amigos_data[:, 1],  tr_index = amigos_train_index[k], te_index = amigos_test_index[k])
                    train_amigos_valence, test_amigos_valence         = one_hot_encoding(arr = amigos_data[:, 2],  tr_index = amigos_train_index[k], te_index = amigos_test_index[k])
                    
                    train_wesad_stress, test_wesad_stress             = one_hot_encoding(arr = wesad_data[:, 1],  tr_index = wesad_train_index[k], te_index = wesad_test_index[k])
                    
                    
                # 
                    training_length = train_ECG.shape[0]
                    testing_length  = test_ECG.shape[0]
                    
                    print('Initializing all parameters.')
                    tf.reset_default_graph()
                    with tf.Session(graph=graph) as sess:   
                        summary_writer = tf.compat.v1.summary.FileWriter(ssl_logs, sess.graph)
                    
                        sess.run(tf.global_variables_initializer())
                        sess.run(tf.local_variables_initializer())
                        
                        print('self supervised training started')
                        
                        train_loss_dict = {}
                        test_loss_dict = {}
                    
                        tr_ssl_result = {}
                        te_ssl_result = {}    
                        
                        ## epoch loop
                        for epoch_counter in tqdm(range(epoch)):
                            
                            tr_loss_task = np.zeros((len(transform_task), 1), dtype  = np.float32)
                            train_pred_task = np.zeros((len(transform_task), actual_batch_size), dtype  = np.float32) -1
                            train_true_task = np.zeros((len(transform_task), actual_batch_size), dtype  = np.float32) -1
                            tr_output_loss = 0
                    
                           
                            tr_total_gen_op = make_total_batch(data = train_ECG, length = training_length, batchsize = batchsize, 
                                                               
                                                               noise_amount=noise_param, 
                                                               scaling_factor=scale_param, 
                                                               permutation_pieces=permu_param, 
                                                               time_warping_pieces=tw_piece_param, 
                                                               time_warping_stretch_factor= twsf_param, 
                                                               time_warping_squeeze_factor= 1/twsf_param)
                    
                            for training_batch, training_labels, tr_counter, tr_steps in tr_total_gen_op:
                                
                                ## run the model here 
                                training_batch, training_labels = unison_shuffled_copies(training_batch, training_labels)
                                training_batch = training_batch.reshape(training_batch.shape[0], training_batch.shape[1], 1)
                                fetches = [all_loss, output_loss, y_pred, train_op]
                                if tr_counter % log_step == 0:
                                    fetches.append(summary_op)
                                    
                                fetched = sess.run(fetches, {input_tensor: training_batch, y: training_labels, drop_out: drop_rate, isTrain: True})
                                
                                if tr_counter % log_step == 0: # 
                                    summary_writer.add_summary(fetched[-1], tr_counter)
                                    summary_writer.flush()
                    
                                tr_loss_task = fetch_all_loss(fetched[0], tr_loss_task) 
                                tr_output_loss += fetched[1]
                                
                                train_pred_task = fetch_pred_labels(fetched[2], train_pred_task)
                                train_true_task = fetch_true_labels(training_labels, train_true_task)
                    
                    #            if tr_counter % log_step == 0:  
                    #                print("train batch: {}/{} - epoch {}/{} - output loss: {:.2f}".format(tr_counter, tr_steps, epoch_counter, epoch, fetched[1]))
                                
                            ## loss after epoch
                            tr_epoch_loss = np.true_divide(tr_loss_task, tr_steps)
                            train_loss_dict.update({epoch_counter: tr_epoch_loss})
                            tr_output_loss = np.true_divide(tr_output_loss, tr_steps)
                            
                            ## performance matrix after each epoch
                            tr_epoch_accuracy, tr_epoch_precision, tr_epoch_recall, tr_epoch_f1_score, tr_epoch_kappa = get_results_ssl(train_true_task, np.asarray(train_pred_task, int))
                            print("fold: {}/{} - training: epoch {}/{} - 'org', 'noised', 'scaled', 'neg', 'flip', 'perm', 'time_warp' - f1 score: {} - loss: {}".format(k, total_fold, epoch_counter, epoch, tr_epoch_f1_score, tr_epoch_loss))
                            
                            tr_ssl_result = write_result(tr_epoch_accuracy, tr_epoch_precision, tr_epoch_recall, tr_epoch_f1_score, tr_epoch_kappa, epoch_counter, tr_ssl_result)
                            
                            write_summary(loss = tr_epoch_loss, total_loss = tr_output_loss, f1_score = tr_epoch_f1_score, epoch_counter = epoch_counter, isTraining = True, summary_writer = summary_writer)
                    
#                            save_path = saver.save(sess, dirname + "\\selfsupervised\\10fold\\saved_model\\SSL_model.ckpt")   #D:/Data Repo/final codes/multi_task/final_model.ckpt
#                            print("SSL Model saved in path: %s" % save_path) 
                    
                    
                            te_loss_task = np.zeros((len(transform_task), 1), dtype  = np.float32)
                            test_pred_task = np.zeros((len(transform_task), actual_batch_size), dtype  = np.float32)-1
                            test_true_task = np.zeros((len(transform_task), actual_batch_size), dtype  = np.float32)-1
                            te_output_loss = 0
                           
                            te_total_gen_op = make_total_batch(data = test_ECG, length = testing_length, batchsize = batchsize, 
                                                               noise_amount=noise_param, 
                                                               scaling_factor=scale_param, 
                                                               permutation_pieces=permu_param, 
                                                               time_warping_pieces=tw_piece_param, 
                                                               time_warping_stretch_factor= twsf_param, 
                                                               time_warping_squeeze_factor= 1/twsf_param)
                    
                            for testing_batch, testing_labels, te_counter, te_steps in te_total_gen_op:
                                
                                ## run the model here 
                                fetches = [all_loss, output_loss, y_pred]
                                    
                                fetched = sess.run(fetches, {input_tensor: testing_batch, y: testing_labels, drop_out: 0.0, isTrain: False})
                                
                    
                                te_loss_task = fetch_all_loss(fetched[0], te_loss_task)
                                te_output_loss += fetched[1]
                                test_pred_task = fetch_pred_labels(fetched[2], test_pred_task)
                                test_true_task = fetch_true_labels(testing_labels, test_true_task)
                    
                    #            if te_counter % log_step == 0:
                    #                print("test batch: {}/{} - epoch {}/{} - output loss: {:.2f}".format(te_counter, te_steps, epoch_counter, epoch, fetched[1]))
                    #            
                            ## loss after epoch
                            te_epoch_loss = np.true_divide(te_loss_task, te_steps)
                            test_loss_dict.update({epoch_counter: te_epoch_loss})
                            te_output_loss = np.true_divide(te_output_loss, te_steps)
                    
                            ## performance matrix after each epoch
                            te_epoch_accuracy, te_epoch_precision, te_epoch_recall, te_epoch_f1_score, te_epoch_kappa = get_results_ssl(test_true_task, test_pred_task)
                            print("fold: {}/{} - testing: epoch {}/{} - 'org', 'noised', 'scaled', 'neg', 'flip', 'perm', 'time_warp' - f1 score: {} - loss: {}".format(k, total_fold, epoch_counter, epoch, te_epoch_f1_score, te_epoch_loss))
                            
                            te_ssl_result = write_result(te_epoch_accuracy, te_epoch_precision, te_epoch_recall, te_epoch_f1_score, te_epoch_kappa, epoch_counter, te_ssl_result)
                    
                            write_summary(loss = te_epoch_loss, total_loss = te_output_loss, f1_score = te_epoch_f1_score, epoch_counter = epoch_counter, isTraining = False, summary_writer = summary_writer)
                            
                    
                #            if ((epoch_counter > 0) & (epoch_counter % supervised_interval == 0) or (epoch_counter == epoch - 1)):  ## perform downstream task after evry 5 epochs
                            if epoch_counter == epoch-1 : #1==1:
                                """
                                supervised task of self supervised learning
                                """
                                """  swell """
                        #        
                                x_tr = swell_data[swell_train_index[k], 4:]
                                x_te = swell_data[swell_test_index[k], 4:]
                                
                                x_tr_feature = extract_feature(x_original = x_tr, featureset_size = featureset_size, batch_super = batchsize, input_tensor = input_tensor, isTrain = isTrain, drop_out = drop_out, extract_layer = main_branch, sess = sess)
                                x_te_feature = extract_feature(x_original = x_te, featureset_size = featureset_size, batch_super = batchsize, input_tensor = input_tensor, isTrain = isTrain, drop_out = drop_out, extract_layer = main_branch, sess = sess)
                        
                                general_supervised_task_swell(x_tr_feature = x_tr_feature, y_tr = train_swell_input_stress, x_te_feature = x_te_feature, y_te = test_swell_input_stress, identifier = 'swell_input_stress', kfold = flag, epoch_super = 200, log_dir = super_logs)        
                                general_supervised_task_swell(x_tr_feature = x_tr_feature, y_tr = train_swell_arousal, x_te_feature = x_te_feature, y_te = test_swell_arousal, identifier = 'swell_arousal', kfold = flag, epoch_super = 200, log_dir = super_logs)
                                general_supervised_task_swell(x_tr_feature = x_tr_feature, y_tr = train_swell_valence, x_te_feature = x_te_feature, y_te = test_swell_valence, identifier = 'swell_valence', kfold = flag, epoch_super = 200, log_dir = super_logs)
                        
                                
                                """
                                supervised task of self supervised learning
                                """
                                """  wesad """  
                                
                                x_tr = wesad_data[wesad_train_index[k], 2:]
                                x_te = wesad_data[wesad_test_index[k], 2:]
                                
                                x_tr_feature = extract_feature(x_original = x_tr, featureset_size = featureset_size, batch_super = batchsize, input_tensor = input_tensor, isTrain = isTrain, drop_out = drop_out, extract_layer = main_branch, sess = sess)
                                x_te_feature = extract_feature(x_original = x_te, featureset_size = featureset_size, batch_super = batchsize, input_tensor = input_tensor, isTrain = isTrain, drop_out = drop_out, extract_layer = main_branch, sess = sess)
                                
                                general_supervised_task_wesad(x_tr_feature = x_tr_feature, y_tr = train_wesad_stress, x_te_feature = x_te_feature, y_te = test_wesad_stress, identifier = 'wesad_stress', kfold = flag, epoch_super = 200, log_dir = super_logs)
                #        
                                
                                """
                                supervised task of self supervised learning
                                """
                                """  dreamer """  
                                
                                x_tr = dreamer_data[dreamer_train_index[k], 3:]
                                x_te = dreamer_data[dreamer_test_index[k], 3:]
                                    
                                x_tr_feature = extract_feature(x_original = x_tr, featureset_size = featureset_size, batch_super = batchsize, input_tensor = input_tensor, isTrain = isTrain, drop_out = drop_out, extract_layer = main_branch, sess = sess)
                                x_te_feature = extract_feature(x_original = x_te, featureset_size = featureset_size, batch_super = batchsize, input_tensor = input_tensor, isTrain = isTrain, drop_out = drop_out, extract_layer = main_branch, sess = sess)
                                
                                general_supervised_task_dreamer(x_tr_feature = x_tr_feature, y_tr = train_dreamer_arousal, x_te_feature = x_te_feature, y_te = test_dreamer_arousal, identifier = 'dreamer_arousal', kfold = flag, epoch_super = 200, log_dir = super_logs) 
                                general_supervised_task_dreamer(x_tr_feature = x_tr_feature, y_tr = train_dreamer_valence, x_te_feature = x_te_feature, y_te = test_dreamer_valence, identifier = 'dreamer_valence', kfold = flag, epoch_super = 200, log_dir = super_logs)
                                
                        
                                """
                                supervised task of self supervised learning
                                """
                                """  amigos """  
                                
                                x_tr = amigos_data[amigos_train_index[k], 3:]
                                x_te = amigos_data[amigos_test_index[k], 3:]
                                    
                                x_tr_feature = extract_feature(x_original = x_tr, featureset_size = featureset_size, batch_super = batchsize, input_tensor = input_tensor, isTrain = isTrain, drop_out = drop_out, extract_layer = main_branch, sess = sess)
                                x_te_feature = extract_feature(x_original = x_te, featureset_size = featureset_size, batch_super = batchsize, input_tensor = input_tensor, isTrain = isTrain, drop_out = drop_out, extract_layer = main_branch, sess = sess)
                                
                                general_supervised_task_amigos(x_tr_feature = x_tr_feature, y_tr = train_amigos_arousal, x_te_feature = x_te_feature, y_te = test_amigos_arousal, identifier = 'amigos_arousal', kfold = flag, epoch_super = 200, log_dir = super_logs) 
                                general_supervised_task_amigos(x_tr_feature = x_tr_feature, y_tr = train_amigos_valence, x_te_feature = x_te_feature, y_te = test_amigos_valence, identifier = 'amigos_valence', kfold = flag, epoch_super = 200, log_dir = super_logs)
                           
                        np.save(tr_ssl_loss_filename, train_loss_dict)
                        np.save(te_ssl_loss_filename, test_loss_dict)
                    
                        np.save(tr_ssl_result_filename, tr_ssl_result)
                        np.save(te_ssl_result_filename, te_ssl_result)
                        try:
                            send_email(subject_line = str(k) + " fold is running...", body = " ")
                        except:
                            print("issue in sending email")
                
