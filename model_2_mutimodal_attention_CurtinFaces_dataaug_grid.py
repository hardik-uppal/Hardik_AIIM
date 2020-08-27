# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 00:10:40 2019

@author: hardi
"""

import gc
import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
#from utils import combine_images
from PIL import Image
#from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
#from keras_vggface.vggface import VGGFace
from model_vgg_face import VGG16
import keras
import pickle
from attention_module import cbam_block
#import config
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Activation, Dropout, BatchNormalization
from keras import backend as K
from triplet_loss_semi_hard import triplet_loss_adapted_from_tf
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import tensorflow
import cv2 
import imgaug.augmenters as iaa

K.set_image_data_format('channels_last')

def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()
    
    try:
        del model # this is from global space - change this as you need
    except:
        pass
    
    print(gc.collect()) # if it's done something you should see a number being outputted
    
    # use the same config as you used to create the session
    config = tensorflow.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.Session(config=config))    
    
    
    
    
def normalized(image):
    norm_batch =[]
    epsilon = 0.00000000000001
    for i in image:
        
        norm=np.zeros((224,224,3),np.float32)
        norm_rgb=np.zeros((224,224,3),np.uint8)

        b=i[:,:,0]
        g=i[:,:,1]
        r=i[:,:,2]

        sum = b+g+r+epsilon

        norm[:,:,0]=b/sum
        norm[:,:,1]=g/sum
        norm[:,:,2]=r/sum

        norm_rgb=cv2.convertScaleAbs(norm)
        norm_batch.append(norm_rgb)
    norm_batch= np.asarray(norm_batch)
    return norm_batch
#
#def VGGnet(input_shape, n_class):
#    """
#    
#    :param input_shape: data shape, 3d, [width, height, channels]
#    :param n_class: number of classes
#    
#    :return:  Keras Model used for training
#    """
##    x = layers.Input(shape=input_shape)
#
#    # Layer 1: Just a conventional Conv2D layer

    

#
#def conv_block(units, dropout=0.2, activation='relu', block=1, layer=1, kernel_initializer=None):
#
#    def layer_wrapper(inp):
#        x = Conv2D(units, (3, 3), padding='same',kernel_initializer=kernel_initializer, name='block{}_conv{}'.format(block, layer))(inp)
##        x = BatchNormalization(name='block{}_bn{}'.format(block, layer))(x)
#        x = Activation(activation, name='block{}_act{}'.format(block, layer))(x)
##        x = Dropout(dropout, name='block{}_dropout{}'.format(block, layer))(x)
#        return x
#
#    return layer_wrapper
#
##    def dense_block(units, dropout=0.2, activation='relu', name='fc1'):
##    
##        def layer_wrapper(inp):
##            x = Dense(units, name=name)(inp)
##            x = BatchNormalization(name='{}_bn'.format(name))(x)
##            x = Activation(activation, name='{}_act'.format(name))(x)
##            x = Dropout(dropout, name='{}_dropout'.format(name))(x)
##            return x
##    
##        return layer_wrapper
#        
#
#def VGG16_BN(input_tensor=None, input_shape=None, conv_dropout=0.1, activation='relu'):
#    """Instantiates the VGG16 architecture with Batch Normalization
#    # Arguments
#        input_tensor: Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model.
#        input_shape: shape tuple
#        classes: optional number of classes to classify images
#    # Returns
#        A Keras model instance.
#    """
#    img_input = Input(shape=input_shape) if input_tensor is None else (
#        Input(tensor=input_tensor, shape=input_shape) if not K.is_keras_tensor(input_tensor) else input_tensor)
#    
#    pickle_in = open("vgg16_face_conv_weights_name.pkl","rb")
#    w_and_b = pickle.load(pickle_in)
#
#    # Block 1
#    x = conv_block(64, dropout=conv_dropout, activation=activation, block=1, layer=1,kernel_initializer= keras.initializers.Constant(w_and_b['conv1_1'][0]))(img_input)
#    x = conv_block(64, dropout=conv_dropout, activation=activation, block=1, layer=2,kernel_initializer= keras.initializers.Constant(w_and_b['conv1_2'][0]))(x)
#    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
#
#    # Block 2
#    x = conv_block(128, dropout=conv_dropout, activation=activation, block=2, layer=1,kernel_initializer= keras.initializers.Constant(w_and_b['conv2_1'][0]))(x)
#    x = conv_block(128, dropout=conv_dropout, activation=activation, block=2, layer=2,kernel_initializer= keras.initializers.Constant(w_and_b['conv2_2'][0]))(x)
#    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
#
#    # Block 3
#    x = conv_block(256, dropout=conv_dropout, activation=activation, block=3, layer=1,kernel_initializer= keras.initializers.Constant(w_and_b['conv3_1'][0]))(x)
#    x = conv_block(256, dropout=conv_dropout, activation=activation, block=3, layer=2,kernel_initializer= keras.initializers.Constant(w_and_b['conv3_2'][0]))(x)
#    x = conv_block(256, dropout=conv_dropout, activation=activation, block=3, layer=3,kernel_initializer= keras.initializers.Constant(w_and_b['conv3_3'][0]))(x)
#    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
#
#    # Block 4
#    x = conv_block(512, dropout=conv_dropout, activation=activation, block=4, layer=1,kernel_initializer= keras.initializers.Constant(w_and_b['conv4_1'][0]))(x)
#    x = conv_block(512, dropout=conv_dropout, activation=activation, block=4, layer=2,kernel_initializer= keras.initializers.Constant(w_and_b['conv4_2'][0]))(x)
#    x = conv_block(512, dropout=conv_dropout, activation=activation, block=4, layer=3,kernel_initializer= keras.initializers.Constant(w_and_b['conv4_3'][0]))(x)
#    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
#
#    # Block 5
#    x = conv_block(512, dropout=conv_dropout, activation=activation, block=5, layer=1,kernel_initializer= keras.initializers.Constant(w_and_b['conv5_1'][0]))(x)
#    x = conv_block(512, dropout=conv_dropout, activation=activation, block=5, layer=2,kernel_initializer= keras.initializers.Constant(w_and_b['conv5_2'][0]))(x)
#    x = conv_block(512, dropout=conv_dropout, activation=activation, block=5, layer=3,kernel_initializer= keras.initializers.Constant(w_and_b['conv5_3'][0]))(x)
#    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
#
#    
#    train_model = models.Model(img_input,  x)
#
#
##    pre_trained_model = VGG16_BN(input_tensor=None, input_shape=input_shape, classes=1000, conv_dropout=0.1, dropout=0.3, activation='relu')
##    trained_model_out = pre_trained_model(input)
##    pre_trained_model.summary()
##        final_layer = pre_trained_model.layers[-3].output
##        x = Dense(n_class, activation='softmax', name='output')(final_layer)
##        x = BatchNormalization()(x)
#
# #    Models for training and evaluation (prediction)
##    train_model = models.Model(img_input,  output)
##    for layer in train_model.layers[:-6]:
##        layer.trainable = False
#
#
#
#
#
#    return train_model

def VGGFace_multimodal(input_shape, n_class, dropout_fc1,dropout_fc2,dropout_fc3, units_fc1,units_fc2,units_fc3):
    """
    
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    
    :return:  Keras Model used for training
    """
    # RGB MODALITY BRANCH OF CNN
    inputs_rgb = layers.Input(shape=input_shape)
    ########################VGG/RESNET or any other network
    vgg_model_rgb = VGG16(include_top=False, weights='vggface', input_tensor=None, input_shape=input_shape, pooling=None, type_name='rgb')
    conv_model_rgb = vgg_model_rgb(inputs_rgb)
    
    ########################
    
    inputs_depth = layers.Input(shape=input_shape)
    vgg_model_depth = VGG16(include_top=False, weights='vggface', input_tensor=None, input_shape=input_shape, pooling=None, type_name='depth')
    conv_model_depth = vgg_model_depth(inputs_depth)


    ######################
#    conv_model_depth = vgg_model_depth(inputs_depth)

    
#    fc6_rgb = layers.Dense(2048, activation='relu', name='fc6_rgb')(dropout_rgb)
#    fc6_depth = layers.Dense(2048, activation='relu', name='fc6_depth')(dropout_depth)
    
    
    # CONACTENATE the ends of RGB & DEPTH 

    merge_rgb_depth = layers.concatenate([conv_model_rgb,conv_model_depth], axis=-1)
    attention_features = cbam_block(merge_rgb_depth)
#  
    
#    
#    ############ for RGB
#    flat_model_rgb = layers.Flatten(name='flatten_rgb')(conv_model_rgb)
#    fc6_rgb = layers.Dense(2048, activation='relu', name='fc6_rgb')(flat_model_rgb)
#    dropout_rgb = layers.Dropout(0.2)(fc6_rgb)
    

    ######## for Depth 
    flat_model = layers.Flatten(name='flatten')(attention_features)
    fc6 = layers.Dense(units_fc1 , activation='relu', name='fc6')(flat_model)
    bn_1 = BatchNormalization(name='1_bn')(fc6)
    dropout_1 = layers.Dropout(dropout_fc1)(bn_1)
    
    
    
#    flatten_concat = layers.Flatten(name='flatten_concat')(merge_rgb_depth)
#    fc6 = layers.Dense(2048, activation='relu', name='fc6')(merge_rgb_depth)
    fc7 = layers.Dense(units_fc2, activation='relu', name='fc7')(dropout_1)
    bn_2 = BatchNormalization(name='2_bn')(fc7)
    dropout_2 = layers.Dropout(dropout_fc2)(bn_2)
    
    fc8 = layers.Dense(units_fc3, activation='relu', name='fc8')(dropout_2)
    bn_3 = BatchNormalization(name='3_bn')(fc8)
    dropout_3 = layers.Dropout(dropout_fc3)(bn_3)
    

    
    #VECTORIZING OUTPUT
    output = layers.Dense(n_class, activation='softmax', name='output')(dropout_3)
    
    # MODAL [INPUTS , OUTPUTS]
    train_model = models.Model(inputs=[inputs_rgb, inputs_depth], outputs=[output])
    
#    weights_path = 'CurtinFaces/vgg_multimodal_dropout-0.5_3fc-512/weights-25.h5'
#    train_model.load_weights(weights_path)
    train_model.summary()
    for layer in train_model.layers[:-26]:
        layer.trainable = False
#    for layer in train_model.layers[2].layers[:-4]:
#        layer.trainable = False
#    for layer in train_model.layers[3].layers[:-4]:
#        layer.trainable = False




    return train_model


#def margin_loss(y_true, y_pred):
#    """
#    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
#    :param y_true: [None, n_classes]
#    :param y_pred: [None, num_capsule]
#    :return: a scalar loss value.
#    """
#    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
#        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
#
#    return K.mean(K.sum(L, 1))


def train(model,batch_size,save_dir, args):
    """
    Training 
    :param model: the  model
    
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
#    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    
    log = callbacks.CSVLogger(save_dir + '/log.csv')
    es_cb = callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
    tb = callbacks.TensorBoard(log_dir=save_dir + '/tensorboard-logs',
                               batch_size=batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(save_dir + '/weights-best.h5', monitor='val_accuracy',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=['categorical_crossentropy'],#triplet_loss_adapted_from_tf,
                  
                  metrics=['accuracy'])

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(batch_size, val_train): 

#        
        batch_size = int(batch_size/5)
        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)#, validation_split=0.2)  
        generator_rgb = train_datagen.flow_from_directory(directory="/home/harry/RGBD/CurtinFaces_crop/RGB/train/", target_size=(224, 224), color_mode="rgb",
                                                      batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)
        generator_depth = train_datagen.flow_from_directory(directory="/home/harry/RGBD/CurtinFaces_crop/normalized/DEPTH/train/", target_size=(224, 224), color_mode="rgb",
                                                      batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)
        
        generator_rgb_val = train_datagen.flow_from_directory(directory="/home/harry/RGBD/CurtinFaces_crop/RGB/test1/", target_size=(224, 224), color_mode="rgb",
                                                      batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)
        generator_depth_val = train_datagen.flow_from_directory(directory="/home/harry/RGBD/CurtinFaces_crop/normalized/DEPTH/test1/", target_size=(224, 224), color_mode="rgb",
                                                      batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)
        if val_train=='train':
            while 1:
                #rgb data aug
                x_batch_rgb, y_batch_rgb = generator_rgb.next()
                flip_img = iaa.Fliplr(1)(images=x_batch_rgb)
                rot_img = iaa.Affine(rotate=(-30, 30))(images=x_batch_rgb)

                shear_aug = iaa.Affine(shear=(-16, 16))(images=x_batch_rgb)
                trans_aug = iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)})(images=x_batch_rgb)
                x_batch_rgb_final = np.concatenate([x_batch_rgb,flip_img,rot_img,shear_aug,trans_aug],axis=0)
                y_batch_rgb_final = np.tile(y_batch_rgb,(5,1))
                ## depth data aug
                x_batch_depth, y_batch_depth = generator_depth.next()
                flip_img = iaa.Fliplr(1)(images=x_batch_depth)
                rot_img = iaa.Affine(rotate=(-30, 30))(images=x_batch_depth)

                shear_aug = iaa.Affine(shear=(-16, 16))(images=x_batch_depth)
                trans_aug = iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)})(images=x_batch_depth)
                x_batch_depth_final = np.concatenate([x_batch_depth,flip_img,rot_img,shear_aug,trans_aug],axis=0)
                y_batch_depth_final = np.tile(y_batch_rgb,(5,1))
                yield [[x_batch_rgb_final, x_batch_depth_final], y_batch_rgb_final]
        elif val_train == 'val':
            while 1:
                x_batch_rgb, y_batch_rgb = generator_rgb_val.next()
                x_batch_depth, y_batch_depth = generator_depth_val.next()
                yield [[x_batch_rgb, x_batch_depth], y_batch_rgb]
            
    
            
    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(batch_size,'train'),
                        steps_per_epoch=int(936 / int(batch_size/5)),##936 curtin faces###424 fold1 iiitd
                        epochs=args.epochs,
                        validation_data=train_generator(batch_size,'val'),
                        validation_steps = int( 2028 / int(batch_size/5)),##4108 curtin faces###4181 fold1 iiitd
                        callbacks=[log, tb, checkpoint, lr_decay, es_cb])
    # End: Training with data augmentation -----------------------------------------------------------------------#

#    model.save_weights(args.save_dir + '/trained_model.h5')
#    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(save_dir + '/log.csv', show=True)

    return model


def test(model,save_dir, args):

    
#    model.compile(optimizer=optimizers.Adam(lr=args.lr),
#              loss=['categorical_crossentropy'],
#              
#              metrics=['accuracy'])
#    model.load_weights('./CurtinFaces/result_multimodal_attention_1/weights-45.h5')
#    
    
    def test_generator(batch_size=1):
        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)  
        generator_rgb = train_datagen.flow_from_directory(directory="/home/harry/RGBD/CurtinFaces_crop/RGB/test2/", target_size=(224, 224), color_mode="rgb",
                                                      batch_size=1, class_mode="categorical", shuffle=True, seed=42)
        generator_depth = train_datagen.flow_from_directory(directory="/home/harry/RGBD/CurtinFaces_crop/normalized/DEPTH/test2/", target_size=(224, 224), color_mode="rgb",
                                                      batch_size=1, class_mode="categorical", shuffle=True, seed=42)

        while 1:
            x_batch_rgb, y_batch_rgb = generator_rgb.next()
            x_batch_depth, y_batch_depth = generator_depth.next()
            yield [[x_batch_rgb, x_batch_depth], y_batch_rgb]
    
    scores = model.evaluate_generator(generator=test_generator(1),steps = 1560)###test1 2028 ###test2 1560##test3 260
    print('Test loss: {} ; Accuracy on Test: {}'.format(scores[0],scores[1]))
    import csv
    test_log = save_dir + '/log_test.csv'
    with open(test_log, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(['Test loss: {} ; Accuracy on Test: {}'.format(scores[0],scores[1])])
    



if __name__ == "__main__":
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="RGB-D network")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=20, type=int)## only divisible by 5
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./testCurtinFaces/dataaug_vgg_multimodal_')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

#    if not os.path.exists(args.save_dir):
#        os.makedirs(args.save_dir)
        
    dropout_list = [0.2]#,0.3,0.4,0.0,0.1,0.5,0.6]
    unit_list= [2048,1024,512,4096,256]
    batch_size_list = [30]#,20,10,40,50]
    
    for units_fc1 in unit_list:
#                save_dirc = save_dirc + '_' +str(units_fc1)
        for units_fc2 in unit_list:
            for units_fc3 in unit_list:
                for dropout_fc1 in dropout_list:
#        save_dirc = args.save_dir +'_' +str(dropout_fc1)
                    for dropout_fc2 in dropout_list:
#            save_dirc = save_dirc + '_' +str(dropout_fc2)
                        for dropout_fc3 in dropout_list:

#                    save_dirc = save_dirc + '_' +str(units_fc2)

                            for batch_size in batch_size_list:
#                        save_dirc = save_dirc + '_' +str(batch_size)

                        
                    
                               save_dirc = args.save_dir +'_' +str(dropout_fc1) + '_' +str(dropout_fc2)+ '_' +str(dropout_fc3) + '_' +str(units_fc1) + '_' +str(units_fc2)+ '_' +str(units_fc3)  + '_' +str(batch_size)
                               if not os.path.exists(save_dirc):
                                   os.makedirs(save_dirc)
                               model = VGGFace_multimodal((224,224,3), 52, dropout_fc1,dropout_fc2,dropout_fc3, units_fc1, units_fc2,units_fc3)
                               model.summary()    
                               trained_model = train(model=model,batch_size=batch_size,save_dir = save_dirc, args=args)
                               test(model=trained_model,save_dir = save_dirc, args=args)
                               reset_keras()
#    model_trained = VGGFace_multimodal(input_shape=(224,224,3), n_class=52)
#    test(model=model_trained, args=args)
  # as long as weights are given, will run testing
