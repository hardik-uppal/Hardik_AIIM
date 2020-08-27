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
from keras_vggface.vggface import VGGFace
import keras
import pickle
from attention_module import cbam_block
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Activation, Dropout, BatchNormalization
#import config
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import tensorflow
import cv2 
import imgaug.augmenters as iaa

K.set_image_data_format('channels_last')

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
#    pre_trained_model = VGGFace(include_top=True, input_shape=input_shape)
##    pre_trained_model.summary()
#    final_layer = pre_trained_model.layers[-3].output
#    output = layers.Dense(n_class, activation='softmax', name='output')(final_layer)
#
#    # Models for training and evaluation (prediction)
#    train_model = models.Model(pre_trained_model.input,  output)
#    for layer in train_model.layers[:-6]:
#        layer.trainable = False
#
#
#    return train_model

def VGGFace_multimodal(input_shape, n_class):
    """
    
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    
    :return:  Keras Model used for training
    """
    # RGB MODALITY BRANCH OF CNN
    inputs_rgb = layers.Input(shape=input_shape)
    ########################VGG/RESNET or any other network
    vgg_model_rgb = VGGFace(include_top=False, input_shape=input_shape)
    
    ########################
    conv_model_rgb = vgg_model_rgb(inputs_rgb)


    
    ## get weights
    pickle_in = open("vgg16_face_conv_weights_name.pkl","rb")
    w_and_b = pickle.load(pickle_in)
    # DEPTH MODALITY BRANCH OF CNN
    inputs_depth = layers.Input(shape=input_shape)
    #######################VGG/RESNET or any other network
#    vgg_model_depth = VGGFace(include_top=False, input_shape=input_shape)
    # Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv1_1'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv1_1'][1]), name='conv1_1d')(inputs_depth)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv1_2'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv1_2'][1]), name='conv1_2d')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1d')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv2_1'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv2_1'][1]), name='conv2_1d')(
        x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv2_2'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv2_1'][1]), name='conv2_2d')(
        x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2d')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv3_1'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv3_1'][1]), name='conv3_1d')(
        x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv3_2'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv3_2'][1]), name='conv3_2d')(
        x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv3_3'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv3_3'][1]), name='conv3_3d')(
        x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3d')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv4_1'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv4_1'][1]), name='conv4_1d')(
        x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv4_2'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv4_2'][1]), name='conv4_2d')(
        x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv4_3'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv4_3'][1]), name='conv4_3d')(
        x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4d')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv5_1'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv5_1'][1]), name='conv5_1d')(
        x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv5_2'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv5_2'][1]), name='conv5_2d')(
        x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv5_3'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv5_3'][1]), name='conv5_3d')(
        x)
    conv_model_depth = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5d')(x)
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
    fc6 = layers.Dense(2048, activation='relu', name='fc6')(flat_model)
    bn_1 = BatchNormalization(name='1_bn')(fc6)
    dropout_1 = layers.Dropout(0.5)(bn_1)
    
    
    
#    flatten_concat = layers.Flatten(name='flatten_concat')(merge_rgb_depth)
#    fc6 = layers.Dense(2048, activation='relu', name='fc6')(merge_rgb_depth)
    fc7 = layers.Dense(1024, activation='relu', name='fc7')(dropout_1)
    bn_2 = BatchNormalization(name='2_bn')(fc7)
    dropout_2 = layers.Dropout(0.5)(bn_2)
    
#    
    fc8 = layers.Dense(512, activation='relu', name='fc8')(dropout_2)
    bn_3 = BatchNormalization(name='3_bn')(fc8)
    dropout_3 = layers.Dropout(0.5)(bn_3)
    
#    fc9 = layers.Dense(2048, activation='relu', name='fc9')(dropout_3)
#    bn_4 = BatchNormalization(name='4_bn')(fc9)
#    dropout_4 = layers.Dropout(0.5)(bn_4)
    
    #VECTORIZING OUTPUT
    output = layers.Dense(n_class, activation='softmax', name='output')(dropout_3)
    
    # MODAL [INPUTS , OUTPUTS]
    train_model = models.Model(inputs=[inputs_rgb, inputs_depth], outputs=[output])
#    weights_path = 'D:/tutorial/rgb+depth+thermal/IIITD_RGBD_new/result_multimodal_attention_bn_fold5_dropout-0.5_2fc_cont2/weights-08.h5'
#    train_model.load_weights(weights_path)
    
    train_model.summary()
    for layer in train_model.layers[:-26]:
        layer.trainable = False


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


def train(model, fold, args):
    """
    Training 
    :param model: the  model
    
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
#    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    
    save_dirc = args.save_dir + str(fold)
    log = callbacks.CSVLogger(save_dirc + '/log.csv')
    es_cb = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    tb = callbacks.TensorBoard(log_dir=save_dirc + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(save_dirc + '/weights-best.h5', monitor='val_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=['categorical_crossentropy'],
                  
                  metrics=['accuracy'])

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(batch_size, val_train): 
#        train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True, validation_split=0.2)
#    
#        train_generator = train_datagen.flow_from_directory("D:/RGB_D_Dataset/train/depth/",target_size=(224, 224), color_mode="rgb",
#                                                            batch_size=args.batch_size, class_mode='categorical',subset='training')
#        
#        validation_generator = train_datagen.flow_from_directory("D:/RGB_D_Dataset/train/depth/",target_size=(224, 224), color_mode="rgb",
#                                                                 batch_size=args.batch_size,class_mode='categorical',subset='validation')
#        
        batch_size = int(batch_size/5)
        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)  
        generator_rgb = train_datagen.flow_from_directory(directory="D:/RGB_D_Dataset_new/fold{}/train/RGB/".format(fold), target_size=(224, 224), color_mode="rgb",
                                                      batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)
        generator_depth = train_datagen.flow_from_directory(directory="D:/RGB_D_Dataset_new/fold{}/train/depth/".format(fold), target_size=(224, 224), color_mode="rgb",
                                                      batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)
        
        generator_rgb_val = train_datagen.flow_from_directory(directory="D:/RGB_D_Dataset_new/fold{}/test/RGB/".format(fold), target_size=(224, 224), color_mode="rgb",
                                                      batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)
        generator_depth_val = train_datagen.flow_from_directory(directory="D:/RGB_D_Dataset_new/fold{}/test/depth/".format(fold), target_size=(224, 224), color_mode="rgb",
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
    model.fit_generator(generator=train_generator(args.batch_size,'train'),
                        steps_per_epoch=int(424 / int(args.batch_size/5)),##936 curtin faces###424 fold1 iiitd
                        epochs=args.epochs,
                        validation_data=train_generator(args.batch_size,'val'),
                        validation_steps = int( 4181 / int(args.batch_size/5)),##4108 curtin faces###4181 fold1 iiitd
                        callbacks=[log, tb, checkpoint, lr_decay, es_cb])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(save_dirc + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % save_dirc)

    from utils import plot_log
    plot_log(save_dirc + '/log.csv', show=True)

    return model

#
#def test(model,fold, args):
##    x_test, y_test = data
##    y_pred, x_recon = model.predict(x_test, batch_size=100)
##    model.compile(optimizer=optimizers.Adam(lr=args.lr),
##              loss=['categorical_crossentropy'],
##              
##              metrics=['accuracy'])
##    model.load_weights('./IIITD_RGBD_new/result_multimodal_attention_fold12/weights-22.h5')
##    
##    
#    def test_generator(batch_size=1):
#        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)  
#        generator_rgb = train_datagen.flow_from_directory(directory="D:/RGB_D_Dataset_new/fold{}/test/RGB/".format(fold), target_size=(224, 224), color_mode="rgb",
#                                                      batch_size=1, class_mode="categorical", shuffle=True, seed=42)
#        generator_depth = train_datagen.flow_from_directory(directory="D:/RGB_D_Dataset_new/fold{}/test/depth/".format(fold), target_size=(224, 224), color_mode="rgb",
#                                                      batch_size=1, class_mode="categorical", shuffle=True, seed=42)
#
#        while 1:
#            x_batch_rgb, y_batch_rgb = generator_rgb.next()
#            x_batch_depth, y_batch_depth = generator_depth.next()
#            yield [[x_batch_rgb, x_batch_depth], y_batch_rgb]
#    
#    scores = model.evaluate_generator(generator=test_generator(1),steps = 4181)
#    print('Test loss: {} ; Accuracy on Test: {}'.format(scores[0],scores[1]))
#



#def load_mnist():
#    # the data, shuffled and split between train and test sets
#    from keras.datasets import mnist
#    (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
#    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
#    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
#    y_train = to_categorical(y_train.astype('float32'))
#    y_test = to_categorical(y_test.astype('float32'))
#    return (x_train, y_train), (x_test, y_test)

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
    
    
  

if __name__ == "__main__":
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="RGB-D network")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=20, type=int)### divisible by 5
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
    parser.add_argument('--save_dir', default='./IIITD_RGBD_new/final')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
#    (x_train, y_train), (x_test, y_test) = load_mnist()

    # define model
#    model = VGGFace_multimodal(input_shape=(224,224,3), n_class=106)
#    model.summary()

#    if not args.testing:
    for fold in range(1,6):
        model = VGGFace_multimodal(input_shape=(224,224,3), n_class=106)
        model.summary()
        model_trained = train(model=model,fold = fold, args=args)
        reset_keras()
#            model_trained = VGGFace_multimodal(input_shape=(224,224,3), n_class=106)
#        test(model=model_trained,fold = fold, args=args)
  # as long as weights are given, will run testing