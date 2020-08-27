# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 00:10:40 2019

@author: hardi
"""


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
from data_gen_pandora import test_generator_with_files,training_generator_with_files
#import config


K.set_image_data_format('channels_last')


def VGGnet(input_shape, n_class):
    """
    
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    
    :return:  Keras Model used for training
    """
#    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    pre_trained_model = VGGFace(include_top=True, input_shape=input_shape)
#    pre_trained_model.summary()
    final_layer = pre_trained_model.layers[-3].output
    output = layers.Dense(n_class, activation='softmax', name='output')(final_layer)

    # Models for training and evaluation (prediction)
    train_model = models.Model(pre_trained_model.input,  output)
    for layer in train_model.layers[:-6]:
        layer.trainable = False


    return train_model

def VGGFace_multimodal(input_shape, n_class):
    """
    
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    
    :return:  Keras Model used for training
    """
    # RGB MODALITY BRANCH OF CNN
    inputs_rgb = layers.Input(shape=input_shape)
    ########################VGG/RESNET or any other network
    # Block 1
    conv1_1d = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1d')(inputs_rgb)
    conv1_2d = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2d')(conv1_1d)
    pool1d = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1d')(conv1_2d)

    # Block 2
    conv2_1d = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1d')(
        pool1d)
    conv2_2d = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2d')(
        conv2_1d)
    pool2d = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2d')(conv2_2d)

    # Block 3
    conv3_1d = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1d')(
        pool2d)
    conv3_2d = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2d')(
        conv3_1d)
    conv3_3d = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3d')(
        conv3_2d)
    pool3d = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3d')(conv3_3d)

    # Block 4
    conv4_1d = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1d')(
        pool3d)
    conv4_2d = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2d')(
        conv4_1d)
    conv4_3d = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3d')(
        conv4_2d)
    pool4d = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4d')(conv4_3d)

    # Block 5
    conv5_1d = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1d')(
        pool4d)
    conv5_2d = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2d')(
        conv5_1d)
    conv5_3d = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3d')(
        conv5_2d)
    conv_model_rgb = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5d')(conv5_3d)
    
    ########################
#    conv_model_rgb = vgg_model_rgb(inputs_rgb)

    flat_model_rgb = layers.Flatten(name='flatten_rgb')(conv_model_rgb)
    fc6_rgb = layers.Dense(1024, activation='relu', name='fc6_rgb')(flat_model_rgb)
    dropout_rgb = layers.Dropout(0.2)(fc6_rgb)
    
    ## get weights
    pickle_in = open("vgg16_face_conv_weights_name.pkl","rb")
    w_and_b = pickle.load(pickle_in)
    # DEPTH MODALITY BRANCH OF CNN
    inputs_depth = layers.Input(shape=input_shape)
    #######################VGG/RESNET or any other network
#    vgg_model_depth = VGGFace(include_top=False, input_shape=input_shape)
    # Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs_depth)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(
        x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(
        x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(
        x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(
        x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(
        x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(
        x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(
        x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(
        x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(
        x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(
        x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(
        x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)
    ######################
#    conv_model_depth = vgg_model_depth(inputs_depth)

    flat_model_depth = layers.Flatten(name='flatten_depth')(x)
    fc6_depth = layers.Dense(1024, activation='relu', name='fc6_depth')(flat_model_depth)
    dropout_depth = layers.Dropout(0.2)(fc6_depth)
    
#    fc6_rgb = layers.Dense(2048, activation='relu', name='fc6_rgb')(dropout_rgb)
#    fc6_depth = layers.Dense(2048, activation='relu', name='fc6_depth')(dropout_depth)
    
    
    # CONACTENATE the ends of RGB & DEPTH 
#    alpha = 0.7
    merge_rgb_depth = layers.concatenate([dropout_rgb,dropout_depth], axis=-1)
#    flatten_concat = layers.Flatten(name='flatten_concat')(merge_rgb_depth)
#    fc6 = layers.Dense(2048, activation='relu', name='fc6')(merge_rgb_depth)
    fc7 = layers.Dense(512, activation='relu', name='fc7')(merge_rgb_depth)
    
    
    #VECTORIZING OUTPUT
    output = layers.Dense(1, activation='sigmoid', name='output')(fc7)
    
    # MODAL [INPUTS , OUTPUTS]
    train_model = models.Model(inputs=[inputs_rgb, inputs_depth], outputs=[output])

    train_model.summary()


#    for layer in train_model.layers[:-6]:
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


def train(model, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
#    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=['binary_crossentropy'],
                  
                  metrics=['accuracy'])

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
#    def train_generator(batch_size):
#        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)  
#        generator_rgb = train_datagen.flow_from_directory(directory="D:/EURECOM_Kinect_Face_Dataset/RGB/train", target_size=(224, 224), color_mode="rgb",
#                                                      batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)
#        generator_depth = train_datagen.flow_from_directory(directory="D:/EURECOM_Kinect_Face_Dataset/depth/train", target_size=(224, 224), color_mode="rgb",
#                                                      batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)
#
#        while 1:
#            x_batch_rgb, y_batch_rgb = generator_rgb.next()
#            x_batch_depth, y_batch_depth = generator_depth.next()
#            yield [[x_batch_rgb, x_batch_depth], y_batch_rgb]
#            
#    def val_generator(batch_size):
#        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)  
#        generator_rgb = train_datagen.flow_from_directory(directory="D:/EURECOM_Kinect_Face_Dataset/RGB/val", target_size=(224, 224), color_mode="rgb",
#                                                      batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)
#        generator_depth = train_datagen.flow_from_directory(directory="D:/EURECOM_Kinect_Face_Dataset/depth/val", target_size=(224, 224), color_mode="rgb",
#                                                      batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)
#
#        while 1:
#            x_batch_rgb, y_batch_rgb = generator_rgb.next()
#            x_batch_depth, y_batch_depth = generator_depth.next()
#            yield [[x_batch_rgb, x_batch_depth], y_batch_rgb]
            
    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=training_generator_with_files(args),
                        steps_per_epoch=int(520 / args.batch_size),
                        epochs=args.epochs,
                        validation_data=test_generator_with_files(args),
                        validation_steps = int( 208 / args.batch_size),
                        callbacks=[log, tb, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data, args):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)




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


if __name__ == "__main__":
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="RGB-D network")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
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
    parser.add_argument('--save_dir', default='./result_pandora_val')
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
    model = VGGFace_multimodal(input_shape=(100,100,1), n_class=52)
    model.summary()

    if not args.testing:
        train(model=model, args=args)
  # as long as weights are given, will run testing
