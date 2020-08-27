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
from attention_module import cbam_block
#import config


K.set_image_data_format('channels_last')


def VGGnet(input_shape, n_class):
    """
    
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    
    :return:  Keras Model used for training
    """
#    x = layers.Input(shape=input_shape)

    pickle_in = open("vgg16_face_conv_weights_name.pkl","rb")
    w_and_b = pickle.load(pickle_in)
    add_weights = np.expand_dims(w_and_b['conv1_1'][0][1], axis=2)
#    add_weights = np.zeros((3,3,1,64))  initiate depth channel with zeros
    w_and_b['conv1_1'][0] = np.concatenate((w_and_b['conv1_1'][0],add_weights), axis = 2)
    # DEPTH MODALITY BRANCH OF CNN
    inputs_rgb_d = layers.Input(shape=input_shape)
    #######################VGG/RESNET or any other network
#    vgg_model_depth = VGGFace(include_top=False, input_shape=input_shape)
    # Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv1_1'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv1_1'][1]), name='conv1_1d')(inputs_rgb_d)
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
    conv_model_rgb = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5d')(x)
    attention_features = cbam_block(conv_model_rgb, att_type='Channel')
    flat_model = layers.Flatten(name='flatten')(attention_features)
    fc6 = layers.Dense(1024, activation='relu', name='fc6')(flat_model)
    dropout_layer_1 = layers.Dropout(0.2)(fc6)
    fc7 = layers.Dense(512, activation='relu', name='fc7')(dropout_layer_1)
    dropout_layer_2 = layers.Dropout(0.2)(fc7)
    output = layers.Dense(n_class, activation='softmax', name='output')(dropout_layer_2)

#     Models for training and evaluation (prediction)
    train_model = models.Model(inputs_rgb_d,  output)
#    for layer in train_model.layers[:-14]:
#        layer.trainable = False


    return train_model




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
                  loss=['categorical_crossentropy'],
                  
                  metrics=['accuracy'])

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(batch_size,t_or_v = 't'):
        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)  
        generator_train_rgb = train_datagen.flow_from_directory(directory="D:/RGB_D_Dataset/train/RGB/", target_size=(224, 224), color_mode="rgb",
                                                      batch_size=batch_size, class_mode="categorical",subset='training', shuffle=True, seed=42)
        generator__train_depth = train_datagen.flow_from_directory(directory="D:/RGB_D_Dataset/train/depth/", target_size=(224, 224), color_mode="grayscale",
                                                      batch_size=batch_size, class_mode="categorical",subset='training', shuffle=True, seed=42)
        
        
        generator_val_rgb = train_datagen.flow_from_directory(directory="D:/RGB_D_Dataset/train/RGB/", target_size=(224, 224), color_mode="rgb",
                                                      batch_size=batch_size, class_mode="categorical",subset='validation', shuffle=True, seed=42)
        generator_val_depth = train_datagen.flow_from_directory(directory="D:/RGB_D_Dataset/train/depth/", target_size=(224, 224), color_mode="grayscale",
                                                      batch_size=batch_size, class_mode="categorical",subset='validation', shuffle=True, seed=42)
        if t_or_v == 't':
            while 1:
                x_batch_rgb, y_batch_rgb = generator_train_rgb.next()
                x_batch_depth, y_batch_depth = generator__train_depth.next()
                x_total = np.concatenate([x_batch_rgb,x_batch_depth], axis = -1)
                yield [x_total, y_batch_rgb]
        elif t_or_v == 'v':
            while 1:
                x_batch_rgb, y_batch_rgb = generator_val_rgb.next()
                x_batch_depth, y_batch_depth = generator_val_depth.next()
                x_total = np.concatenate([x_batch_rgb,x_batch_depth], axis = -1)
                yield [x_total, y_batch_rgb]
            
            
#    def val_generator(batch_size):
#        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)  
#        generator_rgb = train_datagen.flow_from_directory(directory="D:/EURECOM_Kinect_Face_Dataset/RGB/val", target_size=(224, 224), color_mode="rgb",
#                                                      batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)
#        generator_depth = train_datagen.flow_from_directory(directory="D:/EURECOM_Kinect_Face_Dataset/depth/val", target_size=(224, 224), color_mode="grayscale",
#                                                      batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)
#
#        while 1:
#            x_batch_rgb, y_batch_rgb = generator_rgb.next()
#            x_batch_depth, y_batch_depth = generator_depth.next()
#            x_total = np.concatenate([x_batch_rgb,x_batch_depth], axis = -1)
#            yield [x_total, y_batch_rgb]
            
    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(args.batch_size,'t'),
                        steps_per_epoch=int( 106759 / args.batch_size),
                        epochs=args.epochs,
                        validation_data=train_generator(args.batch_size,'v'),
                        validation_steps = int( 26642 / args.batch_size),
                        callbacks=[log, tb, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model_1.h5')
    print('Trained model saved to \'%s/trained_model_1.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log_1.csv', show=True)

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
    parser.add_argument('--epochs', default=5000, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--lr', default=0.0001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
#    parser.add_argument('--lam_recon', default=0.392, type=float,
#                        help="The coefficient for the loss of decoder")
#    parser.add_argument('-r', '--routings', default=3, type=int,
#                        help="Number of iterations used in routing algorithm. should > 0")
#    parser.add_argument('--shift_fraction', default=0.1, type=float,
#                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./all_data/result_channel_att')
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
    model = VGGnet(input_shape=(224,224,4), n_class=72)
    model.summary()
    model.load_weights('./all_data/result_channel_att/weights-01.h5')

    if not args.testing:
        train(model=model, args=args)
  # as long as weights are given, will run testing