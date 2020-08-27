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
import pickle
import keras
#from utils import combine_images
from PIL import Image
#from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
#from keras_vggface.vggface import VGGFace
#from VGG_BN_conv import VGG16_BN
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Activation, Dropout, BatchNormalization
#import config


K.set_image_data_format('channels_last')

def conv_block(units, dropout=0.2, activation='relu', block=1, layer=1, kernel_initializer=None):

    def layer_wrapper(inp):
        x = Conv2D(units, (3, 3), padding='same',kernel_initializer=kernel_initializer, name='block{}_conv{}'.format(block, layer))(inp)
        x = BatchNormalization(name='block{}_bn{}'.format(block, layer))(x)
        x = Activation(activation, name='block{}_act{}'.format(block, layer))(x)
        x = Dropout(dropout, name='block{}_dropout{}'.format(block, layer))(x)
        return x

    return layer_wrapper

#    def dense_block(units, dropout=0.2, activation='relu', name='fc1'):
#    
#        def layer_wrapper(inp):
#            x = Dense(units, name=name)(inp)
#            x = BatchNormalization(name='{}_bn'.format(name))(x)
#            x = Activation(activation, name='{}_act'.format(name))(x)
#            x = Dropout(dropout, name='{}_dropout'.format(name))(x)
#            return x
#    
#        return layer_wrapper
        

def VGG16_BN(input_tensor=None, input_shape=None, conv_dropout=0.1, activation='relu'):
    """Instantiates the VGG16 architecture with Batch Normalization
    # Arguments
        input_tensor: Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model.
        input_shape: shape tuple
        classes: optional number of classes to classify images
    # Returns
        A Keras model instance.
    """
    img_input = Input(shape=input_shape) if input_tensor is None else (
        Input(tensor=input_tensor, shape=input_shape) if not K.is_keras_tensor(input_tensor) else input_tensor)
    
    pickle_in = open("vgg16_face_conv_weights_name.pkl","rb")
    w_and_b = pickle.load(pickle_in)

    # Block 1
    x = conv_block(64, dropout=conv_dropout, activation=activation, block=1, layer=1,kernel_initializer= keras.initializers.Constant(w_and_b['conv1_1'][0]))(img_input)
#    x = conv_block(64, dropout=conv_dropout, activation=activation, block=1, layer=2,kernel_initializer= keras.initializers.Constant(w_and_b['conv1_2'][0]))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = conv_block(128, dropout=conv_dropout, activation=activation, block=2, layer=1,kernel_initializer= keras.initializers.Constant(w_and_b['conv2_1'][0]))(x)
#    x = conv_block(128, dropout=conv_dropout, activation=activation, block=2, layer=2,kernel_initializer= keras.initializers.Constant(w_and_b['conv2_2'][0]))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = conv_block(256, dropout=conv_dropout, activation=activation, block=3, layer=1,kernel_initializer= keras.initializers.Constant(w_and_b['conv3_1'][0]))(x)
#    x = conv_block(256, dropout=conv_dropout, activation=activation, block=3, layer=2,kernel_initializer= keras.initializers.Constant(w_and_b['conv3_2'][0]))(x)
#    x = conv_block(256, dropout=conv_dropout, activation=activation, block=3, layer=3,kernel_initializer= keras.initializers.Constant(w_and_b['conv3_3'][0]))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = conv_block(512, dropout=conv_dropout, activation=activation, block=4, layer=1,kernel_initializer= keras.initializers.Constant(w_and_b['conv4_1'][0]))(x)
#    x = conv_block(512, dropout=conv_dropout, activation=activation, block=4, layer=2,kernel_initializer= keras.initializers.Constant(w_and_b['conv4_2'][0]))(x)
#    x = conv_block(512, dropout=conv_dropout, activation=activation, block=4, layer=3,kernel_initializer= keras.initializers.Constant(w_and_b['conv4_3'][0]))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = conv_block(512, dropout=conv_dropout, activation=activation, block=5, layer=1,kernel_initializer= keras.initializers.Constant(w_and_b['conv5_1'][0]))(x)
#    x = conv_block(512, dropout=conv_dropout, activation=activation, block=5, layer=2,kernel_initializer= keras.initializers.Constant(w_and_b['conv5_2'][0]))(x)
#    x = conv_block(512, dropout=conv_dropout, activation=activation, block=5, layer=3,kernel_initializer= keras.initializers.Constant(w_and_b['conv5_3'][0]))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    
    train_model = models.Model(img_input,  x)
    return train_model

def VGGnet(input_shape, n_class):
    """
    
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    
    :return:  Keras Model used for training
    """
#    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    inputs_depth = layers.Input(shape=input_shape)
    pre_trained_model = VGG16_BN(input_tensor=None, input_shape=input_shape, conv_dropout=0.1, activation='relu')
    conv_model_depth = pre_trained_model(inputs_depth)
    
    flat_model = layers.Flatten(name='flatten')(conv_model_depth)
    fc6 = layers.Dense(2048, activation='relu', name='fc6')(flat_model)
    bn_1 = BatchNormalization(name='1_bn')(fc6)
    dropout_1 = layers.Dropout(0.2)(bn_1)
    
    
    
#    flatten_concat = layers.Flatten(name='flatten_concat')(merge_rgb_depth)
#    fc6 = layers.Dense(2048, activation='relu', name='fc6')(merge_rgb_depth)
    fc7 = layers.Dense(1024, activation='relu', name='fc7')(dropout_1)
    bn_2 = BatchNormalization(name='2_bn')(fc7)
    dropout_2 = layers.Dropout(0.2)(bn_2)
    
    #VECTORIZING OUTPUT
    output = layers.Dense(20, activation='softmax', name='output')(dropout_2)
    
    # MODAL [INPUTS , OUTPUTS]
    
    train_model = models.Model(inputs_depth,  output)
    train_model.summary()
    weights_path =  ('D:/tutorial/rgb+depth+thermal/allData/vgg_1_leyer_blk_allData_depth_bn/weights-22.h5')
    train_model.load_weights(weights_path)
    
    final_layer = train_model.layers[-2].output
    
    
    
#    fc6_final = layers.Dense(512, activation='relu', name='fc6_d')(final_layer)
#    dropout_1_final = layers.Dropout(0.2)(fc6_final)
#    
#    fc7_final = layers.Dense(512, activation='relu', name='fc7_final')(dropout_1_final)
#    dropout_2_final = layers.Dropout(0.2)(fc7_final)

#    
    output_final = layers.Dense(n_class, activation='softmax', name='output_final')(final_layer)

    
    
    new_train_model = models.Model(inputs_depth,  output_final)
    for layer in new_train_model.layers[:-8]:
        layer.trainable = False
#    for layer in new_train_model.layers[1].layers[:-10]:
#        layer.trainable = False



    return new_train_model


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
                  loss=['categorical_crossentropy'],
                  
                  metrics=['accuracy'])

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(batch_size):
        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)  
#        generator_rgb = train_datagen.flow_from_directory(directory="D:/RGB_D_Dataset/train/RGB/", target_size=(224, 224), color_mode="rgb",
#                                                      batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)
        generator_depth = train_datagen.flow_from_directory(directory="D:/CurtinFaces_crop/normalized/DEPTH/train/", target_size=(224, 224), color_mode="rgb",
                                                      batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)

        while 1:
#            x_batch_rgb, y_batch_rgb = generator_rgb.next()
            x_batch_depth, y_batch_depth = generator_depth.next()
            yield [x_batch_depth, y_batch_depth]
            
    def val_generator(batch_size):
        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)  
#        generator_rgb = train_datagen.flow_from_directory(directory="D:/RGB_D_Dataset/train/RGB/", target_size=(224, 224), color_mode="rgb",
#                                                      batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)
        generator_depth = train_datagen.flow_from_directory(directory="D:/CurtinFaces_crop/normalized/DEPTH/test1/", target_size=(224, 224), color_mode="rgb",
                                                      batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)

        while 1:
#            x_batch_rgb, y_batch_rgb = generator_rgb.next()
            x_batch_depth, y_batch_depth = generator_depth.next()
            yield [x_batch_depth, y_batch_depth]
    
#    DATA gen for all data
#    train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True, validation_split=0.2)
#
#    train_generator = train_datagen.flow_from_directory("D:/RGB_D_Dataset/train/depth/",target_size=(224, 224), color_mode="rgb",
#                                                        batch_size=args.batch_size, class_mode='categorical',subset='training')
#    
#    validation_generator = train_datagen.flow_from_directory("D:/RGB_D_Dataset/train/depth/",target_size=(224, 224), color_mode="rgb",
#                                                             batch_size=args.batch_size,class_mode='categorical',subset='validation')
    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(args.batch_size),
                        steps_per_epoch=int(936 / args.batch_size),
                        epochs=args.epochs,
                        validation_data= val_generator(args.batch_size),
                        validation_steps = int( 2028 / args.batch_size),
                        callbacks=[log, tb, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model




def test(model, args):

    
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
              loss=['categorical_crossentropy'],
              
              metrics=['accuracy'])
    model.load_weights('./allData/vgg_1_leyer_blk_allData_depth_bn/weights-22.h5')
#    
    
    def test_generator(batch_size=1):
        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)  
        generator_rgb = train_datagen.flow_from_directory(directory="D:/CurtinFaces_crop/RGB/test2/", target_size=(224, 224), color_mode="rgb",
                                                      batch_size=1, class_mode="categorical", shuffle=True, seed=42)
        generator_depth = train_datagen.flow_from_directory(directory="D:/CurtinFaces_crop/normalized/DEPTH/test2/", target_size=(224, 224), color_mode="rgb",
                                                      batch_size=1, class_mode="categorical", shuffle=True, seed=42)

        while 1:
            x_batch_rgb, y_batch_rgb = generator_rgb.next()
            x_batch_depth, y_batch_depth = generator_depth.next()
            yield [[x_batch_rgb, x_batch_depth], y_batch_rgb]
    
    scores = model.evaluate_generator(generator=test_generator(1),steps = 1560)###test1 2028 ###test2 1560##test3 260
    print('Test loss: {} ; Accuracy on Test: {}'.format(scores[0],scores[1]))


if __name__ == "__main__":
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="RGB-D network")
    parser.add_argument('--epochs', default=5000, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
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
    parser.add_argument('--save_dir', default='./CurtinFaces/result_vgg_allData_bn_20_classesPandora')
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
    model = VGGnet(input_shape=(100,100,1), n_class=52)
    model.summary()


#    if not args.testing:
    model_trained = train(model=model, args=args)
    test(model=model_trained, args=args)
  # as long as weights are given, will run testing
