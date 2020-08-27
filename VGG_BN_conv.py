# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 15:57:02 2019

@author: hardi
"""
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Activation, Dropout, BatchNormalization
import keras
from keras import backend as K
import pickle

from keras import layers, models, optimizers



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
    x = conv_block(64, dropout=conv_dropout, activation=activation, block=1, layer=2,kernel_initializer= keras.initializers.Constant(w_and_b['conv1_2'][0]))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = conv_block(128, dropout=conv_dropout, activation=activation, block=2, layer=1,kernel_initializer= keras.initializers.Constant(w_and_b['conv2_1'][0]))(x)
    x = conv_block(128, dropout=conv_dropout, activation=activation, block=2, layer=2,kernel_initializer= keras.initializers.Constant(w_and_b['conv2_2'][0]))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = conv_block(256, dropout=conv_dropout, activation=activation, block=3, layer=1,kernel_initializer= keras.initializers.Constant(w_and_b['conv3_1'][0]))(x)
    x = conv_block(256, dropout=conv_dropout, activation=activation, block=3, layer=2,kernel_initializer= keras.initializers.Constant(w_and_b['conv3_2'][0]))(x)
    x = conv_block(256, dropout=conv_dropout, activation=activation, block=3, layer=3,kernel_initializer= keras.initializers.Constant(w_and_b['conv3_3'][0]))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = conv_block(512, dropout=conv_dropout, activation=activation, block=4, layer=1,kernel_initializer= keras.initializers.Constant(w_and_b['conv4_1'][0]))(x)
    x = conv_block(512, dropout=conv_dropout, activation=activation, block=4, layer=2,kernel_initializer= keras.initializers.Constant(w_and_b['conv4_2'][0]))(x)
    x = conv_block(512, dropout=conv_dropout, activation=activation, block=4, layer=3,kernel_initializer= keras.initializers.Constant(w_and_b['conv4_3'][0]))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = conv_block(512, dropout=conv_dropout, activation=activation, block=5, layer=1,kernel_initializer= keras.initializers.Constant(w_and_b['conv5_1'][0]))(x)
    x = conv_block(512, dropout=conv_dropout, activation=activation, block=5, layer=2,kernel_initializer= keras.initializers.Constant(w_and_b['conv5_2'][0]))(x)
    x = conv_block(512, dropout=conv_dropout, activation=activation, block=5, layer=3,kernel_initializer= keras.initializers.Constant(w_and_b['conv5_3'][0]))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    
    train_model = models.Model(img_input,  x)


#    pre_trained_model = VGG16_BN(input_tensor=None, input_shape=input_shape, classes=1000, conv_dropout=0.1, dropout=0.3, activation='relu')
#    trained_model_out = pre_trained_model(input)
#    pre_trained_model.summary()
#        final_layer = pre_trained_model.layers[-3].output
#        x = Dense(n_class, activation='softmax', name='output')(final_layer)
#        x = BatchNormalization()(x)

 #    Models for training and evaluation (prediction)
#    train_model = models.Model(img_input,  output)
#    for layer in train_model.layers[:-6]:
#        layer.trainable = False





    return train_model