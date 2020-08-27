# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 03:03:02 2019

@author: hardi
"""
import keras
from keras.models import load_model

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Flatten
from keras.models import Model
#from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import os
import pickle
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks


###params
saveDir = './result/autoencoder_dual'
batch_size = 16
epochs = 1000
num_classes = 52
lr = 0.0001
if not os.path.exists(saveDir):
    os.makedirs(saveDir)


#def autoencoder():
pickle_in = open("vgg16_face_conv_weights_name.pkl","rb")
w_and_b = pickle.load(pickle_in)
    
    
input_img = Input(shape=(224, 224, 3))

from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, UpSampling2D, Input, concatenate

def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

def upsample_simple(filters, kernel_size, strides, padding):
    return UpSampling2D(strides)

def conv2d_block(
    inputs, 
    use_batch_norm=True, 
    dropout=0.3, 
    filters=16, 
    kernel_size=(3,3), 
    activation='relu', 
    kernel_initializer='he_normal', 
    padding='same'):
    
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding) (inputs)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = Dropout(dropout)(c)
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding) (c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    return c

def custom_unet(input_shape,num_classes=1, use_batch_norm=True, upsample_mode='deconv', use_dropout_on_upsampling=False,
                dropout=0.3, dropout_change_per_layer=0.0, filters=16, num_layers=4, output_activation='sigmoid'): # 'sigmoid' or 'softmax'
    
    if upsample_mode=='deconv':
        upsample=upsample_conv
    else:
        upsample=upsample_simple

    # Build U-Net model
    inputs = Input(input_shape)
    x = inputs   

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)
        down_layers.append(x)
        x = MaxPooling2D((2, 2)) (x)
        dropout += dropout_change_per_layer
        filters = filters*2 # double the number of filters with each layer

    x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for conv in reversed(down_layers):        
        filters //= 2 # decreasing number of filters with each layer 
        dropout -= dropout_change_per_layer
        x = upsample(filters, (2, 2), strides=(2, 2), padding='same') (x)
        x = concatenate([x, conv])
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)
    
    outputs = Conv2D(num_classes, (1, 1), activation=output_activation) (x)    
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
    
#################### encoder for classification    
#encoder = Model(input_img, encoder(input_img))
#encoder.compile(optimizer='adam', loss='binary_crossentropy')   
#encoder.summary()
#rand_dict=encoder.get_weights()
#decoder = Model(encoded, decoded)
#decoder.compile(optimizer='adam', loss='binary_crossentropy')   

autoencoder = Model(input_img, decoder(encoder(input_img)))
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()
#rand_dict = autoencoder.get_weights()
#
#
#Setting weights to model 
#layer_dict = dict([(layer.name, layer) for layer in autoencoder.layers])
#
#pickle_in = open("vgg16_face_conv_weights_name.pkl","rb")
#model_vars = pickle.load(pickle_in)
#
#num=1
#for block in range(1,6):
#    if block in (1,2):
#        for layer in range(1,3):
#            if (layer_dict['conv{}_{}'.format(block,layer)].get_weights()[0].shape == model_vars['conv{}_{}'.format(block,layer)].shape):
#                    layer_dict['conv{}_{}'.format(block,layer)].set_weights([model_vars['conv{}_{}'.format(block,layer)],model_vars['conv{}_{}'.format(block,layer)]])
#                    print('weights loaded in layer conv{}_{}'.format(block,layer))
#                    num+=1
#            else:
#                print('shape not same')
#    elif block in (3,4,5):
#        for layer in range(1,4):
#            if (layer_dict['conv{}_{}'.format(block,layer)].get_weights()[0].shape == model_vars['conv{}_{}'.format(block,layer)].shape):
#                    layer_dict['conv{}_{}'.format(block,layer)].set_weights([model_vars['conv{}_{}'.format(block,layer)],model_vars['conv{}_{}'.format(block,layer)]])
#                    print('weights loaded in layer conv{}_{}'.format(block,layer))
#                    num+=1
#            else:
#                print('shape not same')
#                
#


#autoencoder.save('autoencoder.h5')
#print("Saved model to disk")
#def fc(enco):
#    flat = Flatten()(enco)
#    den = Dense(128, activation='relu')(flat)
#    out = Dense(num_classes, activation='softmax')(den)
#    return out
#
#encode = encoder(input_img)
#full_model = Model(input_img,fc(encode))
#full_model.summary()
#
#### extract weights for encoder for 
#for l1,l2 in zip(full_model.layers[1].layers[:29],autoencoder.layers[1].layers[:29]):
#    l1.set_weights(l2.get_weights())
#
#full_model.compile(optimizer= Adam(lr=lr),
#                  loss=['categorical_crossentropy'],
#                  
#                  metrics=['accuracy'])

def train_generator(batch_size):
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)  
    generator_rgb = train_datagen.flow_from_directory(directory="D:/EURECOM_Kinect_Face_Dataset/RGB/train", target_size=(224, 224), color_mode="rgb",
                                                  batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)
    generator_depth = train_datagen.flow_from_directory(directory="D:/EURECOM_Kinect_Face_Dataset/depth/train", target_size=(224, 224), color_mode="rgb",
                                                  batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)

    while 1:
        x_batch_rgb, y_batch_rgb = generator_rgb.next()
        x_batch_depth, y_batch_depth = generator_depth.next()
        yield [x_batch_rgb, x_batch_depth]
        
def val_generator(batch_size):
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)  
    generator_rgb = train_datagen.flow_from_directory(directory="D:/EURECOM_Kinect_Face_Dataset/RGB/val", target_size=(224, 224), color_mode="rgb",
                                                  batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)
    generator_depth = train_datagen.flow_from_directory(directory="D:/EURECOM_Kinect_Face_Dataset/depth/val", target_size=(224, 224), color_mode="rgb",
                                                  batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)

    while 1:
        x_batch_rgb, y_batch_rgb = generator_rgb.next()
        x_batch_depth, y_batch_depth = generator_depth.next()
        yield [x_batch_rgb, x_batch_depth]

es_cb = callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
chkpt = os.path.join(saveDir,'AutoEncoder_Deep_weights_classifier.hdf5') 
#'AutoEncoder_Deep_weights_1.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5')
cp_cb = callbacks.ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
tb = callbacks.TensorBoard(log_dir=saveDir + '/tensorboard-logs', batch_size=batch_size)
log = callbacks.CSVLogger(saveDir + '/log.csv')

#for layer in full_model.layers[1].layers[:-1]:
#    layer.trainable = False


history = autoencoder.fit_generator(generator=train_generator(batch_size),
                                    steps_per_epoch=int( 520 / batch_size),
                                    epochs=epochs,
                                    verbose=1,
                                    validation_data=val_generator(batch_size),
                                    validation_steps = int( 208 / batch_size),callbacks=[es_cb, tb, log])

autoencoder.save_weights(saveDir + '/trained_model.h5')
from utils import plot_log
plot_log(saveDir + '/log.csv', show=True)
#

#test_eval = [] 
#def test_generator(batch_size=batch_size):
#    test_datagen = ImageDataGenerator(rescale=1./255)  
#    generator_test = test_datagen.flow_from_directory(directory="D:/EURECOM_Kinect_Face_Dataset/RGB/test", target_size=(224, 224), color_mode="rgb",
#                                                  batch_size=1, class_mode="categorical", shuffle=True,seed=42)
#    generator_test.samples
#    i=0
#    while i<generator_test.samples:
#        x_batch, y_batch = generator_test.next()
#        test_eval.append(full_model.evaluate(x_batch,y_batch,verbose=1))
#        i=i+1
#        
#total=0
#for i in range(0,208):
#    total = total+test_eval[i][1]
#total/208
##    
#    
#    
#    
    
    
    
    
    
    
    
    
    
#    
#    
#    
#    
#    
#import h5py
##f = h5py.File('resultAutoEncoder_Deep_weights.88-0.48-0.48.hdf5', 'r')
#filename = 'resultAutoEncoder_Deep_weights.88-0.48-0.48.hdf5'
#
#with h5py.File(filename, 'r') as f:
#    # List all groups
#    print("Keys: %s" % f.keys())
#    a_group_key = list(f.keys())[0]
#
#    # Get the data
#    layers = list(f[a_group_key])
#    
#    data = list(f[a_group_key][layers])
#    
#dict_new = read_hdf5(filename)    
#    
#def read_hdf5(path):
#
#    weights = {}
#
#    keys = []
#    with h5py.File(path, 'r') as f: # open file
#        f.visit(keys.append) # append all keys to list
#        for key in keys:
#            if ':' in key: # contains data if ':' in key
#                print(f[key].name)
#                weights[f[key].name] = f[key].value
#    return weights
