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
saveDir = './result/full_model_dual'
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

def encoder(input_img,weights_dict=w_and_b):
#ENCODER
###BLOCK 1
    
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv1_1'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv1_1'][1]),name='conv1_1')(input_img)
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv1_2'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv1_2'][1]),name='conv1_2')(x)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu', name = 'act1')(x)
    x = MaxPooling2D((2, 2), padding='same',name='pool1')(x)
    ###BLOCK 2
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv2_1'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv2_1'][1]),name='conv2_1')(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv2_2'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv2_2'][1]),name='conv2_2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Activation('relu',name = 'act2')(x)
    x = MaxPooling2D((2, 2), padding='same',name='pool2')(x)
    ###BLOCK 3
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv3_1'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv3_1'][1]),name='conv3_1')(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv3_2'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv3_2'][1]),name='conv3_2')(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv3_3'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv3_3'][1]),name='conv3_3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = Activation('relu',name = 'act3')(x)
    x = MaxPooling2D((2, 2), padding='same',name='pool3')(x)
    ###BLOCK 4
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv4_1'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv4_1'][1]),name='conv4_1')(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv4_2'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv4_2'][1]),name='conv4_2')(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv4_3'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv4_3'][1]),name='conv4_3')(x)
    x = BatchNormalization(name='bn4')(x)
    x = Activation('relu',name = 'act4')(x)
    x = MaxPooling2D((2, 2), padding='same',name='pool4')(x)
    ###BLOCK 5
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv5_1'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv5_1'][1]),name='conv5_1')(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv5_2'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv5_2'][1]),name='conv5_2')(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer= keras.initializers.Constant(w_and_b['conv5_3'][0]),bias_initializer= keras.initializers.Constant(w_and_b['conv5_3'][1]),name='conv5_3')(x)
    x = BatchNormalization(name='bn5')(x)
    x = Activation('relu',name = 'act5')(x)
    encoded = MaxPooling2D((2, 2), padding='same',name='pool5')(x)
    return encoded

def decoder(encoded):
####DECODER
###BLOCK 1
    x = Conv2D(512, (3, 3), padding='same',name='de_conv1_1')(encoded)
    x = Conv2D(512, (3, 3), padding='same',name='de_conv1_2')(x)
    x = Conv2D(512, (3, 3), padding='same',name='de_conv1_3')(x)
    x = BatchNormalization(name='bn6')(x)
    x = Activation('relu',name = 'act6')(x)
    x = UpSampling2D((2, 2),name='uppool1')(x)
    ###BLOCK 2
    x = Conv2D(512, (3, 3), padding='same',name='de_conv2_1')(x)
    x = Conv2D(512, (3, 3), padding='same',name='de_conv2_2')(x)
    x = Conv2D(512, (3, 3), padding='same',name='de_conv2_3')(x)
    x = BatchNormalization(name='bn7')(x)
    x = Activation('relu',name = 'act7')(x)
    x = UpSampling2D((2, 2),name='uppool2')(x)
    ###BLOCK 3
    x = Conv2D(256, (3, 3), padding='same',name='de_conv3_1')(x)
    x = Conv2D(256, (3, 3), padding='same',name='de_conv3_2')(x)
    x = Conv2D(256, (3, 3), padding='same',name='de_conv3_3')(x)
    x = BatchNormalization(name='bn8')(x)
    x = Activation('relu',name = 'act8')(x)
    x = UpSampling2D((2, 2),name='uppool3')(x)
    ###BLOCK 4
    x = Conv2D(128, (3, 3), padding='same',name='de_conv4_1')(x)
    x = Conv2D(128, (3, 3), padding='same',name='de_conv4_2')(x)
    x = BatchNormalization(name='bn9')(x)
    x = Activation('relu',name = 'act9')(x)
    x = UpSampling2D((2, 2),name='uppool4')(x)
    ###BLOCK 5
    x = Conv2D(64, (3, 3), padding='same',name='de_conv5_1')(x)
    x = Conv2D(64, (3, 3), padding='same',name='de_conv5_2')(x)
    x = BatchNormalization(name='bn10')(x)
    x = Activation('relu',name = 'act10')(x)
    x = UpSampling2D((2, 2),name='uppool5')(x)
    ###BLOCK 6
    x = Conv2D(3, (3, 3), padding='same',name='de_conv6_1')(x)
    x = BatchNormalization(name='bn11')(x)
    decoded = Activation('sigmoid',name = 'act11')(x)
    return decoded
    
#################### encoder for classification    
encoder = Model(input_img, encoder(input_img))
encoder.compile(optimizer='adam', loss='binary_crossentropy')   
encoder.summary()
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
def fc(enco):
    flat = Flatten()(enco)
    den = Dense(128, activation='relu')(flat)
    out = Dense(num_classes, activation='softmax')(den)
    return out
#
encode = encoder(input_img)
full_model = Model(input_img,fc(encode))
full_model.summary()
#
#### extract weights for encoder for 
for l1,l2 in zip(full_model.layers[1].layers[:29],autoencoder.layers[:29]):
    l1.set_weights(l2.get_weights())
#
for layer in full_model.layers[:-3]:
    layer.trainable = False
    
full_model.compile(optimizer= Adam(lr=lr),
                  loss=['categorical_crossentropy'],
                  
                  metrics=['accuracy'])



def train_generator(batch_size):
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)  
    generator_rgb = train_datagen.flow_from_directory(directory="D:/EURECOM_Kinect_Face_Dataset/RGB/train", target_size=(224, 224), color_mode="rgb",
                                                  batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)
    generator_depth = train_datagen.flow_from_directory(directory="D:/EURECOM_Kinect_Face_Dataset/depth/train", target_size=(224, 224), color_mode="rgb",
                                                  batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)

    while 1:
        x_batch_rgb, y_batch_rgb = generator_rgb.next()
        x_batch_depth, y_batch_depth = generator_depth.next()
        yield [x_batch_rgb, y_batch_rgb]
#        yield [x_batch_rgb, x_batch_depth]## for autoencoder
        
def val_generator(batch_size):
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)  
    generator_rgb = train_datagen.flow_from_directory(directory="D:/EURECOM_Kinect_Face_Dataset/RGB/val", target_size=(224, 224), color_mode="rgb",
                                                  batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)
    generator_depth = train_datagen.flow_from_directory(directory="D:/EURECOM_Kinect_Face_Dataset/depth/val", target_size=(224, 224), color_mode="rgb",
                                                  batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)

    while 1:
        x_batch_rgb, y_batch_rgb = generator_rgb.next()
        x_batch_depth, y_batch_depth = generator_depth.next()
        yield [x_batch_rgb, y_batch_rgb]
#        yield [x_batch_rgb, x_batch_depth] ## for autoencoder

es_cb = callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
chkpt = os.path.join(saveDir,'AutoEncoder_Deep_weights_classifier.hdf5') 
#'AutoEncoder_Deep_weights_1.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5')
cp_cb = callbacks.ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
tb = callbacks.TensorBoard(log_dir=saveDir + '/tensorboard-logs_1', batch_size=batch_size)
log = callbacks.CSVLogger(saveDir + '/log_1.csv')

#for layer in full_model.layers[1].layers[:-1]:
#    layer.trainable = False


history = full_model.fit_generator(generator=train_generator(batch_size),
                                    steps_per_epoch=int( 520 / batch_size),
                                    epochs=epochs,
                                    verbose=1,
                                    validation_data=val_generator(batch_size),
                                    validation_steps = int( 208 / batch_size),callbacks=[es_cb, tb, log])

#
#
#history = autoencoder.fit_generator(generator=train_generator(batch_size),
#                                    steps_per_epoch=int( 520 / batch_size),
#                                    epochs=epochs,
#                                    verbose=1,
#                                    validation_data=val_generator(batch_size),
#                                    validation_steps = int( 208 / batch_size),callbacks=[es_cb, tb, log])

#autoencoder.save_weights(saveDir + '/trained_model.h5')
full_model.save_weights(saveDir + '/trained_model_1.h5')

from utils import plot_log
plot_log(saveDir + '/log_1.csv', show=True)
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
