# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 03:03:02 2019

@author: hardi
"""
import keras.backend as K
from keras.models import load_model
from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.models import Model
#from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import os
import pickle
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks


###params
saveDir = './result/autoencoder'
batch_size = 32
epochs = 1000
if not os.path.exists(saveDir):
    os.makedirs(saveDir)


#def autoencoder():
input_img = Input(shape=(224, 224, 3))
x = Conv2D(128, (3, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(16, (3, 3), padding='same')(encoded)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(3, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
decoded = Activation('sigmoid')(x)
    
    
encoder = Model(input_img, encoded)
encoder.compile(optimizer='adam', loss='binary_crossentropy')   


#decoder = Model(encoded, decoded)
#decoder.compile(optimizer='adam', loss='binary_crossentropy')   
def contractive_loss(y_pred, y_true):
    mse = K.mean(K.square(y_true - y_pred), axis=1)

    W = K.variable(value=model.get_layer('encoded').get_weights()[0])  # N x N_hidden     
    W = K.transpose(W)  # N_hidden x N     
    h = model.get_layer('encoded').output
    dh = h * (1 - h)  # N_batch x N_hidden 
    # N_batch x N_hidden * N_hidden x 1 = N_batch x 1     
    contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)

    return mse + contractive


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()


def train_generator(batch_size=batch_size):
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)  
    generator_train = train_datagen.flow_from_directory(directory="D:/EURECOM_Kinect_Face_Dataset/RGB/train", target_size=(224, 224), color_mode="rgb",
                                                  batch_size=batch_size, class_mode="categorical", shuffle=True,seed=42)

    while 1:
        x_batch, y_batch = generator_train.next()
        yield [x_batch, x_batch]

def val_generator(batch_size):
    val_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)  
    generator_val = val_datagen.flow_from_directory(directory="D:/EURECOM_Kinect_Face_Dataset/RGB/val", target_size=(224, 224), color_mode="rgb",
                                              batch_size=batch_size, class_mode="categorical", shuffle=True,seed=42)
    
    while 1:
        x_batch, y_batch = generator_val.next()
        yield [x_batch, x_batch]

es_cb = callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
chkpt = os.path.join(saveDir, 'AutoEncoder_Deep_weights_1.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5')
cp_cb = callbacks.ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
tb = callbacks.TensorBoard(log_dir=saveDir + '/tensorboard-logs', batch_size=batch_size)
log = callbacks.CSVLogger(saveDir + '/log.csv')




history = autoencoder.fit_generator(generator=train_generator(batch_size),
                                    steps_per_epoch=int( 520 / batch_size),
                                    epochs=epochs,
                                    verbose=1,
                                    validation_data=val_generator(batch_size),
                                    validation_steps = int( 208 / batch_size),
                                    callbacks=[es_cb, cp_cb,tb,log])
from utils import plot_log
plot_log(saveDir + '/log.csv', show=True)
#
def test_generator(batch_size=batch_size):
    test_datagen = ImageDataGenerator(rescale=1./255)  
    generator_test = test_datagen.flow_from_directory(directory="D:/EURECOM_Kinect_Face_Dataset/RGB/test", target_size=(224, 224), color_mode="rgb",
                                                  batch_size=1, class_mode="categorical", shuffle=True,seed=42)

    while 1:
        x_batch, y_batch = generator_test.next()
        yield [x_batch, y_batch]


test_eval = [] 
for x,y in test_generator():
    test_eval.append(autoencoder.evaluate(x,y,verbose=1))
    break
#    
#    
#    
############## metric for face verification    
#def compute_eer(fpr,tpr,thresholds):
#    """ Returns equal error rate (EER) and the corresponding threshold. """
#    fnr = 1-tpr
#    abs_diffs = np.abs(fpr - fnr)
#    min_index = np.argmin(abs_diffs)
#    eer = np.mean((fpr[min_index], fnr[min_index]))
#    return eer, thresholds[min_index]    
    
    
    
    
    
    
    
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
