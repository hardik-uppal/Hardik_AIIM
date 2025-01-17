# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 04:32:08 2019

@author: hardi
"""

import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from model_2_mutimodal_attention_CurtinFaces_dataaug import VGGFace_multimodal ## our model
from keract import get_activations,display_activations
from model_vgg_face import VGG16 ##VGG model
import os
import pandas as pd

################## all activation per layer accumulated in a list
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
#def preprocess_image(image_path):
#    img = load_img(image_path, target_size=(224, 224))
#    img = img_to_array(img)
#    img = np.expand_dims(img, axis=0)
#    img = preprocess_input(img)
#    return img
#

 #### multimodal attention   
model_vgg_multimodal = VGGFace_multimodal(input_shape=(224,224,3), n_class=106)
#model_vgg_multimodal.load_weights('D:/tutorial/rgb+depth+thermal/CurtinFaces/dataaug_vgg_multimodal_dropout-0.5_3fc_batch30/weights-best.h5')
model_vgg_multimodal.load_weights('D:/tutorial/rgb+depth+thermal/IIIT_D/icip_test/weights-best.h5')
model_vgg_multimodal.compile(optimizer=optimizers.Adam(lr=0.01), loss=['categorical_crossentropy'], metrics=['accuracy'])
model_vgg_multimodal.summary()
### vgg rgb modal
model_vgg_rgb = VGG16(include_top=True, weights='vggface', input_tensor=None, input_shape=(224,224,3), pooling=None,classes=2622, type_name='rgb')

model_vgg_rgb.compile(optimizer=optimizers.Adam(lr=0.01), loss=['categorical_crossentropy'], metrics=['accuracy'])
model_vgg_rgb.summary()



rgb_val_dir = 'D:/RGB_D_Dataset_new/fold1/test/RGB/'         #D:/CurtinFaces_crop/RGB/test1/  #D:/CurtinFaces_processed/protocol/rgb/test1/
depth_val_dir = 'D:/RGB_D_Dataset_new/fold1/test/depth/'     #D:/CurtinFaces_crop/normalized/DEPTH/test1/  #D:/CurtinFaces_processed/protocol/depth/test1/


all_image_list =[]
subject_list = []
layer_name = 'flatten'

def test_generator_multimodal(batch_size=1):
    train_datagen = ImageDataGenerator(rescale=1./255)  
    generator_rgb = train_datagen.flow_from_directory(directory=rgb_val_dir, target_size=(224, 224), color_mode="rgb",
                                                  batch_size=1, class_mode="categorical", shuffle=True, seed=42)
    generator_depth = train_datagen.flow_from_directory(directory=depth_val_dir, target_size=(224, 224), color_mode="rgb",
                                                  batch_size=1, class_mode="categorical", shuffle=True, seed=42)
    i=0
    while i < 1560:
        x_batch_rgb, y_batch_rgb = generator_rgb.next()
        x_batch_depth, y_batch_depth = generator_depth.next()
        i=i+1
        yield [x_batch_rgb, x_batch_depth,y_batch_rgb]
def test_generator_rgb(batch_size=1):
    train_datagen = ImageDataGenerator(rescale=1./255)  
    generator_rgb = train_datagen.flow_from_directory(directory=rgb_val_dir, target_size=(224, 224), color_mode="rgb",
                                                  batch_size=1, class_mode="categorical", shuffle=True, seed=42)
    i=0
    while i < 1560:
        x_batch_rgb, y_batch_rgb = generator_rgb.next()
        i=i+1
        yield [x_batch_rgb,y_batch_rgb]
        
        
def test_generator_depth(batch_size=1):
    train_datagen = ImageDataGenerator(rescale=1./255)  
    generator_depth = train_datagen.flow_from_directory(directory=depth_val_dir, target_size=(224, 224), color_mode="rgb",
                                                  batch_size=1, class_mode="categorical", shuffle=True, seed=42)
    i=0
    while i < 1560:
        x_batch_depth, y_batch_depth = generator_depth.next()
        i=i+1
        yield [x_batch_depth,y_batch_depth]
        
        
##### for multimodal        
for image_embedding_rgb,image_embedding_depth,subject in test_generator_multimodal():
#    activations = {}
#    img_abs_path = os.path.join(image_dir_rgb, image)
#    image_embedding = preprocess_image(img_abs_path)
#    print(image_embedding)
#    break
    arr = np.array(list(get_activations(model_vgg_multimodal, [image_embedding_rgb,image_embedding_depth],layer_name).values())[0])
#    arr_depth = np.array(list(get_activations(model_vgg_multimodal, image_embedding_depth,layer_name).values())[0])
            
    all_image_list.append(arr)
    subject_list.append(subject)

all_image_list = np.mean(np.array(all_image_list),axis=1)

subject_list = np.mean(np.array(subject_list),axis=1)
subject_list_label = np.where(subject_list==1)[1]
#calculate tsne embeddings
X_tsne = TSNE(n_components=2).fit_transform(all_image_list)


dataset_x_tsne = pd.DataFrame({'t-SNE_Dim1': X_tsne[:, 0], 't-SNE_Dim2': X_tsne[:, 1],'Class':subject_list_label})
dataset_x_tsne_10_classes = dataset_x_tsne.loc[dataset_x_tsne['Class'] < 10 ]
#sns.palplot(sns.color_palette('hls',n_colors = 10))
plt.figure(figsize=(6,6))
multimodal_plot = sns.scatterplot(
    x="t-SNE_Dim1", y="t-SNE_Dim2",
    hue="Class",
    palette=sns.color_palette(n_colors = 10),
    data=dataset_x_tsne_10_classes,
    legend="full",
    alpha=0.3
)  
multimodal_plot.get_figure().savefig('IIIT_multimodal_out.pdf')


########### for VGG RGB model
all_image_list_rgb =[]
subject_list_rgb = []

for image_embedding_rgb,subject in test_generator_rgb():
#    activations = {}
#    img_abs_path = os.path.join(image_dir_rgb, image)
#    image_embedding = preprocess_image(img_abs_path)
#    print(image_embedding)
#    break
    arr = np.array(list(get_activations(model_vgg_rgb, image_embedding_rgb,layer_name).values())[0])
#    arr_depth = np.array(list(get_activations(model_vgg_multimodal, image_embedding_depth,layer_name).values())[0])
            
    all_image_list_rgb.append(arr)
    subject_list_rgb.append(subject)

all_image_list_rgb = np.mean(np.array(all_image_list_rgb),axis=1)

subject_list_rgb = np.mean(np.array(subject_list_rgb),axis=1)
subject_list_label_rgb = np.where(subject_list_rgb==1)[1]
#calculate tsne embeddings
X_tsne_rgb = TSNE(n_components=2).fit_transform(all_image_list_rgb)


dataset_x_tsne = pd.DataFrame({'t-SNE_Dim1': X_tsne_rgb[:, 0], 't-SNE_Dim2': X_tsne_rgb[:, 1],'Class':subject_list_label_rgb})
dataset_x_tsne_10_classes = dataset_x_tsne.loc[dataset_x_tsne['Class'] < 10 ]
#sns.palplot(sns.color_palette('hls',n_colors = 10))
plt.figure(figsize=(6,6))
multimodal_plot = sns.scatterplot(
    x="t-SNE_Dim1", y="t-SNE_Dim2",
    hue="Class",
    palette=sns.color_palette(n_colors = 10),
    data=dataset_x_tsne_10_classes,
    legend="full",
    alpha=0.3
)  
multimodal_plot.get_figure().savefig('iiit_rgb_out.pdf')

########### for VGG depth model
all_image_list_depth =[]
subject_list_depth = []

for image_embedding_depth,subject in test_generator_depth():
#    activations = {}
#    img_abs_path = os.path.join(image_dir_rgb, image)
#    image_embedding = preprocess_image(img_abs_path)
#    print(image_embedding)
#    break
    arr = np.array(list(get_activations(model_vgg_rgb, image_embedding_depth,layer_name).values())[0])
#    arr_depth = np.array(list(get_activations(model_vgg_multimodal, image_embedding_depth,layer_name).values())[0])
            
    all_image_list_depth.append(arr)
    subject_list_depth.append(subject)

all_image_list_depth = np.mean(np.array(all_image_list_depth),axis=1)

subject_list_depth = np.mean(np.array(subject_list_depth),axis=1)
subject_list_label_depth = np.where(subject_list_depth==1)[1]
#calculate tsne embeddings
X_tsne_depth = TSNE(n_components=2).fit_transform(all_image_list_depth)


dataset_x_tsne = pd.DataFrame({'t-SNE_Dim1': X_tsne_depth[:, 0], 't-SNE_Dim2': X_tsne_depth[:, 1],'Class':subject_list_label_depth})
dataset_x_tsne_10_classes = dataset_x_tsne.loc[dataset_x_tsne['Class'] < 10 ]
#sns.palplot(sns.color_palette('hls',n_colors = 10))
plt.figure(figsize=(6,6))
multimodal_plot = sns.scatterplot(
    x="t-SNE_Dim1", y="t-SNE_Dim2",
    hue="Class",
    palette=sns.color_palette(n_colors = 10),
    data=dataset_x_tsne_10_classes,
    legend="full",
    alpha=0.3
)  
multimodal_plot.get_figure().savefig('iiit_depth_out.pdf')



