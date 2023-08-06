import cv2
import os
from PIL import Image
import numpy as np

image_directory='./datasets/'

no_tumor_images = os.listdir(image_directory+'no/')     #lists all images in no directory
yes_tumor_images = os.listdir(image_directory+'yes/')
dataset = []
label = []

# print(no_tumor_images)

path = 'no0.jpg'

# print(path.split('.'))      #['no0', 'jpg']
path.split('.')[1]

for i, image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image = cv2.imread(image_directory+'no/'+image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64,64))
        dataset.append(np.array(image))
        label.append(0)
        
for i, image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image = cv2.imread(image_directory+'yes/'+image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64,64))
        dataset.append(np.array(image))
        label.append(1)
        
        
print(len(label))
print(len(dataset))

#converting dataset into numpy array:

dataset = np.array(dataset)
label = np.array(label)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.2, random_state= 0)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import normalize

# x_test = x_test/255
x_train = normalize(x_train, axis = 1)
x_test = normalize(x_test, axis = 1)
# x_train = x_train/255

from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense, Flatten
from keras.models import Model

vgg = VGG16(input_shape = (64,64,3),weights = 'imagenet',include_top = False)

for layer in vgg.layers:
    layer.trainable = False
    
x = Flatten()(vgg.output)
prediction = Dense(1,activation = 'softmax')(x)     #one final classes
model = Model(inputs = vgg.input,outputs = prediction)

model.summary()

from tensorflow.keras.optimizers import RMSprop
model.compile(loss = 'binary_crossentropy', optimizer = RMSprop(lr=0.0001), metrics = ['accuracy'])

model.fit(x_train, y_train,
          batch_size = 16,          #2400/16 = 150
          epochs = 2, 
          validation_data = (x_test, y_test),
          shuffle = False)

model.save('BrainTumor10EpochsTransferL.h5') 