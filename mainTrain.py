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

print(x_train.shape)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import normalize

# x_test = x_test/255
x_train = normalize(x_train, axis = 1)
x_test = normalize(x_test, axis = 1)
# x_train = x_train/255


#building our model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers import Dropout, Flatten, Dense


# Model Building

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = (64,64,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))


model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))


model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))    #categorical cross entropy softmax

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(x_train, y_train,
          batch_size = 16,          #2400/16 = 150
          verbose = 1, 
          epochs = 10, 
          validation_data = (x_test, y_test),
          shuffle = False)

model.save('BrainTumor10Epochs.h5')         #test_accuracy => 97.17% train_accuracy => 98.96%

