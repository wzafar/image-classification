# Caltech 256 Image Classification
# Link to dataset: https://www.kaggle.com/jessicali9530/caltech256


# Step 1 - Importing the necessary libraries

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from PIL import Image
import tensorflow as tf
from os import listdir
import itertools
import random
import seaborn as sns
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from sklearn.metrics import confusion_matrix
import scipy
from scipy import misc


# Step 2 - Data Loading 

# Defining a function to load the images (Loading Function)
def loadImages(path, nTrain, nTest):
    X_train = []    # List of Training Images
    X_test = []     # List of Test Images 
    y_train = []    # List of Classes/Categories of Training Images 
    y_test = []     # List of Classes/Categories of Test Images
    cat_list = os.listdir(path)
    cat_list.remove('.DS_Store')
    for cat in cat_list[0:nClasses]:
        deep_path = path + cat + '/'
        image_list = os.listdir(deep_path)
        for image in image_list[0:nTrain]:
            img = load_img(deep_path + image)    
            img = misc.imresize(img, (width,height))
            y_train.append(int(image[0:3])-1)
            X_train.append(img)
        for image in image_list[nTrain:nTrain + nTest]:
            img = load_img(deep_path + image)
            img = misc.imresize(img, (width,height))
            y_test.append(int(image[0:3])-1)
            X_test.append(img)
    return X_train, X_test, tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_test) 
# tf.keras.utils.to_categorical converts a class vector (integers) to binary class matrix. 
# For use with categorical_crossentropy.

path = 'Dataset/256_ObjectCategories/'
nTrain = 20
nTest = 5
width = 128
height = 128
nClasses = 257

X_train, X_test, y_train, y_test = loadImages(path, nTrain, nTest)


# Step 3 - Bulding the CNN

# Initialising the CNN
classifier = Sequential()

# Adding a Convolutional Layer
classifier.add(Conv2D(64, (3, 3), input_shape = (width, height, 3), activation = 'relu'))

# Adding a Pooling Layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a third convolutional layer
classifier.add(Conv2D(256, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

# Adding two densely connected layers
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dense(units = 512, activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 257, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Step 4 - Fitting the CNN to the Images

classifier.fit(X_train, y_train, epochs = 25, validation_data = (X_test, y_test))


# Step 5 - Evaluating the CNN 

