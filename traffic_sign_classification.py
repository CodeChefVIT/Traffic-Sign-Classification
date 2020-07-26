#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: karanrochlani
"""

from PIL import Image
import os
import numpy as np

set1=[]
number=[]

for i in range(0,43):
    s=str(i)
    cur_path=os.path.join('/Users/karanrochlani/Desktop/Traffic-Sign-Classification/untitled folder 2/traffic sign recognition/Train',s)
    images = os.listdir(cur_path)
    for a in images:
            try:
                path=os.path.join(cur_path,a)
                image = Image.open(path)
                image = image.resize((30,30))
                image = np.array(image)
                set1.append(image)
                number.append(1)
            except:
                print("Error")

set1 = np.array(set1)
number = np.array(number)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(set1, number, test_size=0.2, random_state=42)

from keras.utils import to_categorical
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()
classifier.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
classifier.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))


classifier.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
classifier.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))


classifier.add(Flatten())

classifier.add(Dense(256, activation='relu'))

classifier.add(Dense(43, activation='sigmoid'))

classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=32, epochs=2, validation_data=(X_test, y_test))