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
"""collecting the input to train"""
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
                number.append(i)
            except:
                print("Error")

set1 = np.array(set1)
number = np.array(number)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(set1, number, test_size=0.2, random_state=42)

from keras.utils import to_categorical
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

"""training a cnn model"""
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

model=classifier.fit(X_train, y_train, batch_size=32, epochs=15, validation_data=(X_test, y_test))

"""plotting the models"""
import matplotlib.pyplot as plt

epochs=range(1,16)

plt.figure(0)
loss_train = model.history['loss']
loss_val = model.history['val_loss']
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('LOSS')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(1)
accuracy_train = model.history['accuracy']
accuracy_val = model.history['val_accuracy']
plt.plot(epochs, accuracy_train, 'g', label='Training accuracy')
plt.plot(epochs, accuracy_val, 'b', label='validation accuacy')
plt.title('ACCURACY')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""testing the model"""
import pandas as pd
set2=[]
y_path=pd.read_csv('/Users/karanrochlani/Desktop/Traffic-Sign-Classification/untitled folder 2/traffic sign recognition/Test.csv')

labels=y_path["ClassId"].values

imgs=y_path["Path"].values

for a in imgs:
    path=os.path.join('/Users/karanrochlani/Desktop/Traffic-Sign-Classification/untitled folder 2/traffic sign recognition/Test',a)
    try:
        image = Image.open(path)
        image = image.resize((30,30))
        image = np.array(image)
        set2.append(image)
    except:
        print("Error")

set2 = np.array(set2)

pred=classifier.predict_classes(set2)

from sklearn.metrics import accuracy_score
print(accuracy_score(labels, pred))

from sklearn.metrics import confusion_matrix
result=confusion_matrix(pred,labels)

"""output"""
classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)', 
            3:'Speed limit (50km/h)', 
            4:'Speed limit (60km/h)', 
            5:'Speed limit (70km/h)', 
            6:'Speed limit (80km/h)', 
            7:'End of speed limit (80km/h)', 
            8:'Speed limit (100km/h)', 
            9:'Speed limit (120km/h)', 
            10:'No passing', 
            11:'No passing veh over 3.5 tons', 
            12:'Right-of-way at intersection', 
            13:'Priority road', 
            14:'Yield', 
            15:'Stop', 
            16:'No vehicles', 
            17:'Veh > 3.5 tons prohibited', 
            18:'No entry', 
            19:'General caution', 
            20:'Dangerous curve left', 
            21:'Dangerous curve right', 
            22:'Double curve', 
            23:'Bumpy road', 
            24:'Slippery road', 
            25:'Road narrows on the right', 
            26:'Road work', 
            27:'Traffic signals', 
            28:'Pedestrians', 
            29:'Children crossing', 
            30:'Bicycles crossing', 
            31:'Beware of ice/snow',
            32:'Wild animals crossing', 
            33:'End speed + passing limits', 
            34:'Turn right ahead', 
            35:'Turn left ahead', 
            36:'Ahead only', 
            37:'Go straight or right', 
            38:'Go straight or left', 
            39:'Keep right', 
            40:'Keep left', 
            41:'Roundabout mandatory', 
            42:'End of no passing', 
            43:'End no passing veh > 3.5 tons' }

last_path=os.path.join('/Users/karanrochlani/Desktop/Traffic-Sign-Classification/untitled folder 2/traffic sign recognition/Train/26/00026_00011_00029.png')
image = Image.open(last_path)
image = image.resize((30,30))
image = np.expand_dims(image, axis=0)
image = np.array(image)
ans = classifier.predict_classes([image])[0]
result = classes[ans+1]
print(result)


