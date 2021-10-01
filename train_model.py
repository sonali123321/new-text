from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
import h5py
img = image.load_img(r"E:\Machine_learning\OD exe\training\ng\cast_def_0_0.jpeg")
plt.imshow(img)

cv2.imread(r"E:\Machine_learning\OD exe\training\ng\cast_def_0_0.jpeg").shape

train=ImageDataGenerator(rescale=1/255)
validation=ImageDataGenerator(rescale=1/255)

train_dataset=train.flow_from_directory(r'E:\Machine_learning\OD exe\training',target_size=(200,200),batch_size=10,class_mode='binary')

validation_dataset=validation.flow_from_directory(r'E:\Machine_learning\OD exe\validation',target_size=(200,200),batch_size=10,class_mode='binary')

train_dataset.class_indices

train_dataset.classes

model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(200,200,3)),
                                  
                                  tf.keras.layers.MaxPool2D(2,2),
                                  tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                                  
                    
                                  tf.keras.layers.MaxPool2D(2,2),
                                  tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                  
                                  tf.keras.layers.MaxPool2D(2,2),
                                  tf.keras.layers.Flatten(),
                                  tf.keras.layers.Dense(512,activation='relu'),
                                  tf.keras.layers.Dense(1,activation='sigmoid')
                                  
    
])

model.compile(loss='binary_crossentropy',
              optimizer = RMSprop(learning_rate=0.001),
              metrics = ['accuracy'])

model_fit=model.fit(train_dataset,steps_per_epoch=10,
                   epochs=50,validation_data=validation_dataset)




model.save("new_model.h5")
