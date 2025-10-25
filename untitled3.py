# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:01:05 2024

@author: Vik
"""

layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
layers.MaxPooling2D((2, 2))  ,
layers.Flatten()
layers.Dense(128, activation='relu')
layers.Dropout(0.5)
layers.BatchNormalization()
layers.Dense(10, activation='softmax')


'relu' 
'softmax'



optimizers.Adam(learning_rate=0.001)
optimizers.SGD(learning_rate=0.01, momentum=0.9)



regularizers.l2(0.01) 
layers.Dropout(0.5)
