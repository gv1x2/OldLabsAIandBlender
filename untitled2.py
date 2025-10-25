# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:00:02 2024

@author: Vik
"""

layers.Conv2D(32, kernel_size=(3, 3), activation='relu')  # Convolutional layer for feature extraction
layers.MaxPooling2D((2, 2))  # Pooling layer to reduce spatial dimensions
layers.Flatten()  # Flatten the 3D output to 1D before feeding it into the dense layer
layers.Dense(128, activation='relu')  # Dense layer for classification
layers.Dropout(0.5)  # Dropout for regularization to prevent overfitting
layers.BatchNormalization()  # Normalize the activations from the previous layer
layers.Dense(10, activation='softmax')  # Output layer for multi-class classification of digits 0-9


'relu'  # Commonly used in hidden layers for its efficiency
'softmax'  # Used in the output layer for multi-class classification



optimizers.Adam(learning_rate=0.001)  # Adam optimizer, a popular choice for its adaptiveness
optimizers.SGD(learning_rate=0.01, momentum=0.9)  # Stochastic Gradient Descent with momentum



regularizers.l2(0.01)  # L2 regularization can be added to any layer, commonly used in Dense layers
layers.Dropout(0.5)  # Dropout, as a layer, effectively prevents overfitting by randomly deactivating neurons during training
