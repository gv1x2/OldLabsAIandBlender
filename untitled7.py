# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 08:55:02 2024

@author: Vik
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# # Înc?rcam setul de date CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# # Normalizam valorile pixelilor s? fie între 0 ?i 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Conversia vectorilor de clas? în matrice de clas? binar?
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Definim modelul CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    #layers.Flatten(),
    #layers.Dense(64, activation='relu'),
    #layers.MaxPooling2D((2, 2)),
    #layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilam modelul
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Rezumatul modelului
model.summary()

# Antrenarea modelului
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluarea modelul
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

# Salvarea modelul
model.save('cifar10_model.h5')
print("Model saved to 'cifar10_model.h5'.")
