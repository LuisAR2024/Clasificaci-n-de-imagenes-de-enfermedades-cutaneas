import tensorflow as tf
from keras import layers, models
import os
import numpy as np
import cv2
import random

width = 300
height = 300
ruta_train = 'skin-disease-datasaet/train_set/'
ruta_predict = 'skin-disease-datasaet/test_set/FU-nail-fungus/_11_8230.jpg'

train_x = []
train_y = []

labels = os.listdir(ruta_train)

for i in os.listdir(ruta_train):
  for j in os.listdir(ruta_train + i):
    img = cv2.imread(ruta_train+i+'/'+j)
    resized_image = cv2.resize(img, (width, height))

    train_x.append(resized_image)

    for x,y in enumerate(labels):
      if y == i:
        array = np.zeros(len(labels))
        array[x] = 1
        train_y.append(array)

x_data = np.array(train_x)
y_data = np.array(train_y)

model = tf.keras.Sequential([
    layers.Conv2D(32, 3, 3, input_shape=(width, height, 3)),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(32, 3, 3),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(64, 3, 3),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dense(64),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(8),
    layers.Activation('sigmoid')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 100
model.fit(x_data, y_data, epochs = epochs)

models.save_model(model, 'mimodelo.keras')