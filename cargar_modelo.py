from keras.models import load_model
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

width = 300
height = 300
ruta_train = 'skin-disease-datasaet/train_set/'
ruta_predict = 'skin-disease-datasaet/test_set/FU-nail-fungus/_11_8230.jpg'

modelo = load_model('mimodelo.keras')

my_image = cv2.imread(ruta_predict)
my_image = cv2.resize(my_image, (width, height))

result = modelo.predict(np.array([my_image]))[0]
porcentaje = max(result)*100
grupo = labels[result.argmax()]
print(grupo, round(porcentaje))

image_path = ruta_predict

# Cargar la imagen usando PIL
imagen = Image.open(image_path)

# Mostrar la imagen utilizando Matplotlib
plt.imshow(imagen)
plt.axis('off')
plt.show()