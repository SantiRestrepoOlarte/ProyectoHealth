import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import funciones as fn  
import joblib


############################### Exploración por imagen ####################################

img1=cv2.imread('data\\Testing\\tumor\\Te-gl_0010.jpg')
img2 = cv2.imread('data/Training/notumor/Tr-no_0010.jpg')

plt.imshow(img1)
plt.title('tumor')
plt.show()

plt.imshow(img2)
plt.title('notumor')
plt.show()

img2.shape ### tamaño de imágenes
img1.shape
img1.max() ### máximo valor de intensidad en un pixel
img1.min() ### mínimo valor de intensidad en un pixel

############################### Reescalar imagenes ########################################

# Para saber que imagen tiene menor resolución y proceder a reescalar las demás

imagenes_train = 'data/Training'  
imagenes_test = 'data/Testing'

# Buscar la imagen con menor resolución en ambos conjuntos
resolucion_menor_train, imagen_menor_train = fn.imagen_menor_resolucion(imagenes_train)
resolucion_menor_test, imagen_menor_test = fn.imagen_menor_resolucion(imagenes_test)

# Comparar las resoluciones menores encontradas
if resolucion_menor_train < resolucion_menor_test:
    resolucion_menor = resolucion_menor_train
    imagen_menor = imagen_menor_train
else:
    resolucion_menor = resolucion_menor_test
    imagen_menor = imagen_menor_test

print(f"Resolución menor encontrada: {resolucion_menor} en la imagen: {imagen_menor}")

# Reescalar todas las imágenes a la resolución de la imagen con menor resolución
fn.reescalar_imagenes(imagenes_train, resolucion_menor)
fn.reescalar_imagenes(imagenes_test, resolucion_menor)  

