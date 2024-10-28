import numpy as np
import os
import cv2
from os import listdir
from os.path import join
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

################################## Para cargar las imagenes del directorio ##################################

def img2data(path, resolucion=(100, 100)):
   
    rawImgs = []   # Lista para almacenar las imágenes en formato array
    labels = []    # Lista para almacenar las etiquetas (0 para 'notumor', 1 para 'tumor')
    file_list = [] 

    # Listar las carpetas dentro de la ruta (ej. 'tumor', 'notumor')
    list_labels = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

    for imagePath in list_labels:
        files_list = os.listdir(imagePath)  # Lista de archivos en cada carpeta ('tumor', 'notumor')

        for item in tqdm(files_list, desc=f"Procesando imágenes de {imagePath}"):
            file = os.path.join(imagePath, item)  

            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                img = cv2.imread(file)  # Cargar la imagen
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
                img = cv2.resize(img, resolucion)  # Reescala la imagen según la resolución 
                
                rawImgs.append(img)  # Agrega la imagen a la lista
                file_list.append(item)  

                # Obtener la etiqueta según el nombre de la carpeta
                label = imagePath.split(os.path.sep)[-1]  # Extrae el nombre de la carpeta ('tumor' o 'notumor')
                if label == 'notumor':
                    labels.append([0])  
                elif label == 'tumor':
                    labels.append([1])  
    
    return rawImgs, labels, file_list


################################## Para la imagen con menor resoluciòn ##################################

def imagen_menor_resolucion(imagenes):
    menor_resolucion = float('inf')
    resolucion_menor = (0, 0)
    imagen_menor = None

    # Listar todas las imágenes en el imagenes
    for clase in os.listdir(imagenes):
        clase_dir = os.path.join(imagenes, clase)
        for img_name in os.listdir(clase_dir):
            img_path = os.path.join(clase_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                altura, ancho, _ = img.shape
            
                # Calcular la resolución
                resolucion_actual = altura * ancho
                if resolucion_actual < menor_resolucion:
                    menor_resolucion = resolucion_actual
                    resolucion_menor = (ancho, altura)
                    imagen_menor = img_path

    return resolucion_menor, imagen_menor


################################### Reescalar todas las imagenes #####################################
################################### a la que menor resoluciòn tenga ##################################


def reescalar_imagenes(imagenes, nueva_resolucion):
    for clase in os.listdir(imagenes):
        clase_dir = os.path.join(imagenes, clase)
        for img_name in os.listdir(clase_dir):
            img_path = os.path.join(clase_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img_reescalada = cv2.resize(img, nueva_resolucion)
                cv2.imwrite(img_path, img_reescalada)