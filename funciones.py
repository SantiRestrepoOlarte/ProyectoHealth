import numpy as np
import os
import cv2
from os import listdir
from os.path import join
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def img2data(path, width=100):
    
    rawImgs = []   #### una lista con el array que representa cada imágen
    labels = [] ### el label de cada imágen
    
    list_labels = [path+f for f in listdir(path)] ### crea una lista de los archivos en la ruta 'tumor' o 'no tumor'

    for imagePath in list_labels: ### recorre cada carpeta de la ruta ingresada
        
        files_list=listdir(imagePath) ### crea una lista con todos los archivos
        for item in tqdm(files_list): ### le pone contador a la lista: tqdm
            file = join(imagePath, item) ## crea ruta del archivo
            if file[-1] =='g': ### verificar que se imágen extensión jpg o jpeg
                img = cv2.imread(file) ### cargar archivo
                img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB) ### invierte el orden de los colores en el array para usar el más estándar RGB
                img = cv2.resize(img ,(width,width)) ### cambia resolución de imágnenes
                rawImgs.append(img) ###adiciona imágen al array final
                l = imagePath.split('/')[-1] ### identificar en qué carpeta está
                if l == 'notumor':  ### verificar en qué carpeta está para asignar el label
                    labels.append([0]) # Etiqueta 0 para 'no tumor'
                elif l == 'tumor':
                    labels.append([1]) # Etiqueta 1 para 'tumor'
    return rawImgs, labels, files_list




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

def reescalar_imagenes(imagenes, nueva_resolucion):
    for clase in os.listdir(imagenes):
        clase_dir = os.path.join(imagenes, clase)
        for img_name in os.listdir(clase_dir):
            img_path = os.path.join(clase_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img_reescalada = cv2.resize(img, nueva_resolucion)
                cv2.imwrite(img_path, img_reescalada)  

