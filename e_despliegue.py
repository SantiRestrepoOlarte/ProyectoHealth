import numpy as np
import pandas as pd
import cv2
import funciones as fn
import tensorflow as tf
import openpyxl
import sys
sys.executable
sys.path

if __name__ =="main_":

    # Cargar datos de despliegue
    path = 'data/Despliegue/'
    x, _, files= fn.img2data(path)

    # Imagenes de despliegue en array
    x=np.array(x)

    # Disminuir resolución de imagenes
    new_size = (99, 75)
    x = np.array([cv2.resize(img, new_size, interpolation=cv2.INTER_AREA) for img in x])

    # Escalar
    x=x.astype('float')
    x/=255
    
    # Eliminar extension a nombre de archivo
    files2= [name.rsplit('.', 1)[0] for name in files] 

    # Cargar modelo
    modelo=tf.keras.models.load_model('salidas/best_model.keras') 
    prob=modelo.predict(x)

    # Clasificación
    clas=['Tumor' if prob > 0.6 else 'No tumor' for prob in prob]

    # Guardar resultados
    res_dict={
        "paciente": files2,
        "clas": clas   
    }
    resultados=pd.DataFrame(res_dict)

    # Exportar resultados a archivo excel
    resultados.to_excel('salidas/clasificaciones.xlsx', index=False)