import numpy as np
import joblib
import tensorflow as tf
from sklearn import metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

# Cargar bases procesadas
x_train = joblib.load('salidas\\x_train.pkl')
y_train = joblib.load('salidas\\y_train.pkl')
x_test = joblib.load('salidas\\x_test.pkl')
y_test = joblib.load('salidas\\y_test.pkl')


################################################
############## Preprocesamiento ################
################################################

# Disminuir resolución
new_size = (99, 75)
x_train = np.array([cv2.resize(img, new_size, interpolation=cv2.INTER_AREA) for img in x_train])
x_test = np.array([cv2.resize(img, new_size, interpolation=cv2.INTER_AREA) for img in x_test])

# Escalar
x_train=x_train.astype('float32') ## para poder escalarlo
x_test=x_test.astype('float32') 

x_train /=255
x_test /=255


# Cargar modelo
modelo=tf.keras.models.load_model('salidas\\best_model.keras')

######################################################################################################

# Desempeño en evaluación para tumor
prob=modelo.predict(x_test)
sns.histplot(prob, legend=False)
plt.title("Probabilidades imágenes en evaluación") # conocer el comportamiento de las probabilidades para revisar threshold

# Definir el threshold para tumor
threshold_tumor=0.6

# Clasification report en evaluación
pred_test=(modelo.predict(x_test)>=threshold_tumor).astype('int')
print(metrics.classification_report(y_test, pred_test))

# Matriz de confusión en evaluación
cm=metrics.confusion_matrix(y_test,pred_test, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Tumor', 'No tumor'])
disp.plot()

#######################################################################################################

# Desempeño en evaluación para no tumor
prob=modelo.predict(x_test)
sns.histplot(prob, legend=False)
plt.title("Probabilidades imágenes en evaluación") # conocer el comportamiento de las probabilidades para revisar threshold

# Definir el threshold para no tumor
threshold_no_tumor=0.4

# Clasification report en entrenamiento
pred_test=(prob>=threshold_no_tumor).astype('int')
print(metrics.classification_report(y_test, pred_test))

# Matriz de confusión entrenamiento
cm=metrics.confusion_matrix(y_test,pred_test, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Tumor', 'No tumor'])
disp.plot()


####### Clasificación final del modelo ################

prob=modelo.predict(x_test)

clas=['Tumor' if prob > 0.6 else 'No tumor' for prob in prob]

clases, count =np.unique(clas, return_counts=True)

count*100/np.sum(count)