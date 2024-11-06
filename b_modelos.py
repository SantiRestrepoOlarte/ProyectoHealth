import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import cv2
from matplotlib import pyplot as plt


# Cargar bases de datos procesadas
x_train = joblib.load('salidas\\x_train.pkl')
y_train = joblib.load('salidas\\y_train.pkl')
x_test = joblib.load('salidas\\x_test.pkl')
y_test = joblib.load('salidas\\y_test.pkl')


################################################
############## Preprocesamiento ################
################################################

# Escalar
x_train=x_train.astype('float32') # para poder escalarlo
x_test=x_test.astype('float32')   # para poder escalarlo

# Escalarlo para que quede entre 0 y 1
x_train /=255
x_test /=255

x_train.shape
x_test.shape

np.product(x_train[1].shape) # cantidad de variables por imagen

np.unique(y_train, return_counts=True)
np.unique(y_test, return_counts=True)

y_train.shape
y_test.shape     

# Convertir a 1d array
x_train2=x_train.reshape(5712,89100)
x_test2=x_test.reshape(1311,89100)
x_train2.shape
x_test2.shape

x_train2[1]

############################################################
################ Probar modelos de tradicionales############
############################################################

#################### Regresión logística ##########

lr=LogisticRegression()
lr.fit(x_train2, y_train)

pred_train_lr=lr.predict(x_train2)
print(metrics.classification_report(y_train, pred_train_lr))
metrics.roc_auc_score(y_train, pred_train_lr)

pred_test_lr=lr.predict(x_test2)
print(metrics.classification_report(y_test, pred_test_lr))
metrics.roc_auc_score(y_test, pred_test_lr)




#################### Random Forest ##########

rf=RandomForestClassifier()
rf.fit(x_train2, y_train)

pred_train_rf=rf.predict(x_train2)
print(metrics.classification_report(y_train, pred_train_rf))
metrics.roc_auc_score(y_train, pred_train_rf)

pred_test_rf=rf.predict(x_test2)
print(metrics.classification_report(y_test, pred_test_rf))
metrics.roc_auc_score(y_test, pred_test_rf)



#################### K Neighbors Classifier ##########

knc=KNeighborsClassifier()
knc.fit(x_train2, y_train)

pred_train_knc=knc.predict(x_train2)
print(metrics.classification_report(y_train, pred_train_knc))
metrics.roc_auc_score(y_train, pred_train_knc)

pred_test_knc=knc.predict(x_test2)
print(metrics.classification_report(y_test, pred_test_knc))
metrics.roc_auc_score(y_test, pred_test_knc)



############### Red Neuronal tradicional 1 ############

fc_model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

fc_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','AUC', 'Recall', 'Precision'])

fc_model.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))

pred_train=fc_model.predict(x_train).astype('int')
#print(metrics.classification_report(y_train, pred_train))
metrics.roc_auc_score(y_train, pred_train)

pred_test=fc_model.predict(x_test)
#print(metrics.classification_report(y_test, pred_test))
metrics.roc_auc_score(y_test, pred_test)

# Evaluar el modelo
test_loss, test_acc, test_auc, test_recall, test_precision = fc_model.evaluate(x_test, y_test, verbose=2)
print("Test auc:", test_auc)
print("Test recall:", test_recall)

# Matriz de confusión test
pred_test=(fc_model.predict(x_test) > 0.50).astype('int')
cm=metrics.confusion_matrix(y_test,pred_test, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Tumor', 'No tumor'])
disp.plot()

# Classification report train
print(metrics.classification_report(y_train, pred_train))

# Classification report test
print(metrics.classification_report(y_test, pred_test))




######## Red Neuronal tradicional con Dropout y Regularización #########

reg_strength = 0.001

dropout_rate = 0.2 ## porcentaje de neuronas que utiliza 

fc_model2=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

fc_model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

fc_model2.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))

test_loss, test_acc, test_auc, test_recall, test_precision = fc_model2.evaluate(x_test, y_test, verbose=2)

pred_train2=fc_model2.predict(x_train).astype('int')
#print(metrics.classification_report(y_train, pred_train))
metrics.roc_auc_score(y_train, pred_train2)

pred_test2=fc_model2.predict(x_test)
#print(metrics.classification_report(y_test, pred_test))
metrics.roc_auc_score(y_test, pred_test2)

# Matriz de confusión test
pred_test2=(fc_model2.predict(x_test) > 0.50).astype('int')
cm=metrics.confusion_matrix(y_test,pred_test2, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Tumor', 'No tumor'])
disp.plot()

# Classification report train
print(metrics.classification_report(y_train, pred_train2))

# Classification report test
print(metrics.classification_report(y_test, pred_test2))