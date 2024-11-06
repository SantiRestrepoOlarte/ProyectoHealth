import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn import metrics
import keras_tuner as kt
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from matplotlib import pyplot
import cv2


# Cargar bases procesadas
x_train = joblib.load('salidas\\x_train.pkl')
y_train = joblib.load('salidas\\y_train.pkl')
x_test = joblib.load('salidas\\x_test.pkl')
y_test = joblib.load('salidas\\y_test.pkl')

x_train[0]

################################################
############## Preprocesamiento ################
################################################

# Disminuir resolución
new_size = (99, 75)
x_train = np.array([cv2.resize(img, new_size, interpolation=cv2.INTER_AREA) for img in x_train])
x_test = np.array([cv2.resize(img, new_size, interpolation=cv2.INTER_AREA) for img in x_test])

x_train[4]

# Escalar
x_train=x_train.astype('float32') # para poder escalarlo
x_test=x_test.astype('float32')   # para poder escalarlo
x_train.max()
x_train.min()

x_train /=255 # escalarlo para que quede entre 0 y 1, con base en el valor máximo
x_test /=255

x_train.shape
x_test.shape

np.product(x_train[1].shape) # cantidad de variables por imagen

np.unique(y_train, return_counts=True)
np.unique(y_test, return_counts=True)


##########################################################
############### Red Neuronal Convolucional ###############
##########################################################

reg_strength = 0.001

dropout_rate = 0.1  

cnn_model2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

cnn_model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC','recall'])

cnn_model2.fit(x_train, y_train, batch_size=100, epochs=5, validation_data=(x_test, y_test))

pred_train=cnn_model2.predict(x_train)
#print(metrics.classification_report(y_train, pred_train))
metrics.roc_auc_score(y_train, pred_train)

pred_test=cnn_model2.predict(x_test)
#print(metrics.classification_report(y_test, pred_test))
metrics.roc_auc_score(y_test, pred_test)

# Curva ROC - AUC
fpr, tpr, thresholds = roc_curve(y_test, pred_test)
roc_auc = auc(fpr, tpr)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
roc_display.plot()

# Evaluar el modelo
test_auc, test_recall = cnn_model2.evaluate(x_test, y_test, verbose=2)
print("Test auc:", test_auc)
print("Test recall:", test_recall)

# Matriz de confusión train
pred_train=(cnn_model2.predict(x_train) > 0.50).astype('int')
cm=metrics.confusion_matrix(y_train,pred_train, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Tumor', 'No tumor'])
disp.plot()

# Classification report train
print(metrics.classification_report(y_train, pred_train))

# Matriz de confusión test
pred_test=(cnn_model2.predict(x_test) > 0.50).astype('int')
cm=metrics.confusion_matrix(y_test,pred_test, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Tumor', 'No tumor'])
disp.plot()

# Classification report test
print(metrics.classification_report(y_test, pred_test))



#####################################################
########### Opimización de hiperparámetros ##########
#####################################################

# Función con definición de hiperparámetros a afinar
hp = kt.HyperParameters()

# Función para afinar hiperparámetros
def build_model(hp):
    # Búsqueda de dropout y regularización L2
    dropout_rate = hp.Float('DO', min_value=0.001, max_value=0.3, step=0.001)
    reg_strength = hp.Float("rs", min_value=0.0001, max_value=0.01, step=0.0001)
    optimizer = hp.Choice('optimizer', ['adam'])

    # Búsqueda del número de filtros y neuronas en las capas densas
    filters = hp.Int('filters', min_value=4, max_value=64, step=4)
    dense_units = hp.Int('dense_units', min_value=4, max_value=64, step=4)

    cnn_model2 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), activation='relu', 
                               input_shape=x_train.shape[1:], kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(dense_units, activation='relu'),
        #tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compilación del modelo
    opt = tf.keras.optimizers.Adam(learning_rate=0.001) if optimizer == 'adam' else tf.keras.optimizers.SGD(learning_rate=0.001)
    cnn_model2.compile(optimizer=opt, loss="binary_crossentropy", metrics=["Recall", "AUC"])

    return cnn_model2


# Definir el tuner
tuner = kt.RandomSearch(
    hypermodel=build_model,
    hyperparameters=hp,
    tune_new_entries=True, 
    objective=kt.Objective("val_AUC", direction="max"),
    max_trials=10,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld", 
)

# Busqueda aleatoria de hiperparámetros
tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=100)

fc_best_model = tuner.get_best_models(num_models=1)[0]

# Mostrar resultados
tuner.results_summary()
fc_best_model.summary()

# Evaluar el modelo
test_loss, test_auc=fc_best_model.evaluate(x_test, y_test)
pred_test=(fc_best_model.predict(x_test)>=0.50).astype('int')

# Matriz de confusión de la red neuronal convolucional afinada
cm=metrics.confusion_matrix(y_test,pred_test, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Tumor', 'No tumor'])
disp.plot()

# Classification report
print(metrics.classification_report(y_test, pred_test))


############# Exportar modelo afinado #############
fc_best_model.save('salidas\\best_model.keras')