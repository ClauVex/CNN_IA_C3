import numpy as np # linear algebra
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import random
#Dependencies
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
#CNN
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import warnings
import os
import shutil
from PIL import ImageFile
warnings.simplefilter('error', Image.DecompressionBombWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

carpetaDataSet = 'datasetImagenes'
MODEL_FILENAME="modelo_cv.h5"
archivosFuente = []
labelsClases = ['almendrados','asiaticos','caidos','encapuchados','profundos','redondos','saltones']

def transferirEntreCarpetas(source, dest, splitRate):
    global archivosFuente
    archivosFuente=os.listdir(source)
    if(len(archivosFuente)!=0):
        transferirNumArchivos = int(len(archivosFuente)*splitRate)
        transferirIndice = random.sample(range(0,len(archivosFuente)),transferirNumArchivos)
        for i in transferirIndice:
            shutil.move(source+str(archivosFuente[i]),dest+str(archivosFuente[i]))
    else:
        print("No hay archivos fuente")

def transferirClasesEntreCarpetas(source, dest, splitRate):
    for label in labelsClases:
        transferirEntreCarpetas(carpetaDataSet+'/'+source+'/'+label+'/',
        carpetaDataSet+'/'+dest+'/'+label+'/',
        splitRate)

#checar si el folder test tiene archivos o no
transferirClasesEntreCarpetas('test','entrenamiento',1.0)
#dividir una parte de los datos de entrenamiento a la carpeta de test
transferirClasesEntreCarpetas('entrenamiento','test',0.20)

vector_X=[]
vector_Y=[]

def prepararNombreConLabels(carpeta):
    archivosFuente=os.listdir(carpetaDataSet+'/entrenamiento/'+carpeta)
    for archivo in archivosFuente:
        vector_X.append(archivo)
        if(carpeta==labelsClases[0]):
            vector_Y.append(0)
        elif(carpeta==labelsClases[1]):
            vector_Y.append(1)
        elif(carpeta==labelsClases[2]):
            vector_Y.append(2)
        elif(carpeta==labelsClases[3]):
            vector_Y.append(3)
        elif(carpeta==labelsClases[4]):
            vector_Y.append(4)
        elif(carpeta==labelsClases[5]):
            vector_Y.append(5)
        else:
            vector_Y.append(6)

prepararNombreConLabels(labelsClases[0])
prepararNombreConLabels(labelsClases[1])
prepararNombreConLabels(labelsClases[2])
prepararNombreConLabels(labelsClases[3])
prepararNombreConLabels(labelsClases[4])
prepararNombreConLabels(labelsClases[5])
prepararNombreConLabels(labelsClases[6])

vector_X=np.asarray(vector_X)
vector_Y=np.asarray(vector_Y)

tamaño_batch = 20
epoch = 20

def getModelo():
    modelo = Sequential()
    modelo.add(Conv2D(64,(7,7),padding='same', activation='relu', input_shape=(img_filas,img_columnas,3)))
    modelo.add(Conv2D(64,(7,7),activation='relu'))
    modelo.add(MaxPooling2D(pool_size=(2,2)))
    modelo.add(Dropout(0.25))

    modelo.add(Conv2D(32,(7,7),padding='same', activation='relu'))
    modelo.add(Conv2D(32,(7,7),activation='relu'))
    modelo.add(MaxPooling2D(pool_size=(2,2)))
    modelo.add(Dropout(0.25))

    modelo.add(Conv2D(16,(7,7),padding='same', activation='relu'))
    modelo.add(Conv2D(16,(7,7),activation='relu'))
    modelo.add(MaxPooling2D(pool_size=(2,2)))
    modelo.add(Dropout(0.25))

    modelo.add(Flatten())
    modelo.add(Dense(64, activation='relu'))
    modelo.add(Dropout(0.1))
    modelo.add(Dense(32, activation='relu'))
    modelo.add(Dropout(0.1))
    modelo.add(Dense(16, activation='relu'))
    modelo.add(Dropout(0.1))
    modelo.add(Dense(7, activation='softmax'))

    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return modelo

def metrica(y_true, y_pred):
    exactitud=accuracy_score(y_true, y_pred)
    precision=precision_score(y_true, y_pred,average='weighted')
    f1Puntaje=f1_score(y_true, y_pred, average='weighted') 
    print("Exactitud  : {}".format(exactitud))
    print("Precision : {}".format(precision))
    print("Puntaje : {}".format(f1Puntaje))
    confusion=confusion_matrix(y_true, y_pred)
    print(confusion)
    return exactitud, precision, f1Puntaje

img_filas, img_columnas =  32, 32

entrenamiento_path=carpetaDataSet+'/entrenamiento/'
validacion_path=carpetaDataSet+'/validacion/'
test_path=carpetaDataSet+'/test/'
modelo=getModelo()

kFold = StratifiedKFold(n_splits=3, shuffle=True)
kFold.get_n_splits(vector_X,vector_Y)
foldNum=0
for entrenamiento_index, validacion_index in kFold.split(vector_X,vector_Y):
    transferirClasesEntreCarpetas('validacion', 'entrenamiento', 1.0)
    foldNum +=1
    print("Resultados de fold", foldNum)
    X_entrenamiento, X_validacion = vector_X[entrenamiento_index], vector_X[validacion_index]
    Y_entrenamiento, Y_validacion = vector_Y[entrenamiento_index], vector_Y[validacion_index]

    for _ in range(len(X_validacion)):
        etiqueta = ''
        if(Y_validacion[_]==0):
            etiqueta=labelsClases[0]
        elif(Y_validacion[_]==1):
            etiqueta=labelsClases[1]
        elif(Y_validacion[_]==2):
            etiqueta=labelsClases[2]
        elif(Y_validacion[_]==3):
            etiqueta=labelsClases[3]
        elif(Y_validacion[_]==4):
            etiqueta=labelsClases[4]
        elif(Y_validacion[_]==5):
            etiqueta=labelsClases[5]
        else:
            etiqueta=labelsClases[6]
        shutil.move(carpetaDataSet+'/entrenamiento/'+etiqueta+'/'+X_validacion[_],
        carpetaDataSet+'/validacion/'+etiqueta+'/'+X_validacion[_])
    
    generadorDato_entrenamiento = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.20,
        fill_mode="nearest"
    )
    validacion_generadorDato = ImageDataGenerator(rescale=1./255)
    test_generadorDato = ImageDataGenerator(rescale=1./255)

    generar_entrenamiento = generadorDato_entrenamiento.flow_from_directory(
        entrenamiento_path,
        target_size=(img_filas,img_columnas),
        batch_size=tamaño_batch,
        class_mode='categorical',
        subset='training'
    )

    validacion_generador = validacion_generadorDato.flow_from_directory(
        validacion_path,
        target_size=(img_filas,img_columnas),
        batch_size=tamaño_batch,
        class_mode=None,
        shuffle=False
    )

    # fit model
    historia=modelo.fit_generator(generar_entrenamiento, 
                        epochs=epoch)
    predicciones = modelo.predict_generator(validacion_generador, verbose=1)
    yPredicciones = np.argmax(predicciones, axis=1)
    true_classes = validacion_generador.classes
    # evaluate validation performance
    print("***Performance on Validation data***")    
    valAcc, valPrec, valFScore = metrica(true_classes, yPredicciones)

    # =============TESTING=============
print("==============TEST RESULTS============")
generador_test = test_generadorDato.flow_from_directory(
        test_path,
        target_size=(img_filas, img_columnas),
        batch_size=tamaño_batch,
        class_mode=None,
        shuffle=False) 
predicciones = modelo.predict(generador_test, verbose=1)
yPredicciones = np.argmax(predicciones, axis=1)
true_classes = generador_test.classes

testAcc,testPrec, testFPuntaje = metrica(true_classes, yPredicciones)
modelo.save(MODEL_FILENAME)