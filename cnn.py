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

tamaño_batch = 25
epoch = 20

def getModelo():
    modelo = Sequential()
    modelo.add(Conv2D(64,(3,3),padding='same', activation='relu', input_shape=(largo,ancho,3)))
    modelo.add(Conv2D(64,(3,3),activation='relu'))
    modelo.add(MaxPooling2D(pool_size=(2,2)))
    modelo.add(Dropout(0.25))
