from django.db import models
from django.urls import reverse
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.python.keras.models import load_model, model_from_json
from keras import backend as K
from apiCNN import models
import os
from tensorflow.python.keras.models import Sequential
import pathlib
from keras.applications.resnet50 import preprocess_input

from apiCNN.models import Image
from PIL import Image as pil_im

class modeloCNN():
    """Clase modelo SNN"""

    Selectedmodel = Sequential()
    
    def suma(num1=0,num2=0):
        resultado=num1+num2
        return resultado
    def cargarRNN(nombreArchivoModelo,nombreArchivoPesos):
        K.reset_uids()
        # Cargar la Arquitectura desde el archivo JSON
        with open(nombreArchivoModelo+'.json', 'r') as f:
            model = model_from_json(f.read())
        # Cargar Pesos (weights) en el nuevo modelo
        model.load_weights(nombreArchivoPesos+'.h5') 
        print("Red Neuronal Cargada desde Archivo") 
        return model

    def predecirSobrevivencia(self, image, param):
        #Modelo optimizado
        print('MODELO OPTIMIZADO')
        nombreArchivoModelo=r'apiCNN/Logica/architectura_optimizada'
        nombreArchivoPesos=r'apiCNN/Logica/pesos_optimizados'
        #return (str(pathlib.Path().absolute())+'\Modelos')
        self.Selectedmodel=self.cargarRNN(nombreArchivoModelo,nombreArchivoPesos) 

        resultadoProbabilidad = []
        resultadoProbabilidad = self.predict(self, image, param)
        
        resultado = np.argmax(resultadoProbabilidad)

        mensaje = ""
        if resultado == 0:
            etiqueta = 'Alien'
            probabilidad = resultadoProbabilidad[0, 0]
            probabilidad = probabilidad * 100
            mensaje = "La imagen corresponde al personaje: " + etiqueta + "; con certeza del " + str(round(probabilidad, 2)) + "%"
        else:
            etiqueta = 'Depredador'
            probabilidad = resultadoProbabilidad[0, 1]
            probabilidad = probabilidad * 100
            mensaje = "La imagen corresponde al personaje: " + etiqueta + "; con certeza del " + str(round(probabilidad, 2)) + "%"           
            
        print(mensaje)
        return mensaje

    def predict(self, image, param):


        validation = image
        url= "test/"
        val_url = url + validation
        img = pil_im.open(val_url)
        val = np.stack(preprocess_input(np.array(img.resize((150, 150)))))
        val = (np.expand_dims(val,0))
        imagenModel = models.Image()
        imagenModel.image = param

        pred_probs = self.Selectedmodel.predict(val)
        resultado = np.argmax(pred_probs)
        if resultado == 0:
            etiqueta = 'Alien'
            probabilidad = pred_probs[0, 0]
            imagenModel.label = etiqueta
            imagenModel.probability = round(probabilidad, 2)
        else:
            etiqueta = 'Depredador'
            probabilidad = pred_probs[0, 1]
            imagenModel.label = etiqueta
            imagenModel.probability = round(probabilidad, 2)

        imagenModel.save()        
        print('Imagen Guardada')

        return pred_probs