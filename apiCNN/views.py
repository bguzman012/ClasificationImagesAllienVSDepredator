#CONTROLADOR

from rest_framework import generics #para microservicio
from apiCNN import models
from apiCNN import serializers

from apiCNN.models import Image
from apiCNN.models import ImageTest

from django.shortcuts import render
from apiCNN.Logica import modeloCNN #para utilizar modelo SNN

config = {

    'apiKey': "AIzaSyDBYpL2tb3yh3SIPo2BFhlS7slKruVGOic",
    'authDomain': "proyectotiendajpri.firebaseapp.com",
    'databaseURL': "https://proyectotiendajpri.firebaseio.com",
    'projectId': "proyectotiendajpri",
    'storageBucket': "proyectotiendajpri.appspot.com",
    'messagingSenderId': "1046831721926",
    'appId': "1:1046831721926:web:7402a636a8cd165f4b16c7",
    'measurementId': "G-MKSCN84RDE"
}

#firebase = pyrebase.initialize_app(config)
#auth = firebase.auth()
    

class Clasificacion():
    def inicio(request):

        return render(request, "uploadImage.html")
    
    def predecir(request):
        test = ImageTest()

        param = request.FILES.get('archivosubido')
        
        nombre = param.name
        print(nombre)
        test.image = param
        test.save()
        
        resul=modeloCNN.modeloCNN.predecirSobrevivencia(modeloCNN.modeloCNN, nombre)
        
        
        return render(request, "uploadImage.html")
        
        



