from django.db import models

class Image(models.Model):
    #https://mc.ai/integrar-modelo-de-red-neuronal-convolucional-en-django/
    # file will be uploaded to MEDIA_ROOT / uploads 
    image = models.ImageField(upload_to ='uploads/') 
    # or... 
    # file will be saved to MEDIA_ROOT / uploads / 2015 / 01 / 30 
    # upload = models.ImageField(upload_to ='uploads/% Y/% m/% d/')
    label = models.CharField(max_length=20, blank=True)
    probability = models.FloatField()
    
#MAESTRO DETALLE
"""
class Musician(models.Model):
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    instrument = models.CharField(max_length=100)

class Album(models.Model):
    artist = models.ForeignKey(Musician, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    release_date = models.DateField()
    num_stars = models.IntegerField()
"""