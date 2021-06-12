from django.db import models

# Create your models here.


from unittest.util import _MAX_LENGTH

from datetime import datetime, timedelta

class Song(models.Model):
    MOODS = (
    ('Happy', 'Happy'),
    ('Relaxing', 'Relaxing'),
    ('Sad', 'Sad'),
    )
    id = models.AutoField(primary_key=True)  
    title = models.CharField(max_length=50)
    singer = models.CharField(max_length=20) 
    image = models.ImageField()
    # audio_file = models.FileField(blank = True,null = True)
    mood = models.CharField(max_length=30,choices=MOODS) 
    link = models.URLField(max_length = 200)
    
        
    def __str__(self):
        return self.title