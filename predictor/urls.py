from django.urls import path
from . import views

urlpatterns = [
   path('emotion/', views.predictImage, name='predictEmotion'),
   path('song-upload-view/', views. songUploadView, name="song-upload-view"),
   path('playlist/<str:la>/', views.playlistView, name="play-list-view"),
]