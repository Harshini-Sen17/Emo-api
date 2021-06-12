import io
#from rest_framework.decorators import api_view
import json
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
from PIL import Image




from rest_framework import viewsets 
from .models import Song
from .serializers import SongSerializer
from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from django.core.files.base import ContentFile
import base64
import uuid


face_classifier = cv2.CascadeClassifier('predictor\model\haarcascade_frontalface_default.xml')

img_height,img_width = 48,48
path='predictor\model\emotion_detection_model_state.pth'
model_state = torch.load(path,map_location=torch.device('cpu'))

class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Suprise']


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ELU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 128)
        self.conv2 = conv_block(128, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.drop1 = nn.Dropout(0.5)
        
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 256, pool=True)
        self.res2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        self.drop2 = nn.Dropout(0.5)
        
        self.conv5 = conv_block(256, 512)
        self.conv6 = conv_block(512, 512, pool=True)
        self.res3 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.drop3 = nn.Dropout(0.5)
        
        self.classifier = nn.Sequential(nn.MaxPool2d(6), 
                                        nn.Flatten(),
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.drop1(out)
        
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.drop2(out)
        
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.res3(out) + out
        out = self.drop3(out)
        
        out = self.classifier(out)
        return out

model = ResNet(1, len(class_labels))
model.load_state_dict(model_state)


@api_view(['POST'])
def predictImage(request):

    imgstr = request.data['base64']
    # image = decodeDesignImage(testimage)
    # print("testimage",testimage)
    # format, imgstr = testimage.split(';base64,') 
    # ext = format.split('/')[-1] 

    data = ContentFile(base64.b64decode(imgstr), name='temp.' + ext) 
   
    image = grab_image(stream=data)
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(img_gray, 1.3, 5)
    print(faces)
    print(img_gray)
    
    x = faces[0][0]
    y = faces[0][1]
    w = faces[0][2]
    h = faces[0][3]

    roi_gray = img_gray[y:y+h, x:x+w]

    img_gray = cv2.resize(roi_gray,(img_height,img_width), interpolation=cv2.INTER_AREA)
 
    roi = tt.functional.to_pil_image(img_gray)
    img = tt.functional.to_grayscale(roi)
    img = tt.ToTensor()(img).unsqueeze(0)

    # make a prediction on the image
    tensor = model(img)
    pred = torch.max(tensor, dim=1)[1].tolist()
    label = class_labels[pred[0]]


    
    context = {'emotion': label}

    return Response(context)


@api_view(['POST'])
# @permission_classes([IsAuthenticated])
def songUploadView(request):
    serializer = SongSerializer(data=request.data)
    
    if serializer.is_valid():
        serializer.save()
    else: 
       print("error")  
    return Response(serializer.data)   


def songFetchingView(md = []):
    ser = []
    print('í m here')
    for m in md:
        task = Song.objects.get(mood = m)
        serializer = SongSerializer(instance = task)
        ser.append(serializer.data)
        print(ser)
    return ser

@api_view(['GET'])
# @permission_classes([IsAuthenticated])
def playlistView(request,la):
    context=[]
    print(la)

    if la == 'Sad':   
        print('why')     
        context = songFetchingView(['Sad'])
        return Response(context)
    # if la == 'Happy':
    #     context = songFetchingView(['Happy'])
    #     return Response(context)    
    # if la == 'Angry':
    #     context = songFetchingView(['Relaxing'])
    #     return Response(context)
    # if la == 'Neutral':
    #     context = songFetchingView(['Relaxing','Happy','Sad'])
    #     return Response(context)
    # if la == 'Surprised':
    #     context = songFetchingView(['Relaxing','Happy','Sad'])
    #     return Response(context)
    return Response('error')




def decodeDesignImage(data):
    try:
        data = base64.b64decode(data.encode('UTF-8'))
        buf = io.BytesIO(data)
        img = Image.open(buf)
        return img
    except:
        return None

def grab_image(path=None, stream=None, url=None):

	if path is not None:
		image = cv2.imread(path)
	
	else:	
		
		if url is not None:
			resp = urllib.urlopen(url)
			data = resp.read()
		
		elif stream is not None:
			data = stream.read()
		
		image = np.asarray(bytearray(data), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	return image