from __future__ import division, print_function
# coding=utf-8
#import sys
import os
#import glob
#import re
import numpy as np
import pandas as pd
#import librosa
#from librosa import display
import time

import sys, os
from unittest.mock import MagicMock

sys.path.append(os.path.abspath('..'))


# Mock module to bypass pip install
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = [
    'librosa', 'librosa.display']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)


# Keras
import keras
#from tensorflow.keras.models import load_model
from keras.models import load_model

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)


MODEL_PATH1 = 'models/CNN_1D.h5'
model1 = load_model(MODEL_PATH1)
model1._make_predict_function()      

MODEL_PATH2 = 'models/cnn_1dshallow_8out'
model2 = load_model(MODEL_PATH2)
model2._make_predict_function() 

MODEL_PATH3 = 'models/VGG_50_epoch.h5'
model3 = load_model(MODEL_PATH3)
model3._make_predict_function() 


def normalize(X):
  max_data = np.max(X)
  min_data = np.min(X)
  X = (X-min_data)/(max_data-min_data+1e-6)
  X =  X-0.5

  return X

def cal_power(X, sample_rate):
  energy = np.sum(X.astype(float)**2)
  power = (energy * sample_rate)/(X.size)
  power = (power-85.64)/151.32
  return power
  
#def get_dur(file):
#  return librosa.get_duration(filename=file)

def set_sr(audio_duration):
  if(audio_duration>=5):
    d = 5
    s = 16000
  else:
    d = audio_duration
    s = 80000/audio_duration

  return d,s

#Extract features (mfcc, chroma, mel) from a sound file
'''def extract_feature_1D(X, sample_rate):

  stft=np.abs(librosa.stft(X))  # Getting the Short Term Fourier Transform (STFT)
  result=np.array([])  # Makes an empty result array 
  mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
  mfccs=normalize(mfccs)
  result=np.hstack((result, mfccs))  #Appends mfccs to result array
  chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
  chroma=normalize(chroma)
  result=np.hstack((result, chroma)) # Appends chroma to result array
  mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
  mel=normalize(mel)
  result=np.hstack((result, mel))  # Appends mel to result array
  result = result.reshape(1,180,1)
  return result'''
 
'''def extract_feature_2D(data, sample_rate):
  # dur, sample_rate = set_sr(librosa.get_duration(filename=file))
  # data, sample_rate = librosa.load(file, sr = sample_rate, duration = dur)
  mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13)
  delta_mfcc = librosa.feature.delta(mfcc, order=2)
  delta_mfcc = normalize(delta_mfcc)
  delta_mfcc = delta_mfcc.reshape(1,13,157,1)
  return delta_mfcc'''


# Break Audio Function
'''def break_audio(file):
  lis = []
  sr = 16000
  count = 0
  if(get_dur(file)<=5):
    count+=1
    dur, sample_rate = set_sr(get_dur(file))
    data, sample_rate = librosa.load(file, sr = sample_rate, duration = dur)
    sr = sample_rate
    lis.append(data)

  else:
    dur = get_dur(file)
    sample_rate = 16000
    data, sample_rate = librosa.load(file, sr = sample_rate, duration = dur)
    t = int(dur/5)
    for i in range(0,t):
      arr = data[i*80000 : (i+1)*80000]
      lis.append(arr)
      count+=1

  return lis,count,sr'''


def model_predict(file, model1, model2, model3):  
  emotion_dir = {0: 'Neutral' , 1: 'Happy', 2: 'Sad/Bored', 3: 'Anger/Disgust', 4: 'Fear', 5: 'Surprise',6: 'Anger/Disgust',7: 'Sad/Bored'} 
    
  arr, count, sr = break_audio(file)
  t1 = []
  t2 = []
  emo = []
  conf = []
  p = []
  for i in range(0,count):
    feature_1d = extract_feature_1D(arr[i], sr)
    feature_2d = extract_feature_2D(arr[i], sr)
    power = cal_power(arr[i],sr)
    y1 = model1.predict(feature_1d)
    y2 = model2.predict(feature_1d)
    y3 = model3.predict(feature_2d)

    t1.append(i*5)
    t2.append((i+1)*5)
    emo.append(emotion_dir[np.argmax(y1+y2+2*y3)])
    conf.append((100/4)*np.max(y1+y2+2*y3))
    p.append(int(10*power))

  #return emo[0],conf[0],p[0]
  return emo,conf,p

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        model1.save()
        model2.save()
        model3.save()
        # Failure to return a redirect or render_template
    else:
        return render_template('a.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        #basepath = os.path.dirname(__file__)
        basepath=os.path.dirname(os.path.realpath('__file__'))
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        emotion,confidence,power=model_predict(file_path,model1, model2, model3)  
        n = len(emotion)
        print("n:",n)
        print("emotion:",emotion)
        result = ""
        for i in range(0,n):
            t = " For %02d-%02d sec--> Emotion: %s | Confidence: %.2f %% | Power %s "  % (i*5, (i+1)*5,emotion[i],confidence[i], power[i])
            result += t+'\n'
        return result
        #return(" %s | Confidence: %.2f %% | Power: %.2f" % (emotion, confidence, power))

if __name__ == '__main__':
    
    
    #app.run(debug=True,use_reloader=False)
    #app.run(debug=True,host='192.168.1.207',use_reloader=False)
    app.run(debug=True,host='0.0.0.0',use_reloader=False)
