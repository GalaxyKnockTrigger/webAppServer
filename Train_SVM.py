# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 00:53:56 2022

@author: tngus
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import csv

import librosa, numpy
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

parentPath = 'C:/Users/hyonj/Documents/카카오톡 받은 파일/data'

def get_AllData():
    types = ('sound', 'acc', 'gyro')

    # for path in os.listdir(parentPath):
    #     if 'sound' in path:
    #         paths['sound'].append(path)
    #     elif 'acc' in path:
    #         paths['acc'].append(path)
    #     elif 'gyro' in path:
    #         paths['gyro'].append(path)

    allData = {}
    for dataType in types:
        allData[dataType]={}

    for path in os.listdir(parentPath):
        if not '.csv' in path:
            continue
        # print(path)
        label,num,dataType = path.split('_')
        
        # if label != 'clap':
        #     continue
        
        dataType = dataType.split('.')[0]
        
        if not label in allData[dataType].keys():
            allData[dataType][label] = {}
        
        allData[dataType][label][num] = {}
        
        nowData = allData[dataType][label][num]
        
        nowData['timestamp'] = []
        if dataType == 'sound':
            nowData['value'] = []
        else :
            for v in ('x', 'y', 'z'):
                nowData[v] = []
        with open(os.path.join(parentPath, path)) as file:
            reader = csv.reader(file)
            
            # i = 0
            for raw in reader:
                if '#' in raw[0]:
                    continue
                if raw[0] == '':
                    break
                if dataType == 'sound':
                    nowData['value'].append(int(raw[0]))
                else :
                    k = 0
                    for v in ('x', 'y', 'z'):
                        # print(raw[k])
                        nowData[v].append(float(raw[k]))
                        k+=1
                # i+=1
            file.close()
        
        if dataType == 'sound':
            allData[dataType][label][num]['value'] = np.array(nowData['value'], dtype=np.int16)[:4096]
        else :
            for v in ('x', 'y', 'z'):
                allData[dataType][label][num][v] = np.array(nowData[v], dtype=np.float32)[:8]
    return allData

def get_dataset(Data):    
    data = []
    labels = []
    tabel = {}
    for label in allData[types[0]]:
        for num in allData[types[0]][label]:
            temp = []
            for dataType in types:
                if dataType == 'sound':
                    temp = allData[dataType][label][num]['value']
                    if label not in tabel:
                        tabel[label] = len(tabel.keys())
                    labels.append(tabel[label])
                else :
                    for v in ('x', 'y', 'z'):
                        temp = np.concatenate((temp, allData[dataType][label][num][v]))
                        
            if len(data)==0:
                data = np.array([temp])
            else:
                data = np.concatenate((data, [temp]))
    return data, labels, tabel


def get_mfccs(data):
    sample_rate=48000

    S = librosa.feature.melspectrogram(data, sr=sample_rate, n_fft=2048, n_mels=128, hop_length=64, win_length=None)

    log_S = librosa.power_to_db(S, ref=np.max)

    mfcc = librosa.feature.mfcc(S=log_S, sr=sample_rate, n_mfcc=16, n_fft=64, hop_length=64, win_length=None)

    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    return delta2_mfcc

def get_features(d):
    d = np.array(d)
    
    """
    dataset_sound = np.array(sound[0])
    dataset_acc_x, dataset_acc_y, dataset_acc_z = np.array(acc).T
    dataset_gyr_x, dataset_gyr_y, dataset_gyr_z = np.array(gyr).T
    """
    dataset_sound = d[:4096]
    dataset_acc_x = d[4096:4104]
    dataset_acc_y = d[4104:4112]
    dataset_acc_z = d[4112:4120]
    dataset_gyr_x = d[4120:4128]
    dataset_gyr_y = d[4128:4136]
    dataset_gyr_z = d[4136:4144]
    dataset_sound_mag = fft(dataset_sound)
    dataset_sound_mag = abs(dataset_sound_mag[:2049])
    if 0 in dataset_sound_mag:
        dataset_sound_mag = np.where(dataset_sound_mag==0, 0.0000001, dataset_sound_mag)
    dataset_sound_mag_log = np.log(dataset_sound_mag)
    
    dataset_mfccs = get_mfccs(dataset_sound).flatten()


    
    dataset_acc_x = np.pad(dataset_acc_x, ((0, 248)), 'constant')
    dataset_acc_x_mag = abs(fft(dataset_acc_x))
    
    dataset_gyr_z = np.pad(dataset_gyr_z, ((0, 248)), 'constant')
    dataset_gyr_z_mag = abs(fft(dataset_gyr_z))
    
    """
    print(dataset_sound_mag)
    print(dataset_sound_mag_log.dtype)
    print(dataset_mfccs.dtype)
    print(dataset_acc_x_mag.dtype)
    print(dataset_gyr_z_mag.dtype)
    """
    return np.concatenate ((dataset_sound_mag, dataset_sound_mag_log, dataset_mfccs, abs(dataset_acc_x_mag[:129]), abs(dataset_gyr_z_mag[:129])))


allData = getAlldata()
dataset, labels, tabel = get_dataset(allData)

X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.33, random_state=42)



dataset_pre = np.zeros((len(X_train), 5396))
dataset_test_pre = np.zeros((len(X_test), 5396))
for i, d in enumerate(X_train):
    if i%1000 == 0:
        print(i, '/', len(X_train))
        
    dataset_pre[i]  = get_features(d)
    
for i, d in enumerate(X_test):
    if i%1000 == 0:
        print(i, '/', len(X_test))
    dataset_test_pre[i] = get_features(d)
import pickle

from sklearn import svm

try:
    # load
    with open('model_preproccessed.pkl', 'rb') as f:
        clf = pickle.load(f)
except:

    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)


from sklearn.metrics import accuracy_score
y_pred = clf.predict(X_test)

print(accuracy_score(y_pred, y_test))

with open('model_preproccessed.pkl','wb') as f:
    pickle.dump(clf,f)
  
