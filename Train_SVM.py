# -*- coding: utf-8 -*-import matplotlib.pyplot as plt
import numpy as np
import librosa
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_mfccs(data):
    sample_rate=48000

    S = librosa.feature.melspectrogram(data, sr=sample_rate, n_fft=2048, n_mels=128, hop_length=64, win_length=None)

    log_S = librosa.power_to_db(S, ref=np.max)

    mfcc = librosa.feature.mfcc(S=log_S, sr=sample_rate, n_mfcc=16, n_fft=64, hop_length=64, win_length=None)

    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    return delta2_mfcc	#코드 확인 필요

def get_features(d):
    d=np.array(d)
    """
    dataset_sound = np.array(sound[0])
    dataset_acc_x, dataset_acc_y, dataset_acc_z = np.array(acc).T
    dataset_gyr_x, dataset_gyr_y, dataset_gyr_z = np.array(gyr).T
    """
    dataset_sound = d[:4096]
    
    dataset_IMU = np.array_split(d[4096:], 6)
    
    for d in dataset_IMU:
        d/=np.max(abs(d))
        
    
    #dataset_acc_x = d[4096:4104]
    #dataset_acc_y = d[4104:4112]
    #dataset_acc_z = d[4112:4120]
    #dataset_gyr_x = d[4120:4128]
    #dataset_gyr_y = d[4128:4136]
    #dataset_gyr_z = d[4136:4144]

    dataset_sound_mag = fft(dataset_sound)
    dataset_sound_mag = abs(dataset_sound_mag[:2049])
    if 0 in dataset_sound_mag:
        dataset_sound_mag = np.where(dataset_sound_mag==0, 0.0000001, dataset_sound_mag)
    dataset_sound_mag_log = np.log(dataset_sound_mag)
    
    dataset_mfccs = get_mfccs(dataset_sound).flatten()

    
    #dataset_acc_x = np.pad(dataset_acc_x, ((0, 248)), 'constant')
    #dataset_acc_x_mag = abs(fft(dataset_acc_x))
    #
    #dataset_gyr_z = np.pad(dataset_gyr_z, ((0, 248)), 'constant')
    #dataset_gyr_z_mag = abs(fft(dataset_gyr_z))
    
    """
    print(dataset_sound_mag)
    print(dataset_sound_mag_log.dtype)
    print(dataset_mfccs.dtype)
    print(dataset_acc_x_mag.dtype)
    print(dataset_gyr_z_mag.dtype)
    """
    return np.concatenate (([dataset_sound_mag, dataset_sound_mag_log, dataset_mfccs], dataset_IMU))


def train(userId):
    dataset=[]
    commandset=[]

    ##DB
    #for commandId,data FROM DATA:
    #	dataset.append(data)	#np.arr로 바꿔서
    #	commandset.append(commandId)

    X_train, X_test, y_train, y_test = train_test_split(dataset, commandset, test_size=0.33, random_state=42)

    dataset_pre = np.zeros((len(X_train), 5138))
    dataset_test_pre = np.zeros((len(X_test), 5138))
    for i, d in enumerate(X_train):
        if i%1000 == 0:
            print(i, '/', len(X_train))
            
        dataset_pre[i]  = get_features(d)
        
    for i, d in enumerate(X_test):
        if i%1000 == 0:
            print(i, '/', len(X_test))
        dataset_test_pre[i] = get_features(d)

    from sklearn import svm

    clf = svm.SVC(kernel='linear')
    clf.fit(dataset_pre, y_train)
	##DB
	#insert into USER (model) values clf
