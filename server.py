#-*-  utf-8 -*-

from flask import Flask,request
import ssl,csv,time,os,gzip
from collections import defaultdict
import io
import pickle
import librosa
import numpy as np
from scipy.fftpack import fft
import datetime
from Train_SVM import *

def socket_write(input_str):
    client.send( bytes(input_str+'\n', 'utf-8'))
def sound_augmentation_lib(value, noise_factor=400):
    temp_data = []
    for k in range(10):
        data_up = np.concatenate((value[:4048], np.zeros(48)))
        data_down = np.concatenate((np.zeros(48), value[48:]))
        
        noise = np.random.randn(len(value))
        augmented_data = value + noise_factor * noise
        augmented_data = augmented_data.astype(np.float32)
        
        augmented_data_up = data_up + noise_factor * noise
        augmented_data_up = augmented_data_up.astype(np.float32)
        
        augmented_data_down = data_down + noise_factor * noise
        augmented_data_down = augmented_data_down.astype(np.float32)
        
        try:
            temp_data = np.concatenate((temp_data, [augmented_data, augmented_data_up, augmented_data_down]))
        except:
            temp_data = np.array([augmented_data, augmented_data_up, augmented_data_down])
    return np.concatenate((temp_data, [value]))
    
#SOUND_KEY='sound'
LABEL='Label'
TYPE='Type'
TYPE_COLLECTOR='collector'
TYPE_CLASSIFIER='classifier'
TYPE_END='collectingEnd'
#primary_key_of_fake=defaultdict(int)
primary_key_of_real=defaultdict(int)
#train()
with open('table.pkl', 'rb') as f:
        key_table = pickle.load(f)
table = {key_table[k]:k for k in key_table.keys()}
print(table)
'''for filename in os.listdir('fake-data'):
    label,pkey,_=filename.split('_')
    primary_key_of_fake[label]=max(primary_key_of_fake[label],int(pkey))'''
for filename in os.listdir('real-data'):
    try:
        #label,pkey,_=filename.split('_')
        label,pkey=filename.split('_')
        pkey=pkey.split('.')[0]
    except:
        print(filename)
    else:
        primary_key_of_real[label]=max(primary_key_of_real[label],int(pkey))

print(primary_key_of_real.items())

app = Flask(__name__)
#Inflate(app)

clf=0

with open("model_preproccessed.pkl", "rb") as f:
    clf = pickle.load(f)

@app.route('/', methods = ['POST'])
def postJsonHandler():
    try:
        #starttime=datetime.datetime.now()
        sound=gzip.decompress(request.get_data())#byte array?
        print(len(sound))

        #print(sound[:10])
        typeOfData=request.headers.get(TYPE)
        #print('degzip time: ',(datetime.datetime.now()-starttime).microseconds)
        if typeOfData==TYPE_END:
            train()
        elif typeOfData==TYPE_COLLECTOR:    #save data
            labelOfData=request.headers.get(LABEL)
            pkeyDict=primary_key_of_real
            sound_data = list(int.from_bytes(sound[i*2:i*2+2], "big",signed=True) for i in range(4096))
            #print(sound_data)
            #print(len(sound_data))
            sound_data = np.array(sound_data)
            #print(sound_data.shape)
            aug_sound_data = sound_augmentation_lib(sound_data, noise_factor=400)
            for data in aug_sound_data:
                pkeyDict[labelOfData]+=1
                with open(f'real-data/{labelOfData}_{pkeyDict[labelOfData]}.csv','w', newline='') as f:
                    w=csv.writer(f)
                    #for val in sound[key].split('\n'):
                    #        w.writerow(val.split(','))
                    w.writerow(data[i] for i in range(4096))

        else:   #classify
            try:
                #starttime=datetime.datetime.now()
                feats = get_features_test(np.array([[float(int.from_bytes(sound[i*2:i*2+2], "big",signed=True))] for i in range(4096)])) #관측값들 입력
                #print('get feats time: ',(datetime.datetime.now()-starttime).microseconds)
                #starttime=datetime.datetime.now()
                claass = clf.predict([feats])        #classifier 결과값
                #print('get class time: ',(datetime.datetime.now()-starttime).microseconds)
                print("class: "+table[claass[0]])
                #print("real : "+labelOfData)
                        
                print(claass[0])
                #asdfasdfasdfasdf
                socket_write(table[claass[0]])
                return table[claass[0]]
            
            except Exception as e:
                print(e)
                return 'error'

    except Exception as e:
        print(e)
        return 'error'
    else:
        return 'done'
import socket
import time 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('0.0.0.0', 8585 ))
s.listen(0)  
client, addr = s.accept()
client.settimeout(5)
#client.send(bytes('Hello From Python'+'\n', 'utf-8'))
app.run(host='0.0.0.0', port= 9999, ssl_context='adhoc')

