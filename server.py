#-*-  utf-8 -*-

from flask import Flask,request
#from flask_inflate import Inflate,inflate
import ssl,csv,time,os,gzip
from collections import defaultdict
import io
import pickle
import librosa, numpy
from scipy.fftpack import fft
import datetime
from Train_SVM import *

#SOUND_KEY='sound'
LABEL='Label'
TYPE='Type'
TYPE_COLLECTOR='collector'
TYPE_CLASSIFIER='classifier'
#primary_key_of_fake=defaultdict(int)
primary_key_of_real=defaultdict(int)
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
#@inflate
def postJsonHandler():
	try:
		#starttime=datetime.datetime.now()
		sound=gzip.decompress(request.get_data())#byte array?
		#print(sound[:10])
		labelOfData=request.headers.get(LABEL)
		typeOfData=request.headers.get(TYPE)
		#print('degzip time: ',(datetime.datetime.now()-starttime).microseconds)
		if typeOfData==TYPE_COLLECTOR:    #save data
                        pkeyDict=primary_key_of_real
                        pkeyDict[labelOfData]+=1
                        
                        #with open(f'real-data/{labelOfData}_{pkeyDict[label]}_{SOUND_KEY}.csv','w', newline='') as f:
                        with open(f'real-data/{labelOfData}_{pkeyDict[labelOfData]}.csv','w', newline='') as f:
                                w=csv.writer(f)
                                #for val in sound[key].split('\n'):
                                #        w.writerow(val.split(','))
				####instead of enter, used comma in a row
                                w.writerow(int.from_bytes(sound[i:i+2], "big",signed=True) for i in range(4049))

		else:   #classify
                        try:
                                #starttime=datetime.datetime.now()
                                feats = get_features_test(numpy.array([[float(int.from_bytes(sound[i:i+2], "big",signed=True))] for i in range(4049)])) #관측값들 입력
                                #print('get feats time: ',(datetime.datetime.now()-starttime).microseconds)
                                #starttime=datetime.datetime.now()
                                claass = clf.predict([feats])        #classifier 결과값
                                #print('get class time: ',(datetime.datetime.now()-starttime).microseconds)
                                print("class: "+table[claass[0]])
                                print("real : "+labelOfData)
                                        
                                #print(claass[0])
                                
                                return table[claass[0]]
                    
                        except Exception as e:
                                print(e)
                                return 'error'

	except Exception as e:
		print(e)
		return 'error'
	else:
		return 'done'
app.run(host='0.0.0.0', debug=True, port= 9999, ssl_context='adhoc')

