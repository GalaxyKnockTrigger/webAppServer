#-*-  utf-8 -*-

"""
when /class/ called, load preprocessed model and classifies the input
"""
from flask import Flask,request,jsonify
#from flask_inflate import Inflate,inflate
import ssl,json,csv,time,os,gzip
from collections import defaultdict
import io
import pickle
import librosa, numpy
from scipy.fftpack import fft

from Train_SVM import *

SOUND_KEY='sound'
ACC_KEY='acc'
GYRO_KEY='gyro'
DATA=(SOUND_KEY,ACC_KEY,GYRO_KEY)
LABEL='label'
STATUS='status'
STATUS_FAKE='fake'
STATUS_REAL='real'
primary_key_of_fake=defaultdict(int)
primary_key_of_real=defaultdict(int)

for filename in os.listdir('fake-data'):
	label,pkey,_=filename.split('_')
	primary_key_of_fake[label]=max(primary_key_of_fake[label],int(pkey))
for filename in os.listdir('real-data'):
	try:
		label,pkey,_=filename.split('_')
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
   
@app.route('/class/', methods = ['GET','POST'])
def postJsonHandler_class():
	if request.method == 'POST':
		try:
			content=json.loads(gzip.decompress(request.get_data()).decode("utf-8"))
			
			data = {SOUND_KEY:[],ACC_KEY:[],GYRO_KEY:[]}

			for key in DATA:
				if key not in content.keys():
					continue
				temp=[]
				for val in content[key].split('\n')[0 if key==SOUND_KEY else 1:-1]:
					#x,y,z
					#val
					temp.append(map(int, val.split(',')))
                        
				data[key] = temp
			
			feats = get_features(data[SOUND_KEY], data[ACC_KEY], data[GYRO_KEY]) #관측값들 입력
			class__ = clf.predict(feats)        #classifier 결과값
			print(class__)
                                
			return class__
		    
		except Exception as e:
			print(e)
			#print(request.data)
			return 'error'
	else:
		return 'knot: hello,world'

@app.route('/', methods = ['GET','POST'])
#@inflate
def postJsonHandler():
	if request.method == 'POST':
		try:
			content=json.loads(gzip.decompress(request.get_data()).decode("utf-8"))
			label=content[LABEL]
			statusDir='real-data' if content[STATUS]==STATUS_REAL else 'fake-data'
			pkeyDict=primary_key_of_real if content[STATUS]==STATUS_REAL else primary_key_of_fake
			pkeyDict[label]+=1
			for key in DATA:
				if key not in content.keys():
					continue
				with open(f'{statusDir}/{label}_{pkeyDict[label]}_{key}.csv','w', newline='') as f:
					w=csv.writer(f)
					for val in content[key].split('\n'):
						#x,y,z
						#val
						w.writerow(val.split(','))
                                
			return 'done'
		except Exception as e:
			print(e)
			#print(request.data)
			return 'error'
	else:
		return 'knot: hello,world'
app.run(host='0.0.0.0', debug=True, port= 9999, ssl_context='adhoc')
