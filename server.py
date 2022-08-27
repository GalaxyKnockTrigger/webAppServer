#-*-  utf-8 -*-

from flask import Flask,request
import ssl,gzip
import librosa
import numpy as np
from scipy.fftpack import fft
from Train_SVM import *

def sound_augmentation_lib(value, noise_factor=400):	#사운드만 aug하고 나머지는 그냥 붙임
	#return: 30 data with noise, 1 original data 
	#value: 4096 sound, 24 acc, 24 gyro
    temp_data = []
    for k in range(10):
        data_up = np.concatenate((value[48:], np.zeros(48)))
        data_down = np.concatenate((np.zeros(48), value[:4048]))
        
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

LABEL='Label'
CMD='cmd'
USER_ID='user-id'
TYPE='Type'
TYPE_COLLECTOR='collector'
TYPE_CLASSIFIER='classifier'
TYPE_END='collectingEnd'


app = Flask(__name__)


def generateUserId():
    ##DB
    #insert user into USER and get pk userid
    return userId

def getCommandId(userId,labelOfData):
	##DB
	#check if there's already command in COMMAND
		#generate new row and return pk commandId unless exist
		#else, get and return commandId
	return commandId

def getClf(userId):
	##DB
	#select model from USER where user-id = userId
	return clf




@app.route('/',methods=['GET','POST'])
def index():
	if request.method=='GET':
		#initialize app
		try:
			userId=generateUserId()
		except Exception as e:
			print(e)
			return 'cannot make user id'
		else:
			return userId
	#POST
	try:
		raw_data=gzip.decompress(request.get_data()).decode("utf-8")
		typeOfData=request.headers.get(TYPE)
		userId=request.headers.get(USER_ID)

		#collect ends, train start
		if typeOfData==TYPE_END:
			#!!!can change trainLabel to command-id in COMMAND
			train(userId)

		#collect data
		elif typeOfData==TYPE_COLLECTOR:
			cmdOfData=request.headers.get(CMD)	#ex) 442, 112
			labelOfData=request.headers.get(LABEL)	#ex) door, desk
			
			temp=list(map(float,raw_data.rstrip(',').split(',')))
			sound_data=np.array(temp[:4096])
			accgyro_data=np.array(temp[-48:])

			aug_datas=sound_augmentation_lib(sound_data,noise_factor=400)

			commandId=getCommandId(userId,labelOfData)

			for d in aug_datas:
				data=np.concatenate((d,accgyro_data))
				##DB
				#insert data into DATA with commandId

		#classify
		else:
			temp_data=list(map(float,raw_data.rstrip(',').split(',')))
			feats=get_features(temp_data)
			clf=getClf(userId)
			label=clf.predict([feats])	#originally label number, want to change to label
			print(f'label: {label}')
			return label

	except Exception as e:
		print(e)
		return 'error'

app.run(host='0.0.0.0', port= 9999, ssl_context='adhoc')
