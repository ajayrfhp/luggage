from app import app
from flask import render_template, request
import json
import numpy as np
import conv_model.prediction as prediction

@app.route('/')
@app.route('/index', methods = ['GET','POST'])
def index():
	message = ''
	return render_template('index.html', message = message)

@app.route('/predict',methods = ['POST','GET'])
def predict():
	data = request.values.dicts[1]['data']
	data = str(data)
	data = data.replace('[','')
	data = data.replace(']','')
	data = data.split(',')
	#data = data.encode('ascii','ignore')
	data = np.array(data)
	#return render_template('index.html', message = 'abd' )
	return str(prediction.predict_class(data))


