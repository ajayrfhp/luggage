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
	data = str(request.args.get('data'))
	data = json.loads(data)
	data = np.array(data)
	#return render_template('index.html', message = 'abd' )
	return str(prediction.predict_class(data))


