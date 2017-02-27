#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy.misc
from os import walk
import numpy as np
import os
import sys
import tensorflow as tf
import collections
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing, svm
from sklearn.metrics import confusion_matrix
import random
from sklearn.externals import joblib
# RESTORE TENSORFLOW MODELS
# READ AND PROCESS IMAGE
# PREDICT OUTPUT

def predict_class(img):
	img = [ int(a) for a in img ]
	sess = tf.Session()
	# RESTORE TENSORFLOW MODEL AND DATA
	from nn import *
	rel_path = os.path.dirname(os.path.abspath(__file__))
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver(tf.trainable_variables())
	saver.restore(sess,rel_path +"/model4/model4.ckpt")
	#saver = tf.train.import_meta_graph(rel_path + '/data/cnn-2.ckpt.meta')
	#saver.restore(sess, rel_path + "/data/cnn-2.ckpt")
	random_state = 42
	tf.set_random_seed(random_state)
	y = {0:'backpacks', 1: 'bags', 2:'luggage',3:'travel_accessoires'}

	luggage_inverse = {0: 'american-tourister', 1: 'eastpak', 2: 'epic', 3: 'hartmann', 4: 'lufthansa', 5: 'montblanc', 6: 'rimowa', 7: 'samsonite', 8: 'stratic', 9: 'thule', 10: 'titan', 11: 'travelite', 12: 'victorinox', 13: 'wenger'}
	bag_inverse = {0: 'bag-to-life', 1: 'braun', 2: 'eastpak', 3: 'harolds', 4: 'hartmann', 5: 'leonhard-heyden', 6: 'lufthansa', 7: 'montblanc', 8: 'rimowa', 9: 'samsonite', 10: 'travelite', 11: 'unnamed', 12: 'victorinox', 13: 'wenger'}
	backpack_inverse = {0: 'aigner', 1: 'bag-to-life', 2: 'braun', 3: 'eastpak', 4: 'laessig', 5: 'lufthansa', 6: 'samsonite', 7: 'tatonka', 8: 'thule', 9: 'travelite'}
	luggage = joblib.load(rel_path + '/model4/luggage.pkl')
	bag = joblib.load(rel_path + '/model4/bags.pkl')
	backpacks = joblib.load(rel_path + '/model4/backpacks.pkl')	


	# READ AND PROCESS IMAGE
	img = np.reshape(img,(64,64,4))
	img = img[:,:,:3]
	img = np.mean(img, axis = -1)
	img = img.reshape((1,img.shape[0] * img.shape[1]))
	img = preprocessing.normalize(img)

	# CALCULUATE PREDICTION
	pred = y_conv.eval(session = sess,feed_dict={x:img})
	sess.close()
	brand = ''
	for i in range(4):
		if(pred[0][i] == max(pred[0])):
			if(y[i] == 'luggage'):
                        	brand = luggage_inverse[luggage.predict(img)[0]]
                	if(y[i] == 'bags'):
                        	brand = bag_inverse[bag.predict(img)[0]]
                	if(y[i] == 'backpacks'):
                        	brand = backpack_inverse[backpacks.predict(img)[0]]
			return 'category = ' + y[i] + ' brand = ' + brand

