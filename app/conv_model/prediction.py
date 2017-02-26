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
# RESTORE TENSORFLOW MODELS
# READ AND PROCESS IMAGE
# PREDICT OUTPUT

def predict_class(img):
	sess = tf.Session()
	# RESTORE TENSORFLOW MODEL AND DATA
	from nn import *
	rel_path = os.path.dirname(os.path.abspath(__file__))
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	saver.restore(sess,rel_path +"/model3/model3.ckpt")
	#saver = tf.train.import_meta_graph(rel_path + '/data/cnn-2.ckpt.meta')
	#saver.restore(sess, rel_path + "/data/cnn-2.ckpt")
	random_state = 42
	tf.set_random_seed(random_state)
	y = {0:'backpacks', 1: 'bags', 2:'luggage',3:'travel_accessoires'}

	
	# READ AND PROCESS IMAGE
	img = np.reshape(img,(64,64,4))
	img = img[:,:,:3]
	img = scipy.misc.imresize(img,(64,64,3))
	img = np.mean(img, axis = -1)
	img = img.reshape((1,img.shape[0] * img.shape[1]))
	img = preprocessing.normalize(img)

	# CALCULUATE PREDICTION
	pred = y_conv.eval(session = sess,feed_dict={x:img})
	sess.close()

	
	for i in range(4):
		if(pred[0][i] == max(pred[0])):
			return y[i]

