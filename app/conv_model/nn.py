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

def weight_variable(shape):
        # WRAPPER OVER WEIGHT VARIABLE WITH NORMAL INITIALIZATION
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)

def bias_variable(shape):
        # WRAPPER OVER BIAS VARIABLE
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)

def conv_layer(x,W):
        # WRAPPER OVER CONVOLUTIONAL LAYER
        return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
        # WRAPPER OVER MAXPOOL LAYER
        return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding='SAME')


x = tf.placeholder(tf.float32, shape=[None, 64 * 64])
y_ = tf.placeholder(tf.float32, shape=[None, 4])

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])


W_fc = weight_variable([16*16*64, 1024])
b_fc = bias_variable([1024])

x_image = tf.reshape(x,[-1,64,64,1])

# FIRST CONVOLUTIONAL LAYER
h_conv1 = tf.nn.relu(conv_layer(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
h_pool1 = tf.nn.dropout(h_pool1, 0.5)

# SECOND CONVOLUTIONAL LAYER
h_conv2 = tf.nn.relu(conv_layer(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2 = tf.nn.dropout(h_pool2, 0.5)


# FC1
h_pool2_flat = tf.reshape(h_pool2,[-1,16*16*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)
h_fc1 = tf.nn.dropout(h_fc1, 0.5)

# FC2
W_fc2 = weight_variable([1024,4])
b_fc2 = bias_variable([4])
y_conv = tf.matmul(h_fc1,W_fc2) + b_fc2



correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#type cast to float
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


