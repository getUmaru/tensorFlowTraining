# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 13:35:40 2018

@author: Umar Ali
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.01
epoch = 25
display_step = 1
batch = 100

x = tf.placeholder(tf.float32, [None,5])

tf.print("VAlUE:",x)




