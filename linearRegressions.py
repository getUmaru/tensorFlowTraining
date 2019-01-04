# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 14:02:34 2018

@author: Umar Ali
"""
## LINEAR REGRESSION ##

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Parameters 
learningRate = 0.01
epochs = 10
display_step = 50

#Training Data
arrayX = np.random.normal(5,5,100)
arrayY = np.random.normal(5,5,100)


trainingX = np.asarray(arrayX)
trainingY = np.asarray(arrayY)

nSamples = trainingX.shape[0]
 
# tf Graph inputs
X = tf.placeholder("float")
Y = tf.placeholder("float")

# model weights and bias'
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

# ACTAUL MODEL CONSTRUCTION #
model = tf.add(tf.multiply(X,W), b)
    # y=wx+b

squareError = tf.reduce_sum(tf.pow(model-Y, 2))/(2*nSamples)
gradDescent = tf.train.GradientDescentOptimizer(learningRate).minimize(squareError)


init = tf.global_variables_initializer()

# Start training
sess = tf.Session()
sess.run(init)

# Fit all training data
for epoch in range(epochs):
    for (x, y) in zip(trainingX, trainingY):
        sess.run(gradDescent, feed_dict={X: x, Y: y})

#Graphic display
plt.plot(trainingX, trainingY, 'ro', label='Original data')
plt.plot(trainingX, sess.run(W) * trainingX + sess.run(b), label='Fitted line')
plt.legend()
plt.show()


