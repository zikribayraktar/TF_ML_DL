#---------------------------------------------
# This is a TensorFlow Sample Code.
# Module 01 - Activation Functions.
# Lectures are from IBM BigData University
# (ML0120EN) Deep Learning with TensorFlow
# Zikri Bayraktar
# Created on Windows 10, Python 3.5(64-bit)
#---------------------------------------------
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Start a TF session:
sess = tf.Session()
# Simple input of 3 real values:
i = tf.constant([1.0, 2.0, 3.0], shape=[1,3])
# Weights; a 3x3 matrix:
w = tf.random_normal(shape=[3,3])
# Biases; a 1x3 vector:
b = tf.random_normal(shape=[1,3])
# a dummy activation function:
def func(x): return x
# compute the W*i+b
act = func(tf.matmul(i,w) + b)
# evaluate the tensor to a numpy array
act.eval(session=sess)

# using Sigmoid Activation Function in NN:
act = tf.sigmoid(tf.matmul(i,w)+b)
act.eval(session=sess)

# using Hyperbolic Tangent (tanh) in NN:
act = tf.tanh(tf.matmul(i,w)+b)
act.eval(session=sess)

# using Rectified Linear Unit (ReLU) in NN:
act = tf.nn.relu(tf.matmul(i,w)+b)
act.eval(session=sess)



sess.close()
# end-of-file