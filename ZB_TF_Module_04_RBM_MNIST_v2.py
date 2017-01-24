#---------------------------------------------
# This is a TensorFlow Sample Code.
# Module 04 - Restricted Boltzman Machine
# MNIST data set used.
# Lectures are from IBM BigData University
# (ML0120EN) Deep Learning with TensorFlow
# Zikri Bayraktar
# Created on Windows 10, Python 3.5(64-bit)
#---------------------------------------------
import urllib.request
response = urllib.request.urlopen('http://deeplearning.net/tutorial/code/utils.py')
content = response.read()
target = open('utils.py', mode='w')
target.write( str(content, 'utf-8') )
target.close()

# Import Tensorflow and MNIST data set:
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import image
from utils import tile_raster_images
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# Initial parameters:
width  = 28   #input image width in pixels
height = 28   #input image height in pixels
flat = width * height     #total number of pixels
hiddenN = 500  #number of hidden neurons
class_output = 10    #number of classification

# Create placeholders for TF input:
vb = tf.placeholder(tf.float32, [flat])  #visual layer bias
hb = tf.placeholder(tf.float32, [hiddenN])  #hidden layer bias
W  = tf.placeholder(tf.float32, [flat, hiddenN])

# vO would be the input vector to the visible layer:
vO = tf.placeholder(tf.float32, [None, flat])
# Compute the sigmoid activation using the input and weights and add hidden layer bias:
_h0= tf.nn.sigmoid(tf.matmul(vO, W) + hb)  #probabilities of the hidden units
# Then compute 0 or 1, binary state for the hidden units, i.e. h0 is binary 0 or 1:
h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0)))) #sample_h_given_X

# 2. Backward Pass:
_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb) 
v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1)))) #sample_v_given_h
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

alpha = 1.0
w_pos_grad = tf.matmul(tf.transpose(vO), h0)
w_neg_grad = tf.matmul(tf.transpose(v1), h1)
CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(vO)[0])
update_w = W + alpha * CD
update_vb = vb + alpha * tf.reduce_mean(vO - v1, 0)
update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

#calculate error
err = tf.reduce_mean(tf.square(vO - v1))

cur_w = np.zeros([784, 500], np.float32)
cur_vb = np.zeros([784], np.float32)
cur_hb = np.zeros([500], np.float32)
prv_w = np.zeros([784, 500], np.float32)
prv_vb = np.zeros([784], np.float32)
prv_hb = np.zeros([500], np.float32)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

sess.run(err, feed_dict={vO: trX, W: prv_w, vb: prv_vb, hb: prv_hb})

#Parameters
epochs = 5
batchsize = 100
weights = []
errors = []

# run epochs in batches:
for epoch in range(epochs):
    for start, end in zip( range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
        batch = trX[start:end]
        cur_w = sess.run(update_w, feed_dict={vO: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_vb = sess.run(update_vb, feed_dict={  vO: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_hb = sess.run(update_hb, feed_dict={ vO: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb
        if start % 10000 == 0:
            errors.append(sess.run(err, feed_dict={vO: trX, W: cur_w, vb: cur_vb, hb: cur_hb}))
            weights.append(cur_w)
    print ('Epoch: %d' % epoch,'reconstruction error: %f' % errors[-1])
plt.plot(errors)
plt.xlabel("Batch Number")
plt.ylabel("Error")
plt.show()

uw = weights[-1].T
print (uw) # a weight matrix of shape (500,784)

# end-of-file