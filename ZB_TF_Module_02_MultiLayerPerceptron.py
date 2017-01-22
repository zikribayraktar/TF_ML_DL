#---------------------------------------------
# This is a TensorFlow Sample Code.
# Module 02 - Multilayer Perceptron.
# MNIST data set used.
# Lectures are from IBM BigData University
# (ML0120EN) Deep Learning with TensorFlow
# Zikri Bayraktar
# Created on Windows 10, Python 3.5(64-bit)
#---------------------------------------------
# Import Tensorflow and MNIST data set:
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Create placeholders for TF input:
# (input images(2D) are 28x28 pixels, i.e. 784 pixels in 1D vector form.)
x  = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Bias and Weights Initialization:
# Weight tensor:
W = tf.Variable(tf.zeros([784,10], tf.float32))
# Bias tensor:
b = tf.Variable(tf.zeros([10], tf.float32))

# Apply softmax regression operation:
y = tf.nn.softmax(tf.matmul(x,W)+b)

# Cross Entropy Cost Function operation:
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Gradient Descent minimization operation:
# use learning rate of 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

print("Import and Initialization Done")

# Start TF Session:
sess = tf.Session()
# Initialize weights and biases variables:
sess.run( tf.global_variables_initializer() )

# Train using minibatch Gradient Descent:
for i in range(1000):
  batch = mnist.train.next_batch(50)
  sess.run(train_step, feed_dict={x: batch[0], y_ : batch[1]})

# Test:
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_ : mnist.test.labels}) *100
print("The final accuracy for the simple ANN model is: {} %".format(acc))

sess.close()
# end-of-file