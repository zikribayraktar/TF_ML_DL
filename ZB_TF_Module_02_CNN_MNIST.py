#---------------------------------------------
# This is a TensorFlow Sample Code.
# Module 02 - Convolutional NN
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

# Initial parameters:
width  = 28   #input image width in pixels
height = 28   #input image height in pixels
flat = width * height     #total number of pixels
class_output = 10    #number of classification

# Create placeholders for TF input:
x  = tf.placeholder(tf.float32, shape=[None, flat])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])

# Weights Initialization: (avoid zeros, prefer small number)
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# Bias Initialization: (avoid zeros, prefer small positive value to prevent dead neurons)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Convolutional Layers:
def conv2d(x,W):
  return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')

# Max Pooling
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

  
# First Convolutional and Max-pooling Layer:
# size of kernel: 5x5, input channels:1 (greyscale), output feature maps:32
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])   #need 32 biases for 32 outputs

# Converting images of the data set to tensors:
# 28x28 pixels, 1 channel (greyscale)
x_image = tf.reshape(x, [-1,28,28,1])
# Convolve with weight tensor and add biases:
convolve1 = conv2d(x_image, W_conv1) + b_conv1
# Apply ReLU activation operation:
h_conv1 = tf.nn.relu(convolve1)
# Apply max pooling operation
h_pool1 = max_pool_2x2(h_conv1)
# Complete the first layer:
layer1 = h_pool1


# Second Convolutional and Max-pooling Layer:
# size of kernel: 5x5, input channels:32 (from Layer1), output feature maps:64
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
# Convolution operation:
convolve2 = conv2d(layer1, W_conv2) + b_conv2
# ReLU operation:
h_conv2 = tf.nn.relu(convolve2)
# Max pooling operation
h_pool2 = max_pool_2x2(h_conv2)
# Complete 2nd Layer:
layer2 = h_pool2


# Third Layer: Fully Connected (FC) Layer.  Need to use FC to use the softmax
# note that after appying max_pooling twice, input image of 28x28 becomes 7x7, hence the weight variable has 7x7x64 dimension.
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# Flatten 2nd Layer:
layer2_matrix = tf.reshape(layer2, [-1, 7*7*64])
# Apply Weights and Biases
matmul_fc1 = tf.matmul(layer2_matrix, W_fc1) + b_fc1
# ReLU operation:
h_fc1 = tf.nn.relu(matmul_fc1)
# Complete 3rd Layer:
layer3 = h_fc1

# OPTIONAL: Dropout for reducing overfitting:
keep_prob = tf.placeholder(tf.float32)
layer3_drop = tf.nn.dropout(layer3, keep_prob)

# FINAL layer:
# fully connected, 1024 input, 10 output
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# Apply weight and bias
matmul_fc2 = tf.matmul(layer3_drop, W_fc2) + b_fc2
# Apply softmax regression operation:
y_conv = tf.nn.softmax(matmul_fc2)
# Complete 4th layer:
layer4 = y_conv


# Cross Entropy Cost Function operation:
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(layer4), reduction_indices=[1]))

# Define the optimizer:
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Define Prediction:
correct_prediction = tf.equal(tf.argmax(layer4, 1), tf.argmax(y_, 1))

# Define accuracy:
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("Import and Initialization Done")

# Start TF Session:
sess = tf.Session()
# Initialize weights and biases variables:
sess.run( tf.global_variables_initializer() )

# Train using minibatch Gradient Descent:
for i in range(1100):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    acc = sess.run(accuracy, feed_dict={x: batch[0], y_ : batch[1], keep_prob: 1.0})
    print("Step %d. training accuracy %g"%(i, float(acc)))
  sess.run(train_step, feed_dict={x: batch[0], y_ : batch[1], keep_prob:0.5})

sess.close()
# end-of-file