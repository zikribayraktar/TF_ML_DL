#---------------------------------------------
# This is a TensorFlow Sample Code.
# Module 01 - Logistic Regression Example.
# Lectures are from IBM BigData University
# (ML0120EN) Deep Learning with TensorFlow
# Zikri Bayraktar
# Created on Windows 10, Python 3.5(64-bit)
#---------------------------------------------
import numpy as np
import tensorflow as tf
import pandas as pd
import time
import scipy
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

iris = load_iris()
iris_X, iris_y = iris.data[:-1,:], iris.target[:-1]
iris_y = pd.get_dummies(iris_y).values
trainX, testX, trainY, testY = train_test_split(iris_X, iris_y, test_size=0.33, random_state=42)

# number of features of input:
numFeatures = trainX.shape[1]
# number of classes:
numLabels = trainY.shape[1]

# Create Placeholders for TF:
# 'None' indicates TF should not expect fixed number in that dimension
# Iris data has 4 features (columns), so should X tensor
X = tf.placeholder(tf.float32, [None, numFeatures])
yGold = tf.placeholder(tf.float32, [None, numLabels])

# initialize Weight and bias:
W = tf.Variable(tf.zeros([4,3]))  #4-dim input and 3 classes
b = tf.Variable(tf.zeros([3]))

weights = tf.Variable(tf.random_normal([numFeatures, numLabels], mean=0, stddev=0.01, name="weights"))
bias = tf.Variable(tf.random_normal([1,numLabels], mean=0, stddev=0.01, name="bias"))

# Define Logistic Regression in 3 operation steps:
apply_weights_OP = tf.matmul(X, weights, name = "apply_weights")							
add_bias_OP = tf.add(apply_weights_OP, bias, name = "add_bias")
activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")

# Cost Function:
numEpochs = 1000
# learning rate:
learningRate = tf.train.exponential_decay(learning_rate=0.0008, global_step=1, decay_steps=trainX.shape[0], decay_rate=0.95, staircase=True)
# cost function -- squared mean error
cost_OP = tf.nn.l2_loss(activation_OP-yGold, name="squared_error_cost")
# gradient descent:
training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)

# TF Session:
sess = tf.Session()
# Initialize weights and biases variables:
init_OP = tf.global_variables_initializer()
sess.run(init_OP)

# additional operations to keep track of progress:
# check if correct labeling by logical comparison:
correct_predictions_OP = tf.equal(tf.argmax(activation_OP,1),tf.argmax(yGold,1))
# if every false prediction is 0 and true predictions are 1, then then average returns us the accuracy
accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))
# Summary op for regression output
activation_summary_OP = tf.histogram_summary("output", activation_OP)

# Summary op for accuracy
accuracy_summary_OP = tf.scalar_summary("accuracy", accuracy_OP)

# Summary op for cost
cost_summary_OP = tf.scalar_summary("cost", cost_OP)

# Summary ops to check how variables (W, b) are updating after each iteration
weightSummary = tf.histogram_summary("weights", weights.eval(session=sess))
biasSummary = tf.histogram_summary("biases", bias.eval(session=sess))

# Merge all summaries
merged = tf.merge_summary([activation_summary_OP, accuracy_summary_OP, cost_summary_OP, weightSummary, biasSummary])

# Summary writer
writer = tf.train.SummaryWriter("summary_logs", sess.graph_def)

# Initialize reporting variables
cost = 0
diff = 1
epoch_values = []
accuracy_values = []
cost_values = []

# Training epochs
for i in range(numEpochs):
    if i > 1 and diff < .0001:
        print("change in cost %g; convergence."%diff)
        break
    else:
        # Run training step
        step = sess.run(training_OP, feed_dict={X: trainX, yGold: trainY})
        # Report occasional stats
        if i % 10 == 0:
            # Add epoch to epoch_values
            epoch_values.append(i)
            # Generate accuracy stats on test data
            train_accuracy, newCost = sess.run([accuracy_OP, cost_OP], feed_dict={X: trainX, yGold: trainY})
            # Add accuracy to live graphing variable
            accuracy_values.append(train_accuracy)
            # Add cost to live graphing variable
            cost_values.append(newCost)
            # Re-assign values for variables
            diff = abs(newCost - cost)
            cost = newCost

            #generate print statements
            print("step %d, training accuracy %g, cost %g, change in cost %g"%(i, train_accuracy, newCost, diff))


# How well do we perform on held-out test data?
print("final accuracy on test set: %s" %str(sess.run(accuracy_OP, 
                                                     feed_dict={X: testX, 
                                                                yGold: testY})))


sess.close()
# end-of-file