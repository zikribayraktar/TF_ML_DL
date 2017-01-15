#---------------------------------------------
# This is a TensorFlow Sample Code.
# Module 01 - Linear Regression Example.
# Lectures are from IBM BigData University
# (ML0120EN) Deep Learning with TensorFlow
# Zikri Bayraktar
# Created on Windows 10, Python 3.5(64-bit)
#---------------------------------------------
# import TensorFlow library as tf:
import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
#%matplotlib inline
plt.rcParams['figure.figsize'] = (10,6)

print("Tensorflow version")
print(tf.__version__)

# Define the independent variable X:
X = np.arange(0.0, 5.0, 0.1)

a,b = 1.1, 3
# Define the dependent variable Y:
Y = a*X + b

# Plot the figure for illustration:
plt.plot(X,Y)
plt.ylabel('Dependent Variable Y')
plt.xlabel('Independent Variable X')
plt.show()

# generate random data with linear relation:
x_data = np.random.rand(100).astype(np.float32)
# generate dependent variable data:
y_data = 3*x_data + 2
# add some gaussian noise
y_data = np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=0.1))(y_data)

# Initialize
a = tf.Variable(1.0)
b = tf.Variable(0.2)
y = a*x_data + b

# define loss function, squared error:  Loss = ((Data_meas - Data_sim)^2) / N
loss = tf.reduce_mean(tf.square(y-y_data))

# define Gradient Descent optimizer method:
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Initialize:
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# now start optimization and run the designed graph:
train_data = []
for step in range(100):
  evals = sess.run([train,a,b])[1:]
  if step % 5 == 0:
    print(step, evals)
    train_data.append(evals)

	
# Plot results:
converter = plt.colors
cr, cg, cb = (1.0, 1.0, 0.0)
for f in train_data:
    cb += 1.0 / len(train_data)
    cg -= 1.0 / len(train_data)
    if cb > 1.0: cb = 1.0
    if cg < 0.0: cg = 0.0
    [a, b] = f
    f_y = np.vectorize(lambda x: a*x + b)(x_data)
    line = plt.plot(x_data, f_y)
    plt.setp(line, color=(cr,cg,cb))

plt.plot(x_data, y_data, 'ro')
green_line = mpatches.Patch(color='red', label='Data Points')
plt.legend(handles=[green_line])
plt.show()

# end-of-file