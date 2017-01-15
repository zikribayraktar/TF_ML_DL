#---------------------------------------------
# This is a TensorFlow Sample Code.
# Module 01.
# Lectures are from IBM BigData University
# (ML0120EN) Deep Learning with TensorFlow
# Zikri Bayraktar
# Created on Windows 10, Python 3.5(64-bit)
#---------------------------------------------
# import TensorFlow library as tf:
import tensorflow as tf
print("Tensorflow version")
print(tf.__version__)
#---------------------------------------

# VARIABLES #
# define a variable for a simple counter:
stateValue = tf.Variable(0)
# define the step to increment the counter:
one = tf.constant(1)
# increment the counter:
new_value = tf.add(stateValue, one)
# assign the updated value of counter to variable:
update = tf.assign(stateValue, new_value)
# initialize all variables AFTER having launched the graph:
#init_op = tf.initialize_all_variables()
init_op = tf.global_variables_initializer()

with tf.Session() as ses:
  ses.run(init_op)
  print(ses.run(stateValue))
  for _ in range(3):
    ses.run(update)
    print(ses.run(stateValue))
#---------------------------------------

# PLACEHOLDERS #
# to feed data to TF from outside use placeholders.
# you need to know data types and its precision beforehand.
aa = tf.placeholder(tf.float32)
b=aa*2
with tf.Session() as sess:
  result = sess.run(b, feed_dict={aa:3.5})
  print (result)
  
dictionary ={aa: [[[1,2,3], [4,5,6], [7,8,9]], [[10,11,12], [13,14,15], [16,17,18]]] }
with tf.Session() as sess: 
  # we can also pass a tensor:
  result = sess.run(b, feed_dict=dictionary)
  print (result)
#---------------------------------------
# SCALAR / VECTOR / MATRIX / TENSOR #
# lets source two constants:
a = tf.constant([2])
b = tf.constant([3])

# lets define some arrays/tensors:
vectorData = tf.constant([2,3,4])
matrixData = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
tensorData = tf.constant([[[1,2,3], [4,5,6], [7,8,9]], [[1,2,3], [4,5,6], [7,8,9]], [[1,2,3], [4,5,6], [7,8,9]] ])

# lets print them:
with tf.Session() as sesion:
  results = sesion.run(a)
  print ("Scalar Data \n %s \n" % results)
  results = sesion.run(vectorData)
  print ("Vector Data \n %s \n" % results)
  results = sesion.run(matrixData)
  print ("Matrix Data \n %s \n" % results)
  results = sesion.run(tensorData)
  print ("Tensor Data \n %s \n" % results)
#---------------------------------------
  
# SIMPLE OPERATIONS:  
# Lets try matrix multiplication:
first_op = tf.matmul(matrixData, matrixData)
# Lets try element-wise multiplication:
second_op = tf.multiply(matrixData, matrixData)

with tf.Session() as sesion:
  results = sesion.run(first_op)
  print ("Matrix Multip by TF:")
  print (results)
  results2 = sesion.run(second_op)
  print ("Element-wise Multip by TF:")
  print (results2)
  

#---------------------------------------

# lets apply add operation over these variables:
c = tf.add(a,b) 
# also works:
# c = a+b

# Now, initialize a session to tun the graph:
session = tf.Session()

# Run the session to get the result from the 
# add operation defined previously:
res = session.run(c)
print(res)

# Finally close the session:
session.close()
# To avoid having to close session every time, use
#'with' block, which will automatically close the session.
with tf.Session() as session:
  result=session.run(2*c)
  print(result)

# end-of-file
#------------------------------------------------