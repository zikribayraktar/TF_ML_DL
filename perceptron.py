# Simple Perceptron in Python
# Course Tutorial selfdrivingcars.mit.edu
# Re-written in Python 3 for practice
#-------------------------------------------

import numpy as np
import numpy.random as nr
#import matplotlib.pyplot as pl
#%matplotlib inline

# Generate some points
N = 10
xn = nr.rand(N,2)

# create equally spaced 50 points between [0,1]
x = np.linspace(0,1, 50)

# Function to pick a line  f(x) = ax+b
a, b = 0.8, 0.2
f = lambda x : a*x + b

# Linearly separate the points by the line:
yn = np.zeros([N,1])
for i in range(N):    #note that xrange() changed to range() in python3
  if(f(xn[i,0]) > xn[i,1]):
    #point below line
    yn[i] = 1
  else:
    #point above line
    yn[i] = -1


#-------------------------------------------	
def perceptron(xn, yn, max_iter=1000, w=np.zeros(3)):
#  	Simple perceptron for 2D data
#		Find the best line to separate
	
  N = xn.shape[0]
  # separating curve
  f = lambda x: np.sign(w[0]+w[1]*x[0]+w[2]*x[1])
  
  for _ in range(max_iter):
    i = nr.randint(N)
    if(yn[i] != f(xn[i,:])):
      w[0] = w[0] + yn[i]
      w[1] = w[1] + yn[i] *xn[i,0]
      w[2] = w[2] + yn[i] *xn[i,1]

  return w
#-------------------------------------------  
 
w = perceptron(xn, yn)
 # use the weights w to compute a,b for a line y=ax+b
anew = -w[0]/w[2]
bnew = -w[1]/w[2]
y = lambda x: anew*x + bnew 

# end-of-file