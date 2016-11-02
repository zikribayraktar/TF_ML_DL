# This script is for practicing some of the numpy/scipy/pandas/matplotlib commands
# Zikri Bayraktar

# import some of the useful modules that we need:
import numpy as np
#import scipy as sp
from datetime import datetime
t0 = datetime.now() #save current time

# make a list
newList1 = [1,2,3,4,5]
# append to the list
newList1.append(6)
newList1 = newList1 + [7]

# make an array
newArray1 = np.array([1.0,2,3,4])
# note that the first element is double, hence the array is float64 type

# many of the functions in NP are elementwise
newArray2 = 2*newArray1 + newArray1  #multiply by scalar, add other vector
newArray3 = newArray2 ** 2 #raise to a power
np.sqrt(newArray3)  #square root func
np.log(newArray3)   #natural log func
np.exp(newArray3)   #exponential func
    
# dot product of vectors
np.dot(newArray2, newArray3) #dot product
newArray2.dot(newArray3)  #same dot product
newArray3.dot(newArray2)  #same dot product

# norm - ecludian -L2norm
magn = np.linalg.norm(newArray1)

# random floats in the half-open interval [0.0, 1.0)
np.random.random(8)  #unifromly distributed random numbers 
# normally distribured random numbers
a = np.random.randn(100)
a.mean()  #mean of the vector a
a.var()   #variance of the vector a

# create a 2D matrix
M = np.array([ [1,2], [3,4] ])
# transpose the matrix
M.transpose()
M.T
# element-wise multiplication of two matrices
M*M  #gives element-wise multiplication result
M.dot(M) #gives matrix-multiplication result
Minv = np.linalg.inv(M) #gives inverse of matrix
np.linalg.det(M)  #gives the determinant of matrix
np.diag(M)  #gives the matrix diagonal
np.diag([1,5])  #this creates a matrix of zeros with input assigned to diagonals.
np.trace(M)  #sum of diagonal elements
np.cov(M.T)  #gives the covariance, note the transpose 
np.linalg.eig(M)  # gives eigenvalue and eigenvectors

#Solve Ax=b --> inv(A)*Ax = inv(A)*b --> x =inv(A)*b
a = np.array([ [1,2] , [3,4] ])
b = np.array([3,4])
x = np.linalg.inv(a).dot(b)
x = np.linalg.solve(a,b)  #this is a better method

#vector and matrix of zeros/ones
np.zeros(5)      #vector of zeros
np.zeros((6,6))  #matrix of zeros
np.ones(4)       #vector of ones
np.ones((3,3))   #matrix of ones

# to read float-data in a CSV file line-by-line
X=[] #read-into a list
for line in open("data.csv"):
    row = line.split(',')
    val = map(float,row)
    X.append(val)
X=np.array(X) #convert list to array
X.shape  #size of the matrix


# PANDAS
import pandas as pd
#use pandas to load the data as dataframe
X = pd.read_csv("data.csv", header=None)
type(X)  #type of the variable
X.info()  #summary of the dataframe -- similar to R
X.head()  #top entries of the dataframe
# we can convert the dataframe into numpy array
M = X.as_matrix()
type(M)  #this will show as numpy.ndarray

#Note the following difference
# Numpy  returns : X[0] -> 0th row
# Pandas returns : X[0] -> column that has name 0
# both are different than Matlab syntax, where numpy might be closest.
type(X[0])  #returns pandas.core.series.Series

#to access a row in pandas dataframe
X.iloc[0]  # will return the 0th row of the dataframe
X.ix[0]   #will also return the 0th row of the dataframe
type(X.ix[0])  #returns pandas.core.series.Series




# end-of-file