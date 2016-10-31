# This script is for practicing some of the numpy/scipy commands
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
    
# dot product
np.dot(newArray2, newArray3) #dot product
newArray2.dot(newArray3)  #same dot product
newArray3.dot(newArray2)  #same dot product

# norm - ecludian -L2norm
magn = np.linalg.norm(newArray1)

# normally distribured random numbers
a = np.random.randn(100)


