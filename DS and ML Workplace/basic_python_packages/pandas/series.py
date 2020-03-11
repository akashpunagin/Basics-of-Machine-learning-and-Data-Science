import numpy as np
import pandas as pd

my_list = [10,20,30]
labels = ['a','b','c']
arr = np.array([10,20,30])
d = {'a':10,'b':20,'c':30}

# Using lists
print(pd.Series(data=my_list))
print(pd.Series(data=my_list, index=labels))

# NumPy arrays
print(pd.Series(data=arr))
print(pd.Series(data=arr, index=labels))

# Python dictionary
print(pd.Series(d))

# Using the index
ser1 = pd.Series([1,2,3,4],index = ['USA', 'Germany','USSR', 'Japan'])
ser2 = pd.Series([1,2,5,4],index = ['USA', 'Germany','Italy', 'Japan'])

print(ser1)
print(ser2)

print(ser1['USA'])

print(ser1 + ser2) # NaN means the index is not common in both Series
