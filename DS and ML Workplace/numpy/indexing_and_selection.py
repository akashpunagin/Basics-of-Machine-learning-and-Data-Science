import numpy as np

#Creating sample array
arr = np.arange(0,11)
print(arr)

#Get a value at an index
print(arr[8])

#Get values in a range
print(arr[1:5]) # inclusive of 1, exclusive of 5

#Get values in a range
print(arr[0:5])

# Broadcasting
#Setting a value with index range (Broadcasting)
arr[0:5]=100
print(arr)

# Reset array
arr = np.arange(0,11)

# Slicing
slice_of_arr = arr[0:6]
print("here",arr)
print(slice_of_arr)

# Change Slice
slice_of_arr[:]=99
print(slice_of_arr)
print(arr)
print("Data is not copied, it's a view of the original array! This avoids memory problems!")

#To get a copy, need to be explicit
arr_copy = arr.copy()
print(arr_copy)


# Indexing a 2D array (matrices)
# The general format is arr_2d[row][col] or arr_2d[row,col]
arr_2d = np.array(([5,10,15],[20,25,30],[35,40,45]))
print(arr_2d)

#Indexing row
print(arr_2d[1])

# Getting individual element value
print(arr_2d[1,0])

# 2D array slicing
print(arr_2d[1:,:2])

#Shape bottom row
print(arr_2d[2])

#Shape bottom row
print(arr_2d[2,:])

# Selection
# based off of comparison operators.

arr = np.arange(1,11)
print("here",arr)
print(arr > 4)
bool_arr = arr>4
print(bool_arr)
print(arr[bool_arr])
print(arr[arr>2])
x = 2
print(arr[arr>x])
