import numpy as np

my_list = [1,2,3]
print(my_list)
print(np.array(my_list)) # array to numpy array

my_matrix = [[1,2,3],[4,5,6],[7,8,9]]
print(np.array(my_matrix))



# Return evenly spaced values within a given interval.
print(np.arange(0,10))
print(np.arange(0,10,2))

# Generate arrays of zeros or ones
print(np.zeros(3))
print(np.zeros((2,4)))
print(np.ones(6))
print(np.ones((6,4)))

# Return evenly spaced numbers over a specified interval.
print(np.linspace(2,10,8)) # including 2 and 10

# Create on identity matrix
print(np.eye(3))

# Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1)
print(np.random.rand(3))
print(np.random.rand(3,3))

# Return a sample (or samples) from the "standard normal" distribution
print(np.random.randn(3))
print(np.random.randn(3,3))

# Return random integers from low (inclusive) to high (exclusive)
print(np.random.randint(1,100,5))

# Array Methods
arr = np.arange(25)
ranarr = np.random.randint(0,50,10)

print(arr.reshape(5,5))

# These are useful methods for finding max or min values. Or to find their index locations using argmin or argmax
print(ranarr)
print(ranarr.max())
print(ranarr.argmax())
print(ranarr.min())
print(ranarr.argmin())

# Shapa attribute
print(arr.shape)
