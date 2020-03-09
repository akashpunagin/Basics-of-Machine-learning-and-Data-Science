import numpy as np
arr = np.arange(0,10)

print(arr + arr)
print(arr * arr)
print(arr - arr)

# Warning on division by zero, but not an error! Just replaced with nan
print(arr/arr)

# Also warning, but not an error instead infinity
print(1/arr)

print(arr**3)

# Universal Array Functions

#Taking Square Roots
print(np.sqrt(arr))

#Calcualting exponential (e^)
print(np.exp(arr))

print(np.max(arr)) #same as arr.max()
print(np.sin(arr))
print(np.log(arr))
