import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,5,11)
y = x ** 2

plt.scatter(x,y)
plt.show()

from random import sample
data = sample(range(1, 1000), 5)
print(data)
plt.hist(data)
plt.show()
