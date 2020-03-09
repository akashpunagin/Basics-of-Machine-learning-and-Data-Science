import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 5, 11)
y = x ** 2
print(x)
print(y)

# plt.plot(x, y, 'r') # 'r' is the color red
# plt.xlabel('X Axis Title Here')
# plt.ylabel('Y Axis Title Here')
# plt.title('String Title Here')
# plt.show()

# plt.subplot(nrows, ncols, plot_number)
# plt.subplot(1,2,2)
# plt.plot(x, y, 'r--') # More on color options later
# plt.subplot(1,2,1)
# plt.plot(y, x, 'g*-');
# plt.show()

# Object Oriented Method
# Create Figure (empty canvas)
fig = plt.figure()

# Add set of axes to figure
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)

# Plot on that set of axes
axes.plot(x, y, 'b')
axes.set_xlabel('Set X Label') # Notice the use of set_ to begin methods
axes.set_ylabel('Set y Label')
axes.set_title('Set Title')
# plt.show()

# Creates blank canvas
fig = plt.figure()

axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) # inset axes

# Larger Figure Axes 1
axes1.plot(x, y, 'b')
axes1.set_xlabel('X_label_axes2')
axes1.set_ylabel('Y_label_axes2')
axes1.set_title('Axes 2 Title')

# Insert Figure Axes 2
axes2.plot(y, x, 'r')
axes2.set_xlabel('X_label_axes2')
axes2.set_ylabel('Y_label_axes2')
axes2.set_title('Axes 2 Title');
# plt.show()


# subplots() # The plt.subplots() object will act as a more automatic axis manager.
# Use similar to plt.figure() except use tuple unpacking to grab fig and axes
fig, axes = plt.subplots()

# Now use the axes object to add stuff to plot
axes.plot(x, y, 'r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('title');
# plt.show()


fig, axes = plt.subplots(nrows=1, ncols=2)
for ax in axes:
    ax.plot(x, y, 'g')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('title')
plt.tight_layout()
# plt.show()




# Matplotlib allows the aspect ratio, DPI and figure size to be specified when the Figure object is created. You can use the figsize and dpi keyword arguments.
# figsize is a tuple of the width and height of the figure in inches
# dpi is the dots-per-inch (pixel per inch).

fig = plt.figure(figsize=(8,4), dpi=100)
fig.add_axes([0.1, 0.1, 0.8, 0.8])
# plt.show()

fig, axes = plt.subplots(figsize=(12,3))
axes.plot(x, y, 'r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('title');
fig.savefig("filename.png", dpi=200) # dpi - optional
# plt.show()

# adding Legends
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax.plot(x, x**2, label="x**2")
ax.plot(x, x**3, label="x**3")
ax.legend()
# plt.show()

# If legend is overlapping the graph
# ax.legend(loc=1) # upper right corner
# ax.legend(loc=2) # upper left corner
# ax.legend(loc=3) # lower left corner
# ax.legend(loc=4) # lower right corner
ax.legend(loc=0) # let matplotlib decide the optimal location
# plt.show()
