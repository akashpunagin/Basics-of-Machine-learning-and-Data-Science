# The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor)
# so 150 total samples. Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV

# Get the data
iris = sns.load_dataset('iris')
print(iris.head(5))
print(iris.info())

# Exploratory Data Analysis
# sns.pairplot(iris,hue='species',palette='Dark2')
# plt.show()
# by looking at the graph Setosa is the most separable.

# Create a kde plot of sepal_length versus sepal_width for only setosa species of flower.
# setosa = iris[iris['species']=='setosa']
# sns.kdeplot(setosa['sepal_width'],setosa['sepal_length'],cmap="plasma", shade=True, shade_lowest=False)
# plt.show()

# Train Test Split
# X - features
# y - target
X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Create instance of SVC
svc_model = SVC()

# Train/fit the model
svc_model.fit(X_train,y_train)

# predict off of X_test
predictions = svc_model.predict(X_test)

# Evaluate the model, pass true values and predictions to classification_report
print('Classifiction Report:\n',classification_report(y_test,predictions))
print('\nConfusion Matrix:\n',confusion_matrix(y_test,predictions))

# Gridsearch to find the right vallue for parameters ( C and gamma)
# param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001], 'kernel': ['rbf', 'linear']}
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}

# GridSearchCV takes an estimator and creates an new parameter (ie. meta-estimator)
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
# first it runs the same loop with cross-validation to find the best parameter combination.
# Once it has the best combination, it runs fit again on all data passed to fit (without cross-validation),
# to built a single new model using the best parameter setting.

# fit / train the GridSearchCV instance
grid.fit(X_train,y_train)

# Best combination of Parameters
print('\nBest Parameters : ',grid.best_params_)

# Use best model out of GridSearchCV
best_model = grid.best_estimator_

# predic  of best model
grid_predictions = best_model.predict(X_test)

# Evaluate the model, pass true values and predictions to classification_report
print('Classifiction Report:\n',classification_report(y_test,grid_predictions))
print('\nConfusion Matrix:\n',confusion_matrix(y_test,grid_predictions))

print('\nFinal Scores of different models:')
print(f'SVM model with default parameters : {svc_model.score(X_test, y_test)}')
print(f'SVM model with optimal parameters : {best_model.score(X_test, y_test)}')


# Parameters :
# C :
# It controls the trade off between smooth decision boundary and classifying training points correctly
# A large value of c means you will get more training points correctly.
# Gamma:
# It defines how far the influence of a single training example reaches
# If it has a low value it means that every point has a far reach and
# conversely high value of gamma means that every point has close reach.
# kernel: rbf (Radial basis function), linear, Polynomial, Sigmoid
# It specifies the kernel type to be used in the algorithm
