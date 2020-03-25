# to predict if the tumor malignant or benign

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV

# Get the Data
# use the built in breast cancer dataset from Scikit Learn
cancer = load_breast_cancer()

# The data set is presented in a dictionary form
print('Keys: ',cancer.keys())
print('Feature Names: ', cancer['feature_names'])

# Set up DataFram, pass the data and column names
df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
print(df_feat.head(5))
print(df_feat.info())

# the target
print('Target: ',cancer['target'])
print('Target Names: ', cancer['target_names'])

# Train Test Split
# X - features
# y - target
X = df_feat
y = cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Create instance of SVC
model = SVC()

# Train/fit the model
model.fit(X_train,y_train)

# predict off of X_test
predictions = model.predict(X_test) # for default parameters

# Evaluate the model, pass true values and predictions to classification_report
print('Classifiction Report:\n',classification_report(y_test,predictions))
print('\nConfusion Matrix:\n',confusion_matrix(y_test,predictions))

# Gridsearch to find the right vallue for parameters ( C and gamma)
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} # add linear to the list and check the confusion_matrix

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
print(f'SVM model with default parameters : {model.score(X_test, y_test)}')
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
