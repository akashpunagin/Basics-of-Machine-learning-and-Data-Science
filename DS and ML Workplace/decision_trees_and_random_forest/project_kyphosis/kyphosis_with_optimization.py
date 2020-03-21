import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Get the data
df = pd.read_csv('kyphosis.csv')

print(df.head())

# convert target into binary form
print(df['Kyphosis'].value_counts())
df = pd.get_dummies(df,columns=['Kyphosis'],drop_first=True)
print("DataFrame with dummy values for Kyphosis:\n",df.head())

# Exploratory Data Analysis
sns.pairplot(df,hue='Kyphosis_present',palette='Set1')
# plt.show()

# Training and Testing Data
# X - independent variable, features
# y - dependent variable, target
X = df.drop('Kyphosis_present',axis=1)
y = df['Kyphosis_present']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# Decision Trees
print("\n### DECISION TREE ###\n")

# Create instance of DecisionTreeClassifier
dtree = DecisionTreeClassifier()

# training a single decision tree.
dtree.fit(X_train,y_train)

# Predict values from X_test
predictions = dtree.predict(X_test)

# Evaluate the model, pass true values and predictions to classification_report
print('Classifiction Report:\n',classification_report(y_test,predictions))
print('\nConfusion Matrix:\n',confusion_matrix(y_test,predictions))
print(f'Model Accuracy: {dtree.score(X_test, y_test)}')

# Features and their importance
# features considered most important by the Decision Tree
print("\nFeatures and Importance:")
feat_imp = pd.DataFrame({'feature': list(X_train.columns),'importance': dtree.feature_importances_}).sort_values('importance', ascending = False)
print(feat_imp)


# Random Forests
print("\n### RANDOM FOREST ###\n")
# compare the decision tree model to a random forest.

# Create instance of RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)

# train / fit the RandomForestClassifier instance
rfc.fit(X_train, y_train)

# Predict values from X_test
rfc_pred = rfc.predict(X_test)

# Evaluate the model, pass true values and predictions to classification_report
print('Classifiction Report:\n',classification_report(y_test,rfc_pred))
print('\nConfusion Matrix:\n',confusion_matrix(y_test,rfc_pred))
print(f'Model Accuracy: {rfc.score(X_test, y_test)}')

# Features and their importance
# features considered most important by the Decision Tree
print("\nFeatures and Importance:")
feat_imp = pd.DataFrame({'feature': list(X_train.columns),'importance': rfc.feature_importances_}).sort_values('importance', ascending = False)
print(feat_imp)


# Random Forest Optimization through Random Search
# to maximize the performance of the random forest, we can perform a random search for better hyperparameters
# This will randomly select combinations of hyperparameters from a grid,
# evaluate them using cross validation on the training data, and return the values that perform the best

# Hyperparameter grid
param_grid = {
    'n_estimators': np.linspace(10, 200).astype(int),
    'max_depth': [None] + list(np.linspace(3, 20).astype(int)),
    'max_features': ['auto', 'sqrt', None] + list(np.arange(0.5, 1, 0.1)),
    'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
    'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False]
}

# Estimator for use in random search
estimator = RandomForestClassifier()

# Create the random search model
rs = RandomizedSearchCV(estimator, param_grid, n_jobs = -1,
                        scoring = 'roc_auc', cv = 3,
                        n_iter = 10, verbose = 1)

# Fit / train RandomizedSearchCV instancce
rs.fit(X_train, y_train)

print(f'The list of best parameters:\n {rs.best_params_}')

# Use the best model out of the RandomizedSearchCV instance
best_model = rs.best_estimator_
print('### BEST MODEL OF RANDOM FOREST ###\n')

# train / fit the best model
best_model.fit(X_train, y_train)

# Predict values from X_test off of best_model
best_model_pred = best_model.predict(X_test)

# Evaluate the best model, pass true values and predictions to classification_report
print('Classifiction Report:\n',classification_report(y_test,best_model_pred))
print('\nConfusion Matrix:\n',confusion_matrix(y_test,best_model_pred))
print(f'Model Accuracy: {best_model.score(X_test, y_test)}')

# Features and their importance
# features considered most important by the Decision Tree
print("\nFeatures and Importance:")
feat_imp = pd.DataFrame({'feature': list(X_train.columns),'importance': best_model.feature_importances_}).sort_values('importance', ascending = False)
print(feat_imp)

print('\nFinal Scores of different models:')
print(f'Decision Tree : {dtree.score(X_test, y_test)}')
print(f'Random Forest : {rfc.score(X_test, y_test)}')
print(f'Best Model by RandomizedSearchCV : {best_model.score(X_test, y_test)}')
