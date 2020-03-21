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

# Exploratory Data Analysis
sns.pairplot(df,hue='Kyphosis',palette='Set1')
# plt.show()

# Training and Testing Data
# X - independent variable, features
# y - dependent variable, target
X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']
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

print('\nFinal Scores of different models:')
print(f'Decision Tree : {dtree.score(X_test, y_test)}')
print(f'Random Forest : {rfc.score(X_test, y_test)}')
