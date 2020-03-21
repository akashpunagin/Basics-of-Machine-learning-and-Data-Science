# We will use lending data from 2007-2010 and try to classify and predict whether or not the borrower paid back their loan in full

# Here are what the columns represent:
#     credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
#     purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
#     int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
#     installment: The monthly installments owed by the borrower if the loan is funded.
#     log.annual.inc: The natural log of the self-reported annual income of the borrower.
#     dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
#     fico: The FICO credit score of the borrower.
#     days.with.cr.line: The number of days the borrower has had a credit line.
#     revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
#     revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
#     inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
#     delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
#     pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Get the Data
loans = pd.read_csv('loan_data.csv')

print(loans.head())
print(loans.describe())
print(loans.info())

# Exploratory Data Analysis

# histogram of two FICO distributions on top of each other, one for each credit.policy outcome
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')
# plt.show()

# histogram of two FICO distributions on top of each other, one for each not.fully.paid outcome
plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')
# plt.show()

# countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid
plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')
# plt.show()

# jointplot to see the trend between FICO score and interest rate
sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')
# plt.show()

# lmplots to see if the trend differed between not.fully.paid and credit.policy
plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',col='not.fully.paid',palette='Set1')
# plt.show()

# Setting up the Data
# Categorical Features - notice the purpose is a Categorical feature
final_data = pd.get_dummies(loans,columns=['purpose'],drop_first=True)

print(final_data.info())

# Training and Testing Data
# X - independent variable, features
# y - dependent variable, target
X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Training a Decision Tree Model
print("\n### DECISION TREE ###\n")

# Create an instance of DecisionTreeClassifier()
dtree = DecisionTreeClassifier()

# train / fit the DecisionTreeClassifier instance
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
