# In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement
# We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
sns.set_style('whitegrid')

# Read Dataset
ad_data = pd.read_csv('advertising.csv')

print(ad_data.head())
print(ad_data.info())
print(ad_data.describe())

# Exploratory Data Analysis
# Create a histogram of the Age
# ad_data['Age'].hist(bins=30)
# plt.xlabel('Age')
# plt.show()

# Create a jointplot showing Area Income versus Age.
# sns.jointplot(x='Age',y='Area Income',data=ad_data)
# plt.show()

# Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.
# sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,color='red',kind='kde')
# plt.show()

# Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'
# sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data)
# plt.show()
# two clusters


# Logistic Regression
# X - features
# y - target
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

# Split Dataset into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# create instance of LogisticRegression()
logmodel = LogisticRegression()

# train/fit the model
logmodel.fit(X_train,y_train)

# Predictions
predictions = logmodel.predict(X_test)

# compare the actual output values for X_test with the predicted values
df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print("\nActual values VS Predictions (1st 10 values):\n", df.head(10))

df['is_same'] = df['Actual'] - df['Predicted']
df['same_or_notsame'] = df['is_same'].apply(lambda x: 'not same' if (x !=0) else 'same')

print('\nHow many did the model predicted right?\n',df['same_or_notsame'].value_counts())

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logmodel.score(X_test, y_test)))

print('Classifiction Report:\n',classification_report(y_test,predictions))
print('\nConfusion Matrix:\n',confusion_matrix(y_test,predictions))
