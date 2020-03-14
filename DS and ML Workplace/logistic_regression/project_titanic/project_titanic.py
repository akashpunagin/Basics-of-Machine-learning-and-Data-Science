import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
sns.set_style('whitegrid')

# Read Dataset
train = pd.read_csv('titanic_train.csv')

print(train.head())
print(train.describe())
print(train.info())

# Exploratory Data Analysis
# Missing Data
# sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# plt.tight_layout()
# plt.show()
# Roughly 20 percent of the Age data is missing, therefore it is reasonable to replace the missing data with some form of imputation

# During Classifiction analysis, it is better practice to check the target labels
# sns.countplot(x='Survived',data=train,palette='RdBu_r')
# plt.show()

# Countplot of Survived with hue of sex
# sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
# plt.show()
# Among the male are more deseased than females, females have survived more than males

# Countplot of survived with hue of Pclass
# sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
# plt.show()
# The deseased passengers are more in Pclass 3, survivors are more in Pclass 1

# Distribution plot for Age
# sns.distplot(train['Age'].dropna(),kde=True,color='darkred',bins=30)
# plt.show()
# From Graph, few children, 20 - 30 aged passengers are more in number,
# older people are less abord

# Countplot for SibSp, (siblings and spouses)
# sns.countplot(x='SibSp',data=train)
# plt.show()
# most of the passengers did not have siblings or spouses on board
# 1 SibSp probably means there are people with their spouses on board

# how much did passengers pay?
# train['Fare'].hist(color='green',bins=40,figsize=(10,4))
# plt.show()
# Most of the purchase prices are between 10 and 50


# Data Cleaning
# The age column has null values. fill the null values by Imputation

# Check Age VS Pclass boxplot.
# plt.figure(figsize=(12, 7))
# sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
# plt.show()
# by the figure, people in higher class are older, hence impute the Age according to Pclass

print("Mean Age according to Pclass\n",train.groupby('Pclass').mean()['Age'])
df_mean_age = train.groupby('Pclass').mean()['Age']
print(int(df_mean_age[1]))
print(df_mean_age[2])
print(df_mean_age[3])
# df_mean_age.loc['1', 'Pclass']

# Create a function for imputation
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            # return 37
            return int(df_mean_age[1])
        elif Pclass == 2:
            # return 29
            return int(df_mean_age[2])
        else:
            # return 24
            return int(df_mean_age[3])
    else:
        return Age

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

# Check the headmap again
# sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# plt.show()

# Drop Cabin column because there are more missing values
train.drop('Cabin',axis=1,inplace=True)

# One field in Embarked is NaN, since its only single row, delete it
train.dropna(inplace=True)

# Check the headmap again
# sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# plt.show()
# No Missing values


# Converting Categorical Features
# convert categorical features to dummy variables using pandas,
# otherwise our machine learning algorithm won't be able to directly take in those features as inputs

print('Dummy variable for male-female column:\n',pd.get_dummies(train['Sex'],drop_first=True).head())

# Converting to dummy variable
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
pclass = pd.get_dummies(train['Pclass'], drop_first=True)
pclass.columns = ['Pclass-2', 'Pclass-3']
print("pclass DataFrame:\n",pclass.head())

# Drop Categorical features
train.drop(['Sex','Embarked','Name','Ticket','PassengerId','Pclass'],axis=1,inplace=True)

# Concat the stored dummy variable along colums(axis=1)
train = pd.concat([train,sex,embark,pclass],axis=1)

print("Numerical form of Dataset:\n",train.head())

# Building a Logistic Regression model
# X - features
# y - column that you are trying to predict

X = train.drop('Survived', axis=1)
y = train['Survived']

# Split the Dataset into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Create instance of LogisticRegression()
logmodel = LogisticRegression()

# fir / train the model with training set
logmodel.fit(X_train,y_train)

# Predict y values with X_test
predictions = logmodel.predict(X_test)

# compare the actual output values for X_test with the predicted values
df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print("\nActual values VS Predictions (1st 10 values):\n", df.head(10))

df['is_same'] = df['Actual'] - df['Predicted']
df['same_or_notsame'] = df['is_same'].apply(lambda x: 'not same' if (x !=0) else 'same')
print(df.head())

print('\nHow many did the model predicted right?\n',df['same_or_notsame'].value_counts())

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logmodel.score(X_test, y_test)))


# Evaluate the model, pass true values and predictions to classification_report
print('Classifiction Report:\n',classification_report(y_test,predictions))
print('\nConfusion Matrix:\n',confusion_matrix(y_test,predictions))


# The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives
# The precision is intuitively the ability of the classifier to not label a sample as positive if it is negative.

# The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives.
# The recall is intuitively the ability of the classifier to find all the positive samples.

# The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall,
# where an F-beta score reaches its best value at 1 and worst score at 0.
# The F-beta score weights the recall more than the precision by a factor of beta. beta = 1.0 means recall and precision are equally important.

# The support is the number of occurrences of each class in y_test.
