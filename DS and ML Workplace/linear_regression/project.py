# You just got some contract work with an Ecommerce company based in New York City that sells clothing online but they also have
# in-store style and clothing advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist,
# then they can go home and order either on a mobile app or website for the clothes they want.
# The company is trying to decide whether to focus their efforts on their mobile app experience or their website

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Read the dataset
customers = pd.read_csv("Ecommerce Customers")

print(customers.head())
print(customers.describe())
print(customers.info())

# Exploratory Data Analysis, consider only numerical columns of the dataset
# jointplot to compare the Time on Website and Yearly Amount Spent columns
# sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers, kind='scatter')
# plt.show()

# jointplot to compare the Time on App and Yearly Amount Spent columns
# sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers, kind='scatter')
# plt.show()

# jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership
# sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)
# plt.show()

# using pairplot to explore the entire dataset
# sns.pairplot(customers)
# plt.show()
# according to the graph most correlated feature with Yearly Amount Spent is Length Of Membership

# Create a linear model plot of Yearly Amount Spent vs. Length of Membership
# sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)
# plt.show()


# Training and Testing Data
# X - independent variable, features
# y - dependent variable, target
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# Train The Model
# Create an instance of a LinearRegression()
lm = LinearRegression()

# train or fit lm on training data
lm.fit(X_train, y_train)

# print the coefficient and intercept of the model
print("\nIntercept: ", lm.intercept_)
print("Coefficients:\n", lm.coef_)

# Coefficient DataFrame
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
print("\nCoefficient DataFrame is: \n", coeff_df)
# It means, if all the other features were fixed, than 1 unit change in a particular feature
# will be associated with an increase/decrease of corresponding value of y ie Yearly Amount Spent

# Predict Test Data
predictions = lm.predict(X_test)

# compare the actual output values for X_test with the predicted values
df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print("\nActual values VS Predictions (1st 10 values):\n", df.head(10))

# comparison of Actual and Predicted values using bar plot
# df.head(20).plot(kind='bar',figsize=(10,8))
# plt.show()

# Create a scatterplot of the real test values versus the predicted values
# plt.scatter(y_test,predictions)
# plt.xlabel('Y Test')
# plt.ylabel('Predicted Y')
# plt.title('Actual values VS Predictions')
# plt.tight_layout()
# plt.show()

# Evaluating The Model, Mean Absolute Error, Mean Squared Error, Root Mean Squared Error
print('\nModel Evaluation')
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# mean_Yearly_Amount_spent = customers.describe().loc['mean', 'Yearly Amount Spent']
# print("10% of mean_Yearly_Amount_spent: ",mean_Yearly_Amount_spent / 10) # compare with RMSE

# Residuals
# Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist()
# sns.distplot((y_test-predictions),bins=50)
# plt.title('Residuals')
# plt.show()

# Conclusion (Refer coefficient df)
# there are two ways to think about this: Develop the Website to catch up to the performance of the mobile app,
# or develop the app more since that is what is working better
