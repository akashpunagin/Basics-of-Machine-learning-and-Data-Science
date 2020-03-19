import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

# Get the dsataset
df = pd.read_csv('KNN_Project_Data')

print(df.head())

# Standardize the Variables
# create instance of StandardScaler
scaler = StandardScaler()

# fit the StandardScaler instance with dataset
scaler.fit(df.drop('TARGET CLASS',axis=1))

# transform the data
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))

# Convert the scaled features to a dataframe and make sure the scaling worked.
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
print('\nScaled dataset:\n',df_feat.head())

# Train Test Split
# X - independent variable, features
# y - dependent variable, target
X = scaled_features # or df_feat
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

# Using KNN

# Create instance of KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

# fit/train the model
knn.fit(X_train,y_train)

# predict test data
pred = knn.predict(X_test)

# compare the actual output values for X_test with the predicted values
df_to_compare = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
print("\nActual values VS Predictions (1st 10 values):\n", df_to_compare.head(10))

# Predictions and Evaluations
print("\nFor n_neighbors = 1:\n")
print('Confustion matrix:\n',confusion_matrix(y_test,pred))
print('Classification report:\n',classification_report(y_test,pred))

# Choosing a K Value (n_neighbors value)
# elbow method to pick a good K Value:
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test)) # np.mean(False) = 0.0 # np.mean(True) = 1.0

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.savefig('error_rate.jpg')
# plt.show()
# analize the graph and see the value of K for minimun value of error_rate (refer saved image)

K_for_min_error_rate = error_rate.index(min(error_rate)) + 1 # Because index starts from 0

print('Value of K for minimum error rate: ', K_for_min_error_rate)

knn = KNeighborsClassifier(n_neighbors=K_for_min_error_rate)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print("\nFor n_neighbors = {}:\n".format(K_for_min_error_rate))
print('Confustion matrix:\n',confusion_matrix(y_test,pred))
print('Classification report:\n',classification_report(y_test,pred))
