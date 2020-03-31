# PCA (Principal Components Analysis) gives us our ideal set of features
# It creates a set of principal components that are rank ordered by variance, uncorrelated, and low in number

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Get the data
cancer = load_breast_cancer()

print(cancer.keys())
# print(cancer['data'])
# print(cancer['feature_names'])
# print(cancer['target_names'])

# DataFrame with data - cancer['data'] and columns - cancer['feature_names']
df = pd.DataFrame(data=cancer['data'],columns=cancer['feature_names'])

print(df.head())
print(df.info())

# Standardize the Variables

# The idea behind StandardScaler is that it will transform the data such that its distribution will have a mean value 0 and standard deviation of 1.
# In case of multivariate data, this is done feature-wise (in other words independently for each column of the data).

# create instance of StandardScaler
scaler = StandardScaler()

# fit the StandardScaler instance with dataset
scaler.fit(df)

# transform the data
scaled_data = scaler.transform(df)

#PCA

# Create instance of PCA
# We should specify how many components we want to keep when creating the PCA object.
pca = PCA(n_components=2)

# fit/train instance of PCA
pca.fit(scaled_data)

# Transform the data to its principal components
transformed_data = pca.transform(scaled_data)

print('Transformed Data:\n', transformed_data) # type - numpy.ndarray
print('Shape of Transformed Data : ', transformed_data.shape)

# DataFrame that will have the principal component
principal_component_df = pd.DataFrame(data=transformed_data, columns=['Principal Component-1', 'Principal Component-2'])
print('DataFrame of Principal Components:\n',principal_component_df.head())

# explained_variance_ratio is the amount of information or variance each principal component holds
# after projecting the data to a lower dimensional subspace
print('Explained variation per principal component :\n', pca.explained_variance_ratio_)
for i,variance in enumerate(pca.explained_variance_ratio_):
    print(f'Varience of Principal Component {i+1} : {variance}')

# Plot of Principal Components
plt.figure(figsize=(8,6))
plt.scatter(principal_component_df['Principal Component-1'],principal_component_df['Principal Component-2'],c=cancer['target'],cmap='plasma') # DataFrame
# or
# plt.scatter(transformed_data[:,0],transformed_data[:,1],c=cancer['target'],cmap='plasma') # numpy.ndarray
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
# plt.show()

# Interpreting the components
# The components correspond to combinations of the original features and not particularly single feature
print('\npca.components_ :\n', pca.components_)
components_df = pd.DataFrame(pca.components_,columns=cancer['feature_names'])
print('Component DataFrame : \n', components_df)
# each row represents a principal component, and each column relates back to the original features

# we can visualize the relationship between features and components with a heatmap:
plt.figure(figsize=(12,6))
sns.heatmap(components_df,cmap='plasma',)
plt.tight_layout()
plt.show()
