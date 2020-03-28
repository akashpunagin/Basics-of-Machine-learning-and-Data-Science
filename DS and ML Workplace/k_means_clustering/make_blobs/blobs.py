import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,classification_report

# Create some Data
data = make_blobs(n_samples=200, n_features=2,centers=4, cluster_std=5) # random_state=101
# n_samples - number of samples
# n_features - number of features
# cluster_std - cluster standard deviation, if this is high, the noise between the cluts will increase

print(data)
# data[0] - data
# data[1] - belong to which cluster

print('Shape of data : ', data[0].shape)

# Visualize Data
# plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
# plt.show()

# Creating the Clusters
# Create instance of KMeans
kmeans = KMeans(n_clusters=4)

# train/fit the model
kmeans.fit(data[0])

print('kmeans.cluster_centers_ :\n',kmeans.cluster_centers_)

# Display clusters with its centers
plt.scatter(data[0][:,0],data[0][:,1],c=data[1])
plt.title('Cluster Centers are :')
for i,center in enumerate(kmeans.cluster_centers_):
    print(f'Center - {i+1} : {center[0]} , {center[1]}')
    plt.scatter(center[0], center[1], s=200, c='black', marker='o')
# plt.show()

# Labels predicted by KMeans model
print('Labels predicted by KMeans model:\n',kmeans.labels_)

# Model Evaluation
print('\nConfusion Matrix :\n',confusion_matrix(data[1],kmeans.labels_))
print('\nClassification Report :\n',classification_report(data[1],kmeans.labels_))

# Plot actual clusters and predicted clusters
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))

ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')

ax2.set_title("Original")
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')

for center in kmeans.cluster_centers_:
    ax1.scatter(center[0], center[1], s=200, c='black', marker='o')
    ax2.scatter(center[0], center[1], s=200, c='black', marker='o')

plt.show()
