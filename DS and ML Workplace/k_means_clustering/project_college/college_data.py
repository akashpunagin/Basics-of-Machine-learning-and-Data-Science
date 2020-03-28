# use KMeans Clustering to cluster Universities into to two groups: Private and Public
# we actually have the labels for this data set, but we will NOT use them for the KMeans clustering algorithm
# K-means is unsupervised machine learning algorithm, meaning it does not require labels

# The Data
#     Private A factor with levels No and Yes indicating private or public university
#     Apps Number of applications received
#     Accept Number of applications accepted
#     Enroll Number of new students enrolled
#     Top10perc Pct. new students from top 10% of H.S. class
#     Top25perc Pct. new students from top 25% of H.S. class
#     F.Undergrad Number of fulltime undergraduates
#     P.Undergrad Number of parttime undergraduates
#     Outstate Out-of-state tuition
#     Room.Board Room and board costs
#     Books Estimated book costs
#     Personal Estimated personal spending
#     PhD Pct. of faculty with Ph.D.’s
#     Terminal Pct. of faculty with terminal degree
#     S.F.Ratio Student/faculty ratio
#     perc.alumni Pct. alumni who donate
#     Expend Instructional expenditure per student
#     Grad.Rate Graduation rate

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,classification_report
sns.set_style('whitegrid')

# Get the data
df = pd.read_csv('College_Data',index_col=0)

print(df.head())
print(df.info())
print(df.describe())

# Exploratory data analysis
# scatterplot of Grad.Rate versus Room.Board where the points are colored by the Private column.
sns.lmplot(x='Room.Board',y='Grad.Rate',data=df, hue='Private',palette='coolwarm',size=6,aspect=1,fit_reg=False)
# plt.show()

# scatterplot of F.Undergrad versus Outstate where the points are colored by the Private column.
sns.lmplot(x='Outstate',y='F.Undergrad',data=df, hue='Private',palette='coolwarm',size=6,aspect=1,fit_reg=False)
# plt.show()

# a stacked histogram showing Out of State Tuition based on the Private column
# sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)
# plt.show()

# a stacked histogram showing Grad.Rate Tuition based on the Private column
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)
# plt.show()

# Notice there seems to be a private school with a graduation rate of higher than 100%. Check the name of that school.
print(df[df['Grad.Rate'] > 100])

# Check the Grad.Rate of the college
print('Grad.Rate of college : ',df['Grad.Rate']['Cazenovia College'])

# Set the Grad.Rate to 100
df['Grad.Rate']['Cazenovia College'] = 100 # Ignore the warning

print('Grad.Rate of college after correcting the value : ',df['Grad.Rate']['Cazenovia College'])

# Check the plot again
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)
# plt.show()

# K Means Cluster Creation
# Create an instance of a K Means model with 2 clusters. (for Private and Public)
kmeans = KMeans(n_clusters=2)

# Fit the model to all the data except for the Private label
df_train = df.drop('Private', axis=1)
kmeans.fit(df_train)

print('kmeans.cluster_centers_ :\n',kmeans.cluster_centers_)

# Display clusters with its centers
print('\nCluster Centers are :',)
for i,center in enumerate(kmeans.cluster_centers_):
    print(f'Center - {i+1} : {center[0]} , {center[1]}')
    plt.scatter(center[0], center[1], s=200, c='black', marker='o')
plt.title('Cluster Centers')
plt.tight_layout()
# plt.show()

print('Labels predicted by KMeans model:\n',kmeans.labels_)

# Evaluation
# There is no perfect way to evaluate clustering if you don't have the labels, however we do have labels,
# but in real world problems we will not have this luxury

df['Cluster'] = df['Private'].apply(lambda is_private: 1 if(is_private=='Yes') else 0)
df['Predicted'] = kmeans.labels_
print(df.head())


print('Confusion Matrix :\n',confusion_matrix(df['Cluster'],kmeans.labels_))
print('Classification Report:\n',classification_report(df['Cluster'],kmeans.labels_))
