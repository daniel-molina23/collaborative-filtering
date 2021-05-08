#-------------------------------------------------------------------------
# AUTHOR: Daniel Molina
# FILENAME: clustering.py
# SPECIFICATION: This program is meant to run K-means and find the best silhoutte score
# FOR: CS 4200- Assignment #5
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = df.copy()

#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
bestK = -1
maxScore = -1
k_values = range(2,21)
s_score = []
best_kmeans_model = None
for k in k_values:
	kmeans = KMeans(n_clusters=k, random_state=0)
	kmeans.fit(X_training)
	#for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
	#find which k maximizes the silhouette_coefficient
	score = silhouette_score(X_training, kmeans.labels_)
	s_score.append(score)
	if(score > maxScore):
		bestK = k
		maxScore = score
		best_kmeans_model = kmeans



#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
plt.plot(k_values, s_score, '-o')
plt.show()

#reading the validation data (clusters) by using Pandas library
y_test = pd.read_csv('testing_data.csv', sep=',', header=None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
labels = np.array(y_test.values).reshape(1,len(y_test))[0]

#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, best_kmeans_model.labels_).__str__())


#run agglomerative clustering now by using the best value of k calculated before by kmeans
#Do it:
agg = AgglomerativeClustering(n_clusters=bestK, linkage='ward')
agg.fit(X_training)

#Calculate and print the Homogeneity of this agglomerative clustering
print("Agglomerative Clustering Homogeneity Score = " + metrics.homogeneity_score(labels, agg.labels_).__str__())
