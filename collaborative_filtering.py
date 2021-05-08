#-------------------------------------------------------------------------
# AUTHOR: Daniel Molina
# FILENAME: collaborative_filtering.py
# SPECIFICATION: This program will run collaborative filtering
# FOR: CS 4200- Assignment #5
# TIME SPENT: 1 hour and a half
#-----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('trip_advisor_data.csv', sep=',', header=0) #reading the data by using the Pandas library ()

simArray = []
#iterate over the other 99 users to calculate their similarity with the active user (user 100) according to their category ratings (user-item approach)
cols = list(df.columns)
print(cols)
cols.pop(0)
cols.pop(cols.index('galleries'))
cols.pop(cols.index('restaurants'))
print(cols)

for i in range(99):   
   # do this to calculate the similarity:
   #vec1 = np.array([[1,1,0,1,1]])
   #vec2 = np.array([[0,1,0,1,1]])
   #cosine_similarity(vec1, vec2)
   #do not forget to discard the first column (User ID) when calculating the similarities
   vec1 = np.array([df[cols].loc[i,:]])
   vec2 = np.array([df[cols].loc[99,:]])
   sim = cosine_similarity(vec1,vec2)
   simArray.append((sim,df.loc[i,:]))
#find the top 10 similar users to the active user according to the similarity calculated before
simArray.sort(key=lambda x:-x[0]) # sort by the first value, greatest first
topTen = simArray[:10] # first 10

#Compute a prediction from a weighted combination of selected neighbors’ for both categories evaluated (galleries and restaurants)
averageFromUser100 = sum(df[cols].loc[99,:])/len(df[cols].loc[99,:])
sumOfSimilarities = 0
sumWeightedAverageGalleries = 0
sumWeightedAverageRestaurants = 0
for sim, user in topTen:
   sim = sim[0][0]
   sumOfSimilarities += sim
   arr = [float(x) for x in user[1:]] # convert all to float
   sumWeightedAverageGalleries += (sim * (float(user['galleries']) - sum(arr)/len(arr)))
   sumWeightedAverageRestaurants += (sim * (float(user['restaurants']) - sum(arr)/len(arr)))

galleryPrediction = round(averageFromUser100 + (sumWeightedAverageGalleries/sumOfSimilarities))
resturantPrediction = round(averageFromUser100 + (sumWeightedAverageGalleries/sumOfSimilarities))
print('gallery=%d and restaurant=%d'%(galleryPrediction, resturantPrediction))