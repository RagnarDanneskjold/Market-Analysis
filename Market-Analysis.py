# -*- coding: utf-8 -*-
"""
@author: wmol4
"""

import numpy as np
import pandas as pd
from IPython.display import display
import visuals as vs

#load the dataset and remove Region and Channel features
data = pd.read_csv("customers.csv")
data.drop(['Region', 'Channel'], axis = 1, inplace = True)
print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)

# Display a description of the dataset
display(data.describe())

#Choose three trivial indices from the dataset.
indices = [405, 210, 13]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples of wholesale customers dataset:"
display(samples)

#Make a copy of the DataFrame, using the 'drop' function to drop the given feature
new_data = data.drop(['Milk'], axis = 1)
new_labels = data['Milk']

#split the data into training and testing sets using the given feature as the target
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_data, new_labels, \
                                    test_size = 0.25, \
                                   random_state = 7)

#Create a decision tree regressor and fit it to the training set
from sklearn import tree
regressor = tree.DecisionTreeRegressor(random_state = 7)
regressor.fit(X_train, y_train)

#Report the score of the prediction using the testing set
score = regressor.score(X_test, y_test)
print(score)

# Produce a scatter matrix for each pair of features in the data
pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

#Scale the data using the natural logarithm
log_data = np.log(data)
display(log_data.describe())

#Scale the sample data using the natural logarithm
log_samples = np.log(samples)
display(log_samples.describe())

# Produce a scatter matrix for each pair of newly-transformed features
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

# Display the log-transformed sample data
display(log_samples)

allOutliers = []
repeatDict = {}

# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    
    #Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature], 25)
    
    #Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)
    
    #Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5 * (Q3 - Q1)
    
    # Display the outliers
    print "Data points considered outliers for the feature '{}':".format(feature)
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
    
    #finding repeats
    test = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    
    testIndex = test.index
    
    for ind in testIndex:
        allOutliers.append(ind)
        if ind in repeatDict:
            repeatDict[ind] += 1
        else:
            repeatDict[ind] = 1



print(repeatDict)

#Select the indices for data points you wish to remove
outliers  = [128, 154, 65, 66, 75]

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)

#Apply PCA by fitting the good data with the same number of dimensions as features
from sklearn.decomposition import PCA
pca = PCA(n_components = 6)
pca.fit(good_data)

#Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = vs.pca_results(good_data, pca)

# Display sample log-data after having a PCA transformation applied
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))

#Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components = 2)
pca.fit(good_data)

#Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

#Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))

# Create a biplot
vs.biplot(good_data, reduced_data, pca)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

range1 = np.arange(2, 5, 1)
range2 = np.arange(5, 300, 5)
testRange = np.concatenate((range1, range2), axis = 0)


score = 0

all_scores = []


for i in testRange:
    clusterer = KMeans(n_clusters = i)
    clusterer.fit(reduced_data)
    
    preds = clusterer.predict(reduced_data)
    
    centers = clusterer.cluster_centers_
    labels = clusterer.labels_
    sample_preds = clusterer.predict(pca_samples)
    scoreNew = silhouette_score(X = reduced_data, labels = preds)
    all_scores.append(scoreNew)
    if scoreNew >= score:
        score = scoreNew
        print 'n_clusters:', i, '. New score =', score 
print "Done searching"
print "Final score is", score

#for the next few parts, I will use 4 customer segments
clusterer = KMeans(n_clusters = 2)
clusterer.fit(reduced_data)
preds = clusterer.predict(reduced_data)
centers = clusterer.cluster_centers_
sample_preds = clusterer.predict(pca_samples)
score = silhouette_score(X = reduced_data, labels = preds)

#visualize how the score changes depending on n_clusters
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('n_clusters')
ax.set_ylabel('silhouette score')

ax.scatter(testRange, all_scores, c = 'r', marker = '.')
plt.show() #it should become clear that 2 clusters is the best until you begin overfitting the data (a lot of clusters)

print(type(preds))
# Display the results of the clustering from implementation
vs.cluster_results(reduced_data, preds, centers, pca_samples)

#Inverse transform the centers
log_centers = pca.inverse_transform(centers)

#Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments

print "the cluster centers"
display(true_centers)

#for my own reference to answer the questions
print "the overall data"
display(data.describe())

# Display the predictions
for i, pred in enumerate(sample_preds):
    print "Sample point", i, "predicted to be in Cluster", pred
    
# Display the clustering results based on 'Channel' data
vs.channel_results(reduced_data, outliers, pca_samples)
