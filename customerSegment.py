# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 10:22:10 2017

@author: Georgie
"""

"""
We're going to learn about customer segmentation from yhat's blog
http://blog.yhat.com/posts/customer-segmentation-python-rodeo.html.
'The dataset contains both information on marketing newsletters/e-mail 
campaigns (e-mail offers sent) and transaction level data from customers
 (which offer customers responded to and what they bought).'
"""

#Packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


#Load the offers data
df_offers = pd.read_excel("./WineKMC.xlsx", sheetname=0)
df_offers.columns = ["offer_id", "campaign", "varietal", "min_qty", "discount", "origin", "past_peak"]
df_offers.head()

#Transaction data
df_transactions = pd.read_excel("./WineKMC.xlsx", sheetname=1)
df_transactions.columns = ["customer_name", "offer_id"]
df_transactions['n'] = 1
df_transactions.head()

#join the offers and transactions table
df = pd.merge(df_offers, df_transactions)

# create a "pivot table" which will give us the number 
#of times each customer responded to a given offer
matrix = df.pivot_table(index=['customer_name'], columns=['offer_id'], values='n')

# a little tidying up. fill NA values with 0 and make the index into a column
matrix = matrix.fillna(0).reset_index()

# save a list of the 0/1 columns. we'll use these a bit later
x_cols = matrix.columns[1:]

#Perform a k means clustering on the data
cluster = KMeans(n_clusters=5)

# slice matrix so we only include the 0/1 indicator columns in the clustering
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[2:]])

matrix.cluster.value_counts()

#We'll do some principle components analysis to help visualise the data
#We'll reduce the dimensionaliyt of the data to 2.

pca = PCA(n_components=2)
matrix['x'] = pca.fit_transform(matrix[x_cols])[:,0]
matrix['y'] = pca.fit_transform(matrix[x_cols])[:,1]
matrix = matrix.reset_index()
customer_clusters = matrix[['customer_name', 'cluster', 'x', 'y']]
customer_clusters.head()

#Let's look at a scatter plot of the data

df = pd.merge(df_transactions, customer_clusters)
df = pd.merge(df_offers, df)

plt.figure(1)
plt.scatter(df['x'],df['y'], c = df['cluster'], alpha = 0.5)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Customers Grouped by Cluster')

#We can add the centroids after a PCA on those
cluster_centers = pca.fit_transform(cluster.cluster_centers_)
cluster_centers = pd.DataFrame(cluster_centers, columns=['x', 'y'])
cluster_centers['cluster'] = range(0, len(cluster_centers))
plt.scatter(cluster_centers['x'],cluster_centers['y'],marker = 'D', s = 50)


#Analyse the clusters
#Let's look at cluster 4
df['is_4'] = df.cluster==4
df.groupby("is_4").varietal.value_counts()
#Nearly all the Cab was bought by cluster 4.
#Look at numerical features
df.groupby("is_4")[['min_qty', 'discount']].mean()
#cluster 4 likes to buy in bulk.


