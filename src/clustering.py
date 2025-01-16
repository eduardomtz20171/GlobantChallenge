import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pandas as pd

def perform_clustering(data, n_clusters):
    # Step 1: Perform KMeans clustering to set initial centroids
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    initial_centroids = kmeans.cluster_centers_

    # Step 2: Use the initial centroids to initialize the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_clusters, means_init=initial_centroids, random_state=42)
    gmm.fit(data)

    # Predict the cluster for each data point
    cluster_labels = gmm.predict(data)
    return cluster_labels, gmm

#Function to evaluate the quality of the clustering
def evaluate_clustering(data, cluster_labels, gmm):
    # Calculate the log-likelihood of the data
    log_likelihood = gmm.score(data)
    
    # Calculate the Bayesian Information Criterion (BIC)
    bic = gmm.bic(data)
    
    # Calculate the Akaike Information Criterion (AIC)
    aic = gmm.aic(data)
    
    return log_likelihood, bic, aic

# Example usage:
# Assuming `embeddings` is a dictionary containing the embeddings from the previous script
# embeddings = get_all_embeddings(df_abstracts, PROCESSED_DATA_PATH, API_KEY)

# Choose the embedding type you want to use for clustering
# data = embeddings['tfidf']  # or 'gte' or 'openai'

# Perform clustering analysis
# n_clusters = 5  # Set the number of clusters
# cluster_labels, gmm = perform_clustering(data, n_clusters)

# Add the cluster labels to the DataFrame
# df_abstracts['cluster'] = cluster_labels
# print(df_abstracts.head())