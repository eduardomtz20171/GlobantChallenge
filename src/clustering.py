import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pandas as pd
import matplotlib.pyplot as plt

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

# Function to evaluate the quality of the clustering
def evaluate_clustering(data, cluster_labels, gmm):
    # Calculate the log-likelihood of the data
    log_likelihood = gmm.score(data)
    
    # Calculate the Bayesian Information Criterion (BIC)
    bic = gmm.bic(data)
    
    # Calculate the Akaike Information Criterion (AIC)
    aic = gmm.aic(data)
    
    return log_likelihood, bic, aic

# Function to find the best number of clusters
def find_best_cluster_number(data, max_clusters):
    bic_scores = []
    aic_scores = []
    cluster_range = range(1, max_clusters + 1, 3)

    # Compute BIC and AIC scores for each cluster number
    for n_clusters in cluster_range:
        cluster_labels, gmm = perform_clustering(data, n_clusters)
        log_likelihood, bic, aic = evaluate_clustering(data, cluster_labels, gmm)
        bic_scores.append(bic)
        aic_scores.append(aic)

    # Plot the BIC and AIC scores
    plt.figure(figsize=(10, 5))
    plt.plot(cluster_range, bic_scores, label='BIC')
    plt.plot(cluster_range, aic_scores, label='AIC')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('BIC and AIC Scores for Different Number of Clusters')
    plt.legend()
    plt.show()

    # Calculate the first derivative of AIC scores
    aic_diff = np.diff(aic_scores)

    # Find the index where the improvement slows (derivative closest to zero)
    best_n_index = np.argmin(np.abs(aic_diff))  # Closest to zero change
    best_n_clusters = cluster_range[best_n_index + 1]  # +1 due to diff reducing length by 1

    return best_n_clusters

# Example usage:
# Assuming `embeddings` is a dictionary containing the embeddings from the previous script
# embeddings = get_all_embeddings(df_abstracts, PROCESSED_DATA_PATH, API_KEY)

# Choose the embedding type you want to use for clustering
# data = embeddings['tfidf']  # or 'gte' or 'openai'

# Perform clustering analysis
# max_clusters = 10  # Set the maximum number of clusters to experiment with
# best_n_clusters = find_best_cluster_number(data, max_clusters)
# cluster_labels, gmm = perform_clustering(data, best_n_clusters)

# Add the cluster labels to the DataFrame
# df_abstracts['cluster'] = cluster_labels
# print(df_abstracts.head())

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