import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from nltk.corpus import stopwords
import os

def visualize_clusters(data, cluster_labels, n_clusters, output_path='../results/tsne_clusters.png'):
    # Perform TSNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(data)

    # Create a DataFrame for TSNE results
    df_tsne = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
    df_tsne['cluster'] = cluster_labels

    # Plot TSNE results with clusters colored
    plt.figure(figsize=(10, 8))
    for cluster in range(n_clusters):
        cluster_data = df_tsne[df_tsne['cluster'] == cluster]
        plt.scatter(cluster_data['tsne1'], cluster_data['tsne2'], label=f'Cluster {cluster}', alpha=0.6)
    plt.legend()
    plt.title('TSNE Visualization of Clusters')
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    
    # Save the plot to the specified output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def get_top_words_per_cluster(texts, cluster_labels, n_clusters, top_n=10, output_path='../results/top_words_per_cluster.csv'):
    # Vectorize the texts using TF-IDF
    # Define the custom stopwords
    custom_stopwords = ['nsf','grant','award','proposal','research','project', 'merit', 'broader', 'impacts', 'using', 'intellectual', 'review', 'criteria', 'deemed']
    base_stops = set(stopwords.words('english'))
    base_stops = base_stops.union(set(custom_stopwords))

    vectorizer = TfidfVectorizer(max_features=5000, stop_words=list(base_stops))
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Create a DataFrame to store top words for each cluster
    top_words_df = pd.DataFrame(columns=[f'Cluster {i}' for i in range(n_clusters)])

    for cluster in range(n_clusters):
        cluster_texts = tfidf_matrix[cluster_labels == cluster]
        mean_tfidf = cluster_texts.mean(axis=0).A1
        top_indices = mean_tfidf.argsort()[-top_n:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_words_df[f'Cluster {cluster}'] = top_words

    # Save the DataFrame to the specified output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    top_words_df.to_csv(output_path, index=False)

    return top_words_df

# Example usage:
# Assuming `embeddings` is a dictionary containing the embeddings from the previous script
# embeddings = get_all_embeddings(df_abstracts, PROCESSED_DATA_PATH, API_KEY)

# Choose the embedding type you want to use for clustering
# data = embeddings['tfidf']  # or 'gte' or 'openai'

# Perform clustering analysis
# n_clusters = 10  # Set the number of clusters
# cluster_labels, gmm = perform_clustering(data, n_clusters)

# Visualize the clusters
# visualize_clusters(data, cluster_labels, n_clusters)

# Get the top words for each cluster
# top_words_df = get_top_words_per_cluster(df_abstracts['abstract'], cluster_labels, n_clusters)
# print(top_words_df)