import os
import pandas as pd
from data_processing import process_data
from embeddings import get_all_embeddings
from clustering import perform_clustering, evaluate_clustering
from results import visualize_clusters, get_top_words_per_cluster

def main():
    # Define paths and parameters
    RAW_DATA_PATH = 'data/raw'
    PROCESSED_DATA_PATH = 'data/processed'
    API_KEY = 'your_openai_api_key'
    N_CLUSTERS = 50

    # Step 1: Process the raw data to get cleaned DataFrame
    df_clean = process_data(RAW_DATA_PATH)
    print("Data processing complete. Sample data:")
    print(df_clean.head())

    # Step 2: Generate embeddings
    embeddings = get_all_embeddings(df_clean, PROCESSED_DATA_PATH, API_KEY)
    print("Embeddings generation complete.")

    # Choose the embedding type you want to use for clustering
    data = embeddings['openai']  # or 'gte' or 'openai'

    # Step 3: Perform clustering analysis
    cluster_labels, gmm = perform_clustering(data, N_CLUSTERS)
    df_clean['cluster'] = cluster_labels
    print("Clustering complete. Sample data with clusters:")
    print(df_clean.head())

    # Step 4: Evaluate clustering
    log_likelihood, bic, aic = evaluate_clustering(data, cluster_labels, gmm)
    print(f"Clustering evaluation:\nLog Likelihood: {log_likelihood}\nBIC: {bic}\nAIC: {aic}")

    # Step 5: Visualize the clusters
    visualize_clusters(data, cluster_labels, N_CLUSTERS)
    print("TSNE visualization saved.")

    # Step 6: Get top words for each cluster
    top_words_df = get_top_words_per_cluster(df_clean['abstract'], cluster_labels, N_CLUSTERS)
    print("Top words per cluster:")
    print(top_words_df)

if __name__ == '__main__':
    main()