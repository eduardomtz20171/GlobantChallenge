import unittest
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import tiktoken
from openai import OpenAI
from clustering_analysis import perform_clustering
from visualization import get_top_words_per_cluster

class TestFunctions(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.texts = [
            "This is a test abstract for research project.",
            "Another example of a research abstract.",
            "Yet another research project abstract example.",
            "This is a different kind of abstract for testing.",
            "Testing the clustering and embedding functions."
        ]
        self.processed_data_path = 'data/processed'
        self.api_key = 'your_openai_api_key'
        self.n_clusters = 2

    def test_get_tfidf_embeddings(self):
        from embeddings import get_tfidf_embeddings
        embeddings = get_tfidf_embeddings(self.texts)
        self.assertEqual(embeddings.shape[0], len(self.texts))
        self.assertEqual(embeddings.shape[1], 5000)

    def test_get_gte_embeddings(self):
        from embeddings import get_gte_embeddings
        embeddings = get_gte_embeddings(self.texts, self.processed_data_path)
        self.assertEqual(embeddings.shape[0], len(self.texts))

    def test_get_openai_embeddings(self):
        from embeddings import get_openai_embeddings
        embeddings = get_openai_embeddings(self.texts, self.processed_data_path, self.api_key)
        self.assertEqual(embeddings.shape[0], len(self.texts))

    def test_perform_clustering(self):
        from embeddings import get_tfidf_embeddings
        data = get_tfidf_embeddings(self.texts)
        cluster_labels, gmm = perform_clustering(data, self.n_clusters)
        self.assertEqual(len(cluster_labels), len(self.texts))
        self.assertEqual(len(np.unique(cluster_labels)), self.n_clusters)

    def test_get_top_words_per_cluster(self):
        from embeddings import get_tfidf_embeddings
        data = get_tfidf_embeddings(self.texts)
        cluster_labels, gmm = perform_clustering(data, self.n_clusters)
        top_words_df = get_top_words_per_cluster(self.texts, cluster_labels, self.n_clusters)
        self.assertEqual(top_words_df.shape[1], self.n_clusters)
        self.assertEqual(top_words_df.shape[0], 10)

if __name__ == '__main__':
    unittest.main()