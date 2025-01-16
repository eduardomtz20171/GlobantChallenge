# GlobantChallenge
## Project Overview
This project applies unsupervised machine learning techniques to classify NSF Research Award abstracts into topics based on their semantic similarity. The pipeline covers data preprocessing, embedding generation, clustering analysis, and visualization.
## Pipeline Overview
Data Processing → XML parsing and text cleaning.
Embedding Generation → TF-IDF, GTE Large, OpenAI embeddings.
Clustering Analysis → Optimal cluster search and clustering (GMM).
Evaluation → Model evaluation using Log Likelihood, BIC, and AIC.
Visualization → Cluster visualization using t-SNE.
Interpretation → Extraction of top words per cluster.

## Folder Structure

NSF_Abstracts_Clustering/
├── data/
│   ├── raw/                # Original dataset (XML abstracts)
│   └── processed/          # Processed data and saved embeddings
│
├── notebooks/
│   └── EDA.ipynb           # Exploratory Data Analysis notebook
│
├── src/
│   ├── data_processing.py  # Data loading and preprocessing
│   ├── embeddings.py       # Embedding generation
│   ├── clustering.py       # Clustering models and evaluation
│   ├── visualization.py    # Visualization utilities
│   └── main.py             # Pipeline orchestration script
│
├── results/
│   ├── clusters_visualizations/  # Cluster plots
│   └── evaluation_metrics.txt    # Clustering evaluation metrics
│
├── tests/
│   └── test_functions.py    # Unit tests
│
|
└── README.md                # Project documentation


## Pipeline Details
Step 1: Data Processing
Loads and parses XML abstracts.
Cleans text and removes stopwords.
Step 2: Embedding Generation
TF-IDF (Sparse representation of words)
GTE Large (Pre-trained semantic embeddings)
OpenAI Embeddings (Using OpenAI API)
Step 3: Clustering
USed K-Means to determine best centroids which were then used to initialize a Gaussian Mixture Model (GMM) applied to embeddings.
Optimal cluster number determined via model selection using the first derivative of the Aikake Information Criterion as guide
Step 4: Evaluation
Log-Likelihood, BIC, and AIC metrics.
Step 5: Visualization
t-SNE visualizations for each embedding type.
Step 6: Interpretation
Extraction of the top words for each discovered cluster to determine topic of the cluster.
