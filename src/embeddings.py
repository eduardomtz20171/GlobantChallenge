import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import tiktoken
from openai import OpenAI
import pandas as pd
from nltk.corpus import stopwords

# Define the custom stopwords
custom_stopwords = ['nsf','grant','award','proposal','research','project', 'merit', 'broader', 'impacts', 'using', 'intellectual', 'review', 'criteria', 'deemed']
base_stops = set(stopwords.words('english'))
base_stops = base_stops.union(set(custom_stopwords))

# Define the function to get TF-IDF embeddings
def get_tfidf_embeddings(texts):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=list(base_stops))
    return vectorizer.fit_transform(texts).toarray()

# Define the function to get GTE Large embeddings
def get_gte_embeddings(texts, processed_data_path):
    if os.path.exists(os.path.join(processed_data_path, 'embeddings_GTE-Large.npy')):
        return np.load(os.path.join(processed_data_path, 'embeddings_GTE-Large.npy'))
    else:
        model_gte = SentenceTransformer('thenlper/gte-large')
        embeddings = model_gte.encode(texts, show_progress_bar=True)
        np.save(os.path.join(processed_data_path, 'embeddings_GTE-Large.npy'), embeddings)
        return embeddings

# Define the function to get OpenAI embeddings
def get_openai_embeddings(texts, processed_data_path, api_key):
    if os.path.exists(os.path.join(processed_data_path, 'embeddings_OpenAI.npy')):
        return np.load(os.path.join(processed_data_path, 'embeddings_OpenAI.npy'))
    else:
        client = OpenAI(api_key=api_key)
        encoder = tiktoken.get_encoding("cl100k_base")
        max_tokens = 8000

        embeddings_list = []

        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                text = str(text) if text else ""

            tokens = encoder.encode(text)

            if len(tokens) <= max_tokens:
                response = client.embeddings.create(
                    input=text,
                    model="text-embedding-3-small"
                )
                emb = response.data[0].embedding
                embeddings_list.append(emb)
            else:
                chunk_embeddings = []
                start = 0
                while start < len(tokens):
                    end = min(start + max_tokens, len(tokens))
                    chunk_tokens = tokens[start:end]
                    chunk_text = encoder.decode(chunk_tokens)

                    response = client.embeddings.create(
                        input=chunk_text,
                        model="text-embedding-3-small"
                    )
                    chunk_emb = np.array(response.data[0].embedding, dtype=np.float32)
                    chunk_embeddings.append(chunk_emb)

                    start += max_tokens

                if len(chunk_embeddings) > 0:
                    doc_emb = np.mean(chunk_embeddings, axis=0)
                else:
                    doc_emb = np.zeros(1536, dtype=np.float32)

                embeddings_list.append(doc_emb)

            if (idx + 1) % 50 == 0:
                print(f"Processed {idx+1} / {len(texts)} abstracts")

        embeddings = np.array(embeddings_list, dtype=np.float32)
        np.save(os.path.join(processed_data_path, 'embeddings_OpenAI.npy'), embeddings)
        return embeddings

# Main function to get all embeddings
def get_all_embeddings(df_abstracts, processed_data_path, api_key):
    preprocessed_abstracts = df_abstracts['abstract'].tolist()
    
    tfidf_embeddings = get_tfidf_embeddings(preprocessed_abstracts)
    gte_embeddings = get_gte_embeddings(preprocessed_abstracts, processed_data_path)
    openai_embeddings = get_openai_embeddings(preprocessed_abstracts, processed_data_path, api_key)
    
    return {
        'tfidf': tfidf_embeddings,
        'gte': gte_embeddings,
        'openai': openai_embeddings
    }

# Example usage:
# RAW_DATA_PATH = 'data/raw'
# PROCESSED_DATA_PATH = 'data/processed'
# API_KEY = 'your_openai_api_key'
# df_abstracts = pd.read_csv(os.path.join(RAW_DATA_PATH, 'abstracts.csv'))
# embeddings = get_all_embeddings(df_abstracts, PROCESSED_DATA_PATH, API_KEY)
# print(embeddings)