import json
import re
import pickle
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_wikitext(text):
    # cleans wikitext markup to get plain text
    if not text:
        return ""
    
    # Remove everything from reference sections onward (Sources, References, See Also, External Links, etc.)
    reference_pattern = r'==+\s*(Sources|References|See Also|External Links|Related Articles|Notes|Footnotes)\s*==+.*'
    match = re.search(reference_pattern, text, re.IGNORECASE)
    if match:
        text = text[:match.start()]
    
    text = re.sub(r'\{\{[^}]+\}\}', '', text) #remove template markup
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text) #remove wiki links
    text = re.sub(r'==+[^=]+==+', '', text) #remove remaining section headers
    text = re.sub(r'\[\[(File|Image):.+?\]\]', '', text) #remove file links
    text = re.sub(r"'''?", '', text) #remove bold formatting
    text = re.sub(r'<[^>]+>', '', text) #remove HTML tags
    text = re.sub(r'\s+', ' ', text).strip() #remove extra whitespace
    return text.lower()

def load_character_descriptions(raw_data_path='raw_data', batch_pattern='lexicanum_page_texts_batch_*.json', valid_nodes=None):
    # loads and cleans descriptions from raw batch files
    # if valid_nodes is provided (set of node IDs), only includes characters in that set
    descriptions = {}
    path = Path(raw_data_path)
    if not path.exists():
        path = Path(__file__).parent.parent / raw_data_path
    
    files = list(path.glob(batch_pattern))
    if not files:
        return {}

    # Load filtered network nodes if valid_nodes not provided but we want filtered
    if valid_nodes is None:
        try:
            # Try to import config, handling different import paths
            try:
                from config import PICKLE_FILTERED_FILE
                filter_path = Path(PICKLE_FILTERED_FILE)
            except ImportError:
                # If config import fails, use relative path
                filter_path = Path(__file__).parent.parent / "data" / "lexicanum_network_filtered.pkl"
            
            if not filter_path.exists():
                filter_path = Path(__file__).parent.parent / "data" / "lexicanum_network_filtered.pkl"
            
            if filter_path.exists():
                with open(filter_path, 'rb') as f:
                    G_filtered = pickle.load(f)
                valid_nodes = set(G_filtered.nodes())
            else:
                # If filtered network doesn't exist, warn but proceed without filtering
                print("Warning: Filtered network file not found. Loading all descriptions.")
                valid_nodes = None
        except Exception as e:
            # If we can't load filtered network, warn and proceed without filtering
            print(f"Warning: Could not load filtered network ({e}). Loading all descriptions.")
            valid_nodes = None

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for title, page_data in data.get('pages', {}).items():
                # Only include if in valid_nodes (if provided)
                if valid_nodes is not None and title not in valid_nodes:
                    continue
                    
                wikitext = page_data.get('wikitext', '')
                clean_text = clean_wikitext(wikitext)
                if len(clean_text) > 20:
                    descriptions[title] = clean_text
    return descriptions

def generate_embeddings(texts, model_name='all-MiniLM-L6-v2', batch_size=32):
    # generates embeddings for a list of text strings
    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False)

def cluster_embeddings(embeddings, n_clusters=20):
    # clusters embeddings using k-means
    n_samples = len(embeddings)
    if n_samples < n_clusters:
        n_clusters = n_samples
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(embeddings)

def extract_cluster_keywords(texts, cluster_ids, n_keywords=5):
    # extracts top keywords for each cluster using tf-idf
    df = pd.DataFrame({'text': texts, 'cluster': cluster_ids})
    keywords = {}
    cluster_corpus = df.groupby('cluster')['text'].apply(' '.join)
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000, max_df=0.9)
    
    try:
        tfidf_matrix = tfidf.fit_transform(cluster_corpus)
        feature_names = np.array(tfidf.get_feature_names_out())
        
        for i, cluster_id in enumerate(cluster_corpus.index):
            row = tfidf_matrix[i].toarray().flatten()
            if np.all(row == 0):
                keywords[cluster_id] = "no keywords"
                continue
            top_indices = row.argsort()[-n_keywords:][::-1]
            keywords[cluster_id] = ", ".join(feature_names[top_indices])
    except ValueError:
        unique_clusters = set(cluster_ids)
        return {c: "error" for c in unique_clusters}
        
    return keywords

if __name__ == "__main__":
    descs = load_character_descriptions()
    if descs:
        test_titles = list(descs.keys())[:50]
        test_texts = [descs[t] for t in test_titles]
        embs = generate_embeddings(test_texts)
        clusters = cluster_embeddings(embs, n_clusters=5)
        keywords = extract_cluster_keywords(test_texts, clusters)
