import json
import re
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
    text = re.sub(r'\{\{[^}]+\}\}', '', text)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'==+[^=]+==+', '', text)
    text = re.sub(r'\[\[(File|Image):.+?\]\]', '', text)
    text = re.sub(r"'''?", '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def load_character_descriptions(raw_data_path='raw_data', batch_pattern='lexicanum_page_texts_batch_*.json'):
    # loads and cleans descriptions from raw batch files
    descriptions = {}
    path = Path(raw_data_path)
    if not path.exists():
        path = Path(__file__).parent.parent / raw_data_path
    
    files = list(path.glob(batch_pattern))
    if not files:
        return {}

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for title, page_data in data.get('pages', {}).items():
                wikitext = page_data.get('wikitext', '')
                clean_text = clean_wikitext(wikitext)
                if len(clean_text) > 20:
                    descriptions[title] = clean_text
    return descriptions

def generate_embeddings(texts, model_name='all-MiniLM-L6-v2', batch_size=32):
    # generates embeddings for a list of text strings
    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False)

def cluster_embeddings(embeddings, n_clusters=16):
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
