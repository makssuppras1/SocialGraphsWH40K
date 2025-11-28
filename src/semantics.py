import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_wikitext(text):
    """Clean wikitext markup to get plain text."""
    if not text:
        return ""
    # Simple regex-based cleaning
    # Remove template calls {{...}}
    text = re.sub(r'\{\{[^}]+\}\}', '', text)
    # Remove links [[Link|Text]] -> Text or [[Link]] -> Link
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    # Remove headings ==...==
    text = re.sub(r'==+[^=]+==+', '', text)
    # Remove files/images
    text = re.sub(r'\[\[(File|Image):.+?\]\]', '', text)
    # Remove basic formatting ''' or ''
    text = re.sub(r"'''?", '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def load_character_descriptions(raw_data_path='raw_data', batch_pattern='lexicanum_page_texts_batch_*.json'):
    """Load and clean descriptions from raw batch files."""
    descriptions = {}
    path = Path(raw_data_path)
    if not path.exists():
        # Try relative to project root if not found
        path = Path(__file__).parent.parent / raw_data_path
    
    print(f"Loading descriptions from {path}/{batch_pattern}...")
    files = list(path.glob(batch_pattern))
    if not files:
        print("No batch files found!")
        return {}

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for title, page_data in data.get('pages', {}).items():
                wikitext = page_data.get('wikitext', '')
                clean_text = clean_wikitext(wikitext)
                # Only keep if significant text length
                if len(clean_text) > 20:
                    descriptions[title] = clean_text
    return descriptions

def generate_embeddings(texts, model_name='all-MiniLM-L6-v2', batch_size=32):
    """Generate embeddings for a list of text strings."""
    print(f"Generating embeddings for {len(texts)} texts using {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings

def cluster_embeddings(embeddings, n_clusters=20):
    """Cluster embeddings using K-Means."""
    print(f"Clustering {len(embeddings)} items into {n_clusters} clusters...")
    # If n_samples < n_clusters, reduce n_clusters
    n_samples = len(embeddings)
    if n_samples < n_clusters:
        print(f"Warning: n_samples ({n_samples}) < n_clusters ({n_clusters}). Adjusting n_clusters to {n_samples}.")
        n_clusters = n_samples
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)
    return cluster_ids

def extract_cluster_keywords(texts, cluster_ids, n_keywords=5):
    """Extract top keywords for each cluster using TF-IDF."""
    print("Extracting cluster keywords...")
    df = pd.DataFrame({'text': texts, 'cluster': cluster_ids})
    keywords = {}
    
    # Create a corpus where each document is the concatenated text of a cluster
    cluster_corpus = df.groupby('cluster')['text'].apply(' '.join)
    
    # Adjust max_df to ignore very common words across clusters
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000, max_df=0.9)
    
    try:
        tfidf_matrix = tfidf.fit_transform(cluster_corpus)
        feature_names = np.array(tfidf.get_feature_names_out())
        
        for i, cluster_id in enumerate(cluster_corpus.index):
            # Get top indices
            row = tfidf_matrix[i].toarray().flatten()
            # Check if row is all zeros
            if np.all(row == 0):
                keywords[cluster_id] = "no keywords"
                continue
                
            top_indices = row.argsort()[-n_keywords:][::-1]
            top_words = feature_names[top_indices]
            keywords[cluster_id] = ", ".join(top_words)
    except ValueError as e:
        print(f"Error in TF-IDF: {e}")
        # Fallback if corpus is empty or other issue
        return {c: "error" for c in set(cluster_ids)}
        
    return keywords

if __name__ == "__main__":
    # Test run
    descs = load_character_descriptions()
    print(f"Loaded {len(descs)} descriptions")
    
    if descs:
        # Use small subset for test
        test_titles = list(descs.keys())[:50]
        test_texts = [descs[t] for t in test_titles]
        
        embs = generate_embeddings(test_texts)
        print(f"Generated embeddings shape: {embs.shape}")
        
        clusters = cluster_embeddings(embs, n_clusters=5)
        print(f"Clusters: {clusters[:10]}")
        
        keywords = extract_cluster_keywords(test_texts, clusters)
        print("Cluster keywords:", keywords)

