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
from sklearn.metrics import pairwise_distances, silhouette_score
from joblib import Parallel, delayed
import multiprocessing

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

def _test_single_k(embeddings, k, random_state=42, inner_n_jobs=2):
    # Helper function to test a single k value (for parallelization)
    # inner_n_jobs: cores to use for silhouette calculation (K-Means handles its own threading)
    # Run K-Means clustering with k clusters
    # Note: K-Means doesn't support n_jobs parameter in this scikit-learn version
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # Calculate silhouette score (balances cohesion and separation)
    # Silhouette is the most expensive operation, so parallelization helps here
    silhouette = silhouette_score(embeddings, labels, n_jobs=inner_n_jobs)
    
    # Calculate average within-cluster distance for reference
    # This is less critical, so we can use fewer cores or sequential
    total_within_distance = 0
    total_pairs = 0
    
    for cluster_id in range(k):
        cluster_points = embeddings[labels == cluster_id]
        if len(cluster_points) > 1:
            # For small clusters, sequential is often faster due to overhead
            distances = pairwise_distances(cluster_points, n_jobs=1)
            n_points = len(distances)
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    total_within_distance += distances[i, j]
                    total_pairs += 1
    
    avg_within_distance = total_within_distance / total_pairs if total_pairs > 0 else float('inf')
    
    return {
        'k': k,
        'silhouette_score': silhouette,
        'avg_within_cluster_distance': avg_within_distance
    }

def find_optimal_semantic_clusters(embeddings, k_range=(2, 40), n_jobs=None):
    # Find the optimal number of semantic clusters using silhouette score
    # Silhouette score balances cohesion (within-cluster similarity) and separation (between-cluster distance)
    # Range: -1 to 1, where higher is better (1 = perfect separation, -1 = wrong clustering)
    # n_jobs: number of parallel jobs for outer loop (None = auto-balance)
    # Returns: (results_df, optimal_k, optimal_labels)
    
    total_cores = multiprocessing.cpu_count()
    
    # Balanced approach: use ~60-70% of cores for outer loop, rest for inner operations
    # This prevents over-subscription and contention
    if n_jobs is None:
        # For 11 cores: use 6-7 for outer loop, 2-3 for each inner operation
        outer_jobs = max(4, int(total_cores * 0.6))  # At least 4, but not all cores
        inner_jobs = max(2, total_cores - outer_jobs)  # Remaining cores for inner ops
    else:
        outer_jobs = n_jobs
        inner_jobs = max(2, total_cores - outer_jobs)
    
    print(f"Finding optimal number of clusters (testing k from {k_range[0]} to {k_range[1]})...")
    print(f"Parallelization: {outer_jobs} cores for testing different k values, {inner_jobs} cores per k for K-Means/silhouette")
    
    # Make sure we don't test more clusters than we have data points
    max_k = min(k_range[1], len(embeddings))
    min_k = k_range[0]
    
    # Parallelize the k testing loop - test multiple k values simultaneously
    # Each k test will use inner_jobs cores internally
    print("Testing all k values in parallel...")
    results = Parallel(n_jobs=outer_jobs, verbose=1)(
        delayed(_test_single_k)(embeddings, k, inner_n_jobs=inner_jobs) 
        for k in range(min_k, max_k + 1)
    )
    
    # Convert to DataFrame for easy analysis
    results_df = pd.DataFrame(results)
    
    # Find the k with highest silhouette score (best balance of cohesion and separation)
    optimal_k = results_df.loc[results_df['silhouette_score'].idxmax(), 'k']
    
    print(f"\nOptimal number of clusters: k={optimal_k}")
    print(f"  Silhouette score: {results_df.loc[results_df['k']==optimal_k, 'silhouette_score'].values[0]:.4f}")
    print(f"  Average within-cluster distance: {results_df.loc[results_df['k']==optimal_k, 'avg_within_cluster_distance'].values[0]:.4f}")
    
    # Get the final clustering with optimal k
    final_kmeans = KMeans(n_clusters=int(optimal_k), random_state=42, n_init=10)
    optimal_labels = final_kmeans.fit_predict(embeddings)
    
    return results_df, int(optimal_k), optimal_labels

def plot_cohesion_analysis(results_df, optimal_k, save_path=None):
    # Simple visualization of the clustering analysis
    # results_df: DataFrame returned from find_optimal_semantic_clusters
    # optimal_k: the optimal k value found
    # save_path: optional path to save the plot (e.g., 'images/cohesion_analysis.png')
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Silhouette score (the metric we're maximizing)
    axes[0].plot(results_df['k'], results_df['silhouette_score'], 'b-o', markersize=4, linewidth=1.5)
    axes[0].axvline(optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[0].set_ylabel('Silhouette Score', fontsize=12)
    axes[0].set_title('Silhouette Score vs Number of Clusters\n(Higher is better, range: -1 to 1)', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Average within-cluster distance (cohesion - lower is better)
    axes[1].plot(results_df['k'], results_df['avg_within_cluster_distance'], 'g-o', markersize=4, linewidth=1.5)
    axes[1].axvline(optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[1].set_ylabel('Avg Within-Cluster Distance', fontsize=12)
    axes[1].set_title('Cohesion: Average Distance Within Clusters\n(Lower = tighter clusters)', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    return fig

if __name__ == "__main__":
    # Example: Find optimal number of clusters
    print("Loading character descriptions...")
    descs = load_character_descriptions()
    
    if descs:
        # Use a subset for testing (remove [:50] to use all data)
        test_titles = list(descs.keys())[:100]  # Using 100 for faster testing
        test_texts = [descs[t] for t in test_titles]
        
        print(f"Generating embeddings for {len(test_texts)} characters...")
        embs = generate_embeddings(test_texts)
        
        # Find optimal number of clusters
        results_df, optimal_k, optimal_clusters = find_optimal_semantic_clusters(embs, k_range=(2, 20))
        
        # Visualize results
        print("\nCreating visualization...")
        plot_cohesion_analysis(results_df, optimal_k, save_path='images/cohesion_analysis.png')
        
        # Extract keywords for the optimal clustering
        print(f"\nExtracting keywords for optimal clustering (k={optimal_k})...")
        keywords = extract_cluster_keywords(test_texts, optimal_clusters)
        
        print("\nTop keywords for each cluster:")
        for cluster_id, keyword_str in sorted(keywords.items()):
            cluster_size = sum(optimal_clusters == cluster_id)
            print(f"  Cluster {cluster_id} ({cluster_size} characters): {keyword_str}")
