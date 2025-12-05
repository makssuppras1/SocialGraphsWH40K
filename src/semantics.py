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

def get_all_character_names(valid_nodes=None):
    # Get all character names from the network for name removal
    # Returns a set of character names (titles and display names) to remove from text
    character_names = set()
    
    # Try to load from network first
    try:
        from config import PICKLE_FILTERED_FILE
        filter_path = Path(PICKLE_FILTERED_FILE)
    except ImportError:
        filter_path = Path(__file__).parent.parent / "data" / "lexicanum_network_filtered.pkl"
    
    if not filter_path.exists():
        filter_path = Path(__file__).parent.parent / "data" / "lexicanum_network_filtered.pkl"
    
    if filter_path.exists():
        with open(filter_path, 'rb') as f:
            G = pickle.load(f)
        
        for node in G.nodes():
            if valid_nodes is None or node in valid_nodes:
                # Add the full title
                character_names.add(node.lower())
                # Add display name (title without parenthetical info)
                display_name = re.sub(r'\s*\([^)]*\)$', '', node).strip()
                character_names.add(display_name.lower())
                # Add individual words from multi-word names (e.g., "Roboute Guilliman" -> "roboute", "guilliman")
                words = display_name.split()
                for word in words:
                    # Only add words that are likely names (capitalized, >2 chars)
                    if len(word) > 2:
                        character_names.add(word.lower())
    else:
        # Fallback: try to load from characters JSON file
        chars_path = Path(__file__).parent.parent / "data" / "lexicanum_characters.json"
        if chars_path.exists():
            with open(chars_path, 'r', encoding='utf-8') as f:
                characters = json.load(f)
            for char in characters:
                title = char.get('title', '')
                name = char.get('name', '')
                if valid_nodes is None or title in valid_nodes:
                    character_names.add(title.lower())
                    character_names.add(name.lower())
                    # Add individual words
                    for word in name.split():
                        if len(word) > 2:
                            character_names.add(word.lower())
    
    return character_names

def clean_wikitext(text, character_names_to_remove=None):
    # cleans wikitext markup to get plain text
    # character_names_to_remove: set of character names (lowercase) to remove from text
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
    
    # Remove character names if provided
    if character_names_to_remove:
        text_lower = text.lower()
        # Sort names by length (longest first) to avoid partial matches
        sorted_names = sorted(character_names_to_remove, key=len, reverse=True)
        for name in sorted_names:
            # Use word boundaries to avoid removing parts of words
            # Pattern: word boundary, name, word boundary (case insensitive)
            pattern = r'\b' + re.escape(name) + r'\b'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        # Clean up extra spaces after removal
        text = re.sub(r'\s+', ' ', text).strip()
    
    return text.lower()

def load_character_descriptions(raw_data_path='raw_data', batch_pattern='lexicanum_page_texts_batch_*.json', valid_nodes=None, remove_names=True, cache_file=None):
    # loads and cleans descriptions from raw batch files
    # if valid_nodes is provided (set of node IDs), only includes characters in that set
    # remove_names: if True, removes character names from text to focus on context/themes
    # cache_file: path to cache file to save/load processed descriptions (saves time on subsequent runs)
    
    # Check for cached version first
    if cache_file is None:
        cache_file = Path(__file__).parent.parent / "data" / "character_descriptions_processed.json"
    else:
        cache_file = Path(cache_file)
    
    # Load filtered network nodes first (needed for cache check and processing)
    if valid_nodes is None:
        try:
            try:
                from config import PICKLE_FILTERED_FILE
                filter_path = Path(PICKLE_FILTERED_FILE)
            except ImportError:
                filter_path = Path(__file__).parent.parent / "data" / "lexicanum_network_filtered.pkl"
            
            if not filter_path.exists():
                filter_path = Path(__file__).parent.parent / "data" / "lexicanum_network_filtered.pkl"
            
            if filter_path.exists():
                with open(filter_path, 'rb') as f:
                    G_filtered = pickle.load(f)
                valid_nodes = set(G_filtered.nodes())
            else:
                valid_nodes = None
        except Exception as e:
            valid_nodes = None
    
    # If processed descriptions file exists, load from it
    if cache_file.exists() and remove_names:
        try:
            print(f"  Loading processed descriptions from {cache_file.name}...", flush=True)
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            # Filter by valid_nodes if provided
            if valid_nodes is not None:
                descriptions = {}
                for k, v in cached_data.items():
                    if k in valid_nodes:
                        descriptions[k] = v
            else:
                descriptions = cached_data
            print(f"  Loaded {len(descriptions)} processed descriptions from data file", flush=True)
            return descriptions
        except Exception as e:
            print(f"  Failed to load processed descriptions ({e}), processing from scratch...", flush=True)
    
    # Process from scratch
    descriptions = {}
    path = Path(raw_data_path)
    if not path.exists():
        path = Path(__file__).parent.parent / raw_data_path
    
    files = list(path.glob(batch_pattern))
    if not files:
        return {}

    # Get all character names to remove (if requested)
    character_names_to_remove = None
    if remove_names:
        print("  Loading character names for removal from text...", flush=True)
        sys.stdout.flush()
        character_names_to_remove = get_all_character_names(valid_nodes)
        print(f"  Found {len(character_names_to_remove)} character name variations to remove", flush=True)
        print("  Removing names from text descriptions (this may take a moment)...", flush=True)
        sys.stdout.flush()

    total_files = len(files)
    processed = 0
    for file_idx, file_path in enumerate(files):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for title, page_data in data.get('pages', {}).items():
                # Only include if in valid_nodes (if provided)
                if valid_nodes is not None and title not in valid_nodes:
                    continue
                    
                wikitext = page_data.get('wikitext', '')
                clean_text = clean_wikitext(wikitext, character_names_to_remove)
                if len(clean_text) > 20:
                    descriptions[title] = clean_text
                    processed += 1
                    # Show progress every 500 characters
                    if processed % 500 == 0:
                        print(f"  Processed {processed} characters...", end='\r', flush=True)
    
    print(f"  Processed {processed} characters total", flush=True)
    
    # Always save processed descriptions (without names) as a data file
    if remove_names and descriptions:
        try:
            print(f"  Saving processed descriptions to {cache_file.name}...", flush=True)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            # Save all descriptions (not filtered by valid_nodes) so cache is reusable
            # But we need to save what we actually processed, so save all we have
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(descriptions, f, indent=2, ensure_ascii=False)
            print(f"  Saved {len(descriptions)} processed descriptions to data file", flush=True)
        except Exception as e:
            print(f"  Warning: Could not save processed descriptions ({e})", flush=True)
    
    return descriptions

def generate_embeddings(texts, model_name='all-MiniLM-L6-v2', batch_size=32):
    # generates embeddings for a list of text strings
    # Shows progress bar to estimate time remaining
    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True)

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
        keywords = {}
        for c in unique_clusters:
            keywords[c] = "error"
        return keywords
        
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
    
    # Find optimal k using parsimony principle: smallest k within threshold of max silhouette
    # This balances quality (high silhouette) with parsimony (fewer clusters)
    # Method: Find k where silhouette is within 95% of maximum, then choose smallest k
    # This is similar to the elbow method but applied to silhouette scores
    max_silhouette = results_df['silhouette_score'].max()
    threshold = max_silhouette * 0.95  # Within 95% of maximum silhouette score
    candidates = results_df[results_df['silhouette_score'] >= threshold]
    
    if len(candidates) > 0:
        # Choose the smallest k that meets the quality threshold (parsimony principle)
        optimal_k = candidates.loc[candidates['k'].idxmin(), 'k']
        print(f"\nOptimal number of clusters: k={optimal_k} (parsimony-adjusted)")
        print(f"  Selection method: Smallest k where silhouette >= {threshold:.4f} (95% of max {max_silhouette:.4f})")
    else:
        # Fallback: if no k meets threshold, use max silhouette
        optimal_k = results_df.loc[results_df['silhouette_score'].idxmax(), 'k']
        print(f"\nOptimal number of clusters: k={optimal_k} (maximum silhouette)")
        print(f"  Note: No k values met 95% threshold, using maximum silhouette score")
    
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
        test_texts = []
        for t in test_titles:
            test_texts.append(descs[t])
        
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
