import json
import re
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances, silhouette_score
from joblib import Parallel, delayed
import multiprocessing

from config import load_network
from helpers import get_display_name

def get_all_character_names(valid_nodes=None):
    character_names = set()
    
    G = load_network(use_filtered=True)
    
    for node in G.nodes():
        if valid_nodes is None or node in valid_nodes:
            character_names.add(node.lower())
            display_name = get_display_name(node)
            character_names.add(display_name.lower())
            words = display_name.split()
            for word in words:
                if len(word) > 2:
                    character_names.add(word.lower())
    
    return character_names

def clean_wikitext(text, character_names_to_remove=None):
    if not text:
        return ""
    
    reference_pattern = r'==+\s*(Sources|References|See Also|External Links|Related Articles|Notes|Footnotes)\s*==+.*'
    match = re.search(reference_pattern, text, re.IGNORECASE)
    if match:
        text = text[:match.start()]
    
    text = re.sub(r'\{\{[^}]+\}\}', '', text)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'==+[^=]+==+', '', text)
    text = re.sub(r'\[\[(File|Image):.+?\]\]', '', text)
    text = re.sub(r"'''?", '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if character_names_to_remove:
        text_lower = text.lower()
        sorted_names = sorted(character_names_to_remove, key=len, reverse=True)
        for name in sorted_names:
            pattern = r'\b' + re.escape(name) + r'\b'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
    
    return text.lower()

def load_character_descriptions(raw_data_path='raw_data', batch_pattern='lexicanum_page_texts_batch_*.json', valid_nodes=None, remove_names=True):
    if valid_nodes is None:
        G_filtered = load_network(use_filtered=True)
        valid_nodes = set(G_filtered.nodes())
    
    path = Path(raw_data_path)
    if not path.is_absolute():
        path = Path(__file__).parent.parent / raw_data_path
    
    files = list(path.glob(batch_pattern))
    if not files:
        return {}

    character_names_to_remove = None
    if remove_names:
        character_names_to_remove = get_all_character_names(valid_nodes)

    descriptions = {}
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            pages = data.get('pages', {})
            for title, page_data in pages.items():
                if valid_nodes is not None and title not in valid_nodes:
                    continue
                    
                if 'wikitext' in page_data:
                    wikitext = page_data['wikitext']
                else:
                    wikitext = ''
                clean_text = clean_wikitext(wikitext, character_names_to_remove)
                if len(clean_text) > 20:
                    descriptions[title] = clean_text
    
    return descriptions

def generate_embeddings(texts, model_name='all-MiniLM-L6-v2', batch_size=32, cache_dir='data/embeddings_cache'):
    # Cache embeddings - regenerating takes forever
    import hashlib
    
    cache_path = Path(__file__).parent.parent / cache_dir
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Hash texts to check if we've seen these before
    # Converting to sorted list first - probably not the most efficient but works
    text_strings = []
    for t in texts:
        text_strings.append(str(t))
    texts_str = str(sorted(text_strings))
    cache_key = hashlib.md5(texts_str.encode()).hexdigest()
    cache_file = cache_path / f"{model_name}_{cache_key}.pkl"
    
    if cache_file.exists():
        print(f"  Loading cached embeddings from {cache_file.name}...")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            if cached_data['model'] == model_name and len(cached_data['embeddings']) == len(texts):
                return cached_data['embeddings']
    
    print(f"  Generating embeddings (this may take a few minutes)...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    
    # Save for next run
    with open(cache_file, 'wb') as f:
        pickle.dump({'model': model_name, 'embeddings': embeddings}, f)
    
    return embeddings

def cluster_embeddings(embeddings, n_clusters=20):
    n_samples = len(embeddings)
    if n_samples < n_clusters:
        n_clusters = n_samples
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(embeddings)

def extract_cluster_keywords(texts, cluster_ids, n_keywords=5):
    df = pd.DataFrame({'text': texts, 'cluster': cluster_ids})
    keywords = {}
    
    grouped = df.groupby('cluster')
    cluster_corpus = {}
    for cluster_id in grouped.groups.keys():
        group = grouped.get_group(cluster_id)
        text_list = group['text'].tolist()
        combined_text = ' '.join(text_list)
        cluster_corpus[cluster_id] = combined_text
    
    # Build lists in sorted order for TF-IDF
    corpus_list = []
    cluster_ids_list = []
    sorted_cluster_ids = sorted(cluster_corpus.keys())
    for cluster_id in sorted_cluster_ids:
        corpus_list.append(cluster_corpus[cluster_id])
        cluster_ids_list.append(cluster_id)
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000, max_df=0.9)
    tfidf_matrix = tfidf.fit_transform(corpus_list)
    feature_names = np.array(tfidf.get_feature_names_out())
    
    for i in range(len(cluster_ids_list)):
        cluster_id = cluster_ids_list[i]
        row = tfidf_matrix[i].toarray().flatten()
        if np.all(row == 0):
            keywords[cluster_id] = "no keywords"
            continue
        # Get top keywords by TF-IDF score
        # argsort gives indices sorted by value, we want the highest ones
        sorted_indices = row.argsort()
        top_indices = sorted_indices[-n_keywords:]
        top_indices = top_indices[::-1]  # Reverse to get highest first
        
        keyword_list = []
        for idx in top_indices:
            keyword_list.append(feature_names[idx])
        keywords[cluster_id] = ", ".join(keyword_list)
        
    return keywords

def _test_single_k(embeddings, k, random_state=42, inner_n_jobs=2):
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    silhouette = silhouette_score(embeddings, labels, n_jobs=inner_n_jobs)
    
    total_within_distance = 0
    total_pairs = 0
    
    for cluster_id in range(k):
        cluster_points = embeddings[labels == cluster_id]
        if len(cluster_points) > 1:
            distances = pairwise_distances(cluster_points, n_jobs=1)
            n_points = len(distances)
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    total_within_distance += distances[i, j]
                    total_pairs += 1
    
    if total_pairs > 0:
        avg_within_distance = total_within_distance / total_pairs
    else:
        avg_within_distance = float('inf')
    
    return {
        'k': k,
        'silhouette_score': silhouette,
        'avg_within_cluster_distance': avg_within_distance
    }

def find_optimal_semantic_clusters(embeddings, k_range=(2, 40), n_jobs=None):
    total_cores = multiprocessing.cpu_count()
    
    if n_jobs is None:
        outer_jobs = max(4, int(total_cores * 0.6))
        inner_jobs = max(2, total_cores - outer_jobs)
    else:
        outer_jobs = n_jobs
        inner_jobs = max(2, total_cores - outer_jobs)
    
    max_k = min(k_range[1], len(embeddings))
    min_k = k_range[0]
    
    results = Parallel(n_jobs=outer_jobs, verbose=0)(
        delayed(_test_single_k)(embeddings, k, inner_n_jobs=inner_jobs) 
        for k in range(min_k, max_k + 1)
    )
    
    results_df = pd.DataFrame(results)
    
    max_silhouette = results_df['silhouette_score'].max()
    threshold = max_silhouette * 0.95
    candidates = results_df[results_df['silhouette_score'] >= threshold]
    
    if len(candidates) > 0:
        optimal_k = candidates.loc[candidates['k'].idxmin(), 'k']
    else:
        optimal_k = results_df.loc[results_df['silhouette_score'].idxmax(), 'k']
    
    final_kmeans = KMeans(n_clusters=int(optimal_k), random_state=42, n_init=10)
    optimal_labels = final_kmeans.fit_predict(embeddings)
    
    return results_df, int(optimal_k), optimal_labels

def plot_cohesion_analysis(results_df, optimal_k, save_path=None):
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    axes[0].plot(results_df['k'], results_df['silhouette_score'], 'b-o', markersize=4, linewidth=1.5)
    axes[0].axvline(optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[0].set_ylabel('Silhouette Score', fontsize=12)
    axes[0].set_title('Silhouette Score vs Number of Clusters\n(Higher is better, range: -1 to 1)', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
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
    
    return fig

def main():
    descriptions = load_character_descriptions(remove_names=True)
    
    character_names = list(descriptions.keys())
    character_texts = []
    for name in character_names:
        character_texts.append(descriptions[name])
    
    embeddings = generate_embeddings(character_texts)
    
    results_df, optimal_k, optimal_labels = find_optimal_semantic_clusters(
        embeddings, 
        k_range=(2, 40)
    )
    
    plot_cohesion_analysis(
        results_df, 
        optimal_k, 
        save_path='images/cohesion_analysis.png'
    )
    
    keywords = extract_cluster_keywords(character_texts, optimal_labels, n_keywords=5)
    
    cluster_assignments = {
        'optimal_k': int(optimal_k),
        'total_characters': len(optimal_labels),
        'silhouette_score': float(results_df.loc[results_df['k']==optimal_k, 'silhouette_score'].values[0]),
        'avg_within_cluster_distance': float(results_df.loc[results_df['k']==optimal_k, 'avg_within_cluster_distance'].values[0]),
        'assignments': {}
    }
    
    for i in range(len(character_names)):
        char_name = character_names[i]
        cluster_id = optimal_labels[i]
        if int(cluster_id) in keywords:
            top_keywords = keywords[int(cluster_id)]
        else:
            top_keywords = ''
        cluster_assignments['assignments'][char_name] = {
            'semantic_cluster_id': int(cluster_id),
            'top_keywords': top_keywords
        }
    
    assignments_path = Path(__file__).parent.parent / "data" / "optimal_semantic_clusters.json"
    with open(assignments_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_assignments, f, indent=2, ensure_ascii=False)
    
    evaluation_results = {
        'k_range': {'min': int(results_df['k'].min()), 'max': int(results_df['k'].max())},
        'optimal_k': int(optimal_k),
        'evaluations': []
    }
    
    for index in results_df.index:
        row = results_df.loc[index]
        evaluation_results['evaluations'].append({
            'k': int(row['k']),
            'silhouette_score': float(row['silhouette_score']),
            'avg_within_cluster_distance': float(row['avg_within_cluster_distance']),
            'is_optimal': bool(row['k'] == optimal_k)
        })
    
    evaluation_path = Path(__file__).parent.parent / "data" / "clustering_evaluation_results.json"
    with open(evaluation_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    cluster_summaries = {
        'optimal_k': int(optimal_k),
        'clusters': {}
    }
    
    unique_clusters = sorted(set(optimal_labels))
    for cluster_id in unique_clusters:
        cluster_indices = []
        for i, label in enumerate(optimal_labels):
            if label == cluster_id:
                cluster_indices.append(i)
        
        cluster_characters = []
        for i in cluster_indices:
            cluster_characters.append(character_names[i])
        
        if int(cluster_id) in keywords:
            cluster_keywords = keywords[int(cluster_id)]
        else:
            cluster_keywords = ''
        cluster_summaries['clusters'][int(cluster_id)] = {
            'size': len(cluster_characters),
            'keywords': cluster_keywords,
            'characters': cluster_characters
        }
    
    summaries_path = Path(__file__).parent.parent / "data" / "optimal_cluster_summaries.json"
    with open(summaries_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_summaries, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
