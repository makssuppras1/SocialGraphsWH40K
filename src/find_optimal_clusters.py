

import sys
import time
import json
import pandas as pd
from pathlib import Path

# Force immediate output (no buffering)
sys.stdout.reconfigure(line_buffering=True)

# Add parent directory to path so we can import from src
sys.path.append(str(Path(__file__).parent.parent))

from src.semantics import (
    load_character_descriptions,
    generate_embeddings,
    find_optimal_semantic_clusters,
    plot_cohesion_analysis,
    extract_cluster_keywords
)

def main():
    script_start_time = time.time()
    print("Starting optimal cluster analysis...", flush=True)
    print(flush=True)
    
    # load character descriptions
    # remove_names=True removes character names to focus on context/themes rather than name mentions
    print("Loading character descriptions (removing character names)...", flush=True)
    print("  This may take 30-60 seconds...", flush=True)
    sys.stdout.flush()
    start_time = time.time()
    descriptions = load_character_descriptions(remove_names=True)
    elapsed = time.time() - start_time
    print(f"Loaded {len(descriptions)} character descriptions in {elapsed:.1f}s", flush=True)
    print("Character names have been removed to focus on semantic context/themes", flush=True)
    print(flush=True)
    
    if not descriptions or len(descriptions) != 3232:
        print("Error: Incorrect number of characters loaded")
        return
    
    # Get the character names and texts
    character_names = list(descriptions.keys())
    character_texts = [descriptions[name] for name in character_names]
    
    print(f"Processing {len(character_texts)} character descriptions", flush=True)
    print(flush=True)
    
    # Generate embeddings (this converts text to numbers)
    print("Generating embeddings...", flush=True)
    print("  This is the longest step - converting text to numerical vectors", flush=True)
    print("  Progress bar will show below:", flush=True)
    sys.stdout.flush()
    start_time = time.time()
    embeddings = generate_embeddings(character_texts)
    elapsed = time.time() - start_time
    print(f"\nEmbeddings generated in {elapsed:.1f}s ({elapsed/60:.1f} minutes)", flush=True)
    print(f"Embeddings shape: {embeddings.shape}", flush=True)
    print(flush=True)
    
    # Find optimal number of clusters
    # k_range=(2, 40) means test from 2 to 40 clusters
    print("Testing k from 2 to 40...", flush=True)
    print("  Estimated time: 2-4 minutes with parallel processing", flush=True)
    print("  Testing 39 different k values (you'll see progress below):", flush=True)
    sys.stdout.flush()
    start_time = time.time()
    results_df, optimal_k, optimal_labels = find_optimal_semantic_clusters(
        embeddings, 
        k_range=(2, 40)
    )
    elapsed = time.time() - start_time
    print(f"\nClustering analysis completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)", flush=True)
    
    print()
    print(f"Optimal number of clusters: {optimal_k}")
    print(f"Total characters clustered: {len(optimal_labels)}")
    print()
    
    plot_cohesion_analysis(
        results_df, 
        optimal_k, 
        save_path='images/cohesion_analysis.png'
    )
    print("Visualization saved to images/cohesion_analysis.png")
    print()
    
    print("Extracting keywords for each cluster...")
    keywords = extract_cluster_keywords(character_texts, optimal_labels, n_keywords=5)
    print()
    
    for cluster_id in sorted(keywords.keys()):
        # count how many characters are in this cluster
        cluster_size = 0
        for label in optimal_labels:
            if label == cluster_id:
                cluster_size += 1
        print(f"Cluster {cluster_id} ({cluster_size} characters):")
        print(f"Keywords: {keywords[cluster_id]}")
        print()
    
    print("Cluster examples:")
    unique_clusters = sorted(set(optimal_labels))
    for cluster_id in unique_clusters:
        # find all indices for this cluster
        cluster_indices = []
        for i, label in enumerate(optimal_labels):
            if label == cluster_id:
                cluster_indices.append(i)
        
        # get first 5 example names
        example_names = []
        for i in cluster_indices[:5]:
            example_names.append(character_names[i])
        print(f"Cluster {cluster_id}: {', '.join(example_names)}")
        if len(cluster_indices) > 5:
            remaining = len(cluster_indices) - 5
            print(f"  ... and {remaining} more")
        print()
    
    # Save results to JSON files
    print("Saving results to data files...")
    
    # Save 1: Character-to-cluster assignments
    cluster_assignments = {
        'optimal_k': int(optimal_k),
        'total_characters': len(optimal_labels),
        'silhouette_score': float(results_df.loc[results_df['k']==optimal_k, 'silhouette_score'].values[0]),
        'avg_within_cluster_distance': float(results_df.loc[results_df['k']==optimal_k, 'avg_within_cluster_distance'].values[0]),
        'assignments': {}
    }
    
    for i, (char_name, cluster_id) in enumerate(zip(character_names, optimal_labels)):
        cluster_assignments['assignments'][char_name] = {
            'semantic_cluster_id': int(cluster_id),
            'top_keywords': keywords.get(int(cluster_id), '')
        }
    
    assignments_path = Path(__file__).parent.parent / "data" / "optimal_semantic_clusters.json"
    with open(assignments_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_assignments, f, indent=2, ensure_ascii=False)
    print(f"  Saved character assignments to {assignments_path.name}")
    
    # Save 2: Clustering evaluation results (all k values tested)
    evaluation_results = {
        'k_range': {'min': int(results_df['k'].min()), 'max': int(results_df['k'].max())},
        'optimal_k': int(optimal_k),
        'evaluations': []
    }
    
    for _, row in results_df.iterrows():
        evaluation_results['evaluations'].append({
            'k': int(row['k']),
            'silhouette_score': float(row['silhouette_score']),
            'avg_within_cluster_distance': float(row['avg_within_cluster_distance']),
            'is_optimal': bool(row['k'] == optimal_k)
        })
    
    evaluation_path = Path(__file__).parent.parent / "data" / "clustering_evaluation_results.json"
    with open(evaluation_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    print(f"  Saved evaluation results to {evaluation_path.name}")
    
    # Save 3: Cluster summaries with keywords and examples
    cluster_summaries = {
        'optimal_k': int(optimal_k),
        'clusters': {}
    }
    
    unique_clusters = sorted(set(optimal_labels))
    for cluster_id in unique_clusters:
        # find all indices for this cluster
        cluster_indices = []
        for i, label in enumerate(optimal_labels):
            if label == cluster_id:
                cluster_indices.append(i)
        
        # get character names for this cluster
        cluster_characters = []
        for i in cluster_indices:
            cluster_characters.append(character_names[i])
        
        cluster_summaries['clusters'][int(cluster_id)] = {
            'size': len(cluster_characters),
            'keywords': keywords.get(int(cluster_id), ''),
            'characters': cluster_characters
        }
    
    summaries_path = Path(__file__).parent.parent / "data" / "optimal_cluster_summaries.json"
    with open(summaries_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_summaries, f, indent=2, ensure_ascii=False)
    print(f"  Saved cluster summaries to {summaries_path.name}")
    print()
    
    total_time = time.time() - script_start_time
    print(f"Analysis complete! Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

if __name__ == "__main__":
    main()

