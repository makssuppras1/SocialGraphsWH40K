

import sys
from pathlib import Path

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
    
    # load character descriptions
    descriptions = load_character_descriptions()
    print(f"{len(descriptions)} = 3,232 characters")
    print()
    
    if not descriptions or len(descriptions) != 3232:
        print("Error: Incorrect number of characters loaded")
        return
    
    # Step 2: Get the character names and texts
    character_names = list(descriptions.keys())
    character_texts = [descriptions[name] for name in character_names]
    
    print(f"Processing {len(character_texts)} character descriptions")
    print()
    
    # Step 3: Generate embeddings (this converts text to numbers)
    print("Generating embeddings (this may take a few minutes)...")
    embeddings = generate_embeddings(character_texts)
    print(f"Embeddings shape: {embeddings.shape}")
    print()
    
    # Step 4: Find optimal number of clusters
    # k_range=(2, 40) means test from 2 to 40 clusters
    # Adjust max_k based on your Louvain community count
    print("Optimal number of clusters")
    
    results_df, optimal_k, optimal_labels = find_optimal_semantic_clusters(
        embeddings, 
        k_range=(2, 40)  # Test from 2 to 40 clusters
    )
    
    print()
    print(f"Optimal number of clusters: {optimal_k}")
    print(f"Total characters clustered: {len(optimal_labels)}")
    print()
    
    plot_cohesion_analysis(
        results_df, 
        optimal_k, 
        save_path='images/cohesion_analysis.png'
    )
    print()
    
    print("Cluster keyword.")
    keywords = extract_cluster_keywords(character_texts, optimal_labels, n_keywords=5)
    
    for cluster_id in sorted(keywords.keys()):
        cluster_size = sum(optimal_labels == cluster_id)
        print(f"Cluster {cluster_id} ({cluster_size} characters):")
        print(f"Keywords: {keywords[cluster_id]}")
        print()
    
    print("Examples")
    for cluster_id in sorted(set(optimal_labels)):
        cluster_indices = [i for i, label in enumerate(optimal_labels) if label == cluster_id]
        example_names = [character_names[i] for i in cluster_indices[:5]]  # First 5
        print(f"Cluster {cluster_id}: {', '.join(example_names)}")
        if len(cluster_indices) > 5:
            print(f"  ... and {len(cluster_indices) - 5} more")
        print()
    
    print("Done!")

if __name__ == "__main__":
    main()

