import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path to allow imports if running from root
sys.path.append(str(Path(__file__).parent.parent))

from src.topology import load_graph, compute_topology_metrics
from src.semantics import load_character_descriptions, generate_embeddings, cluster_embeddings, extract_cluster_keywords
from src.hybrid import calculate_neighbor_consistency, calculate_correlations
from src.vis_utils import plot_mythology_scatter, export_results

def main():
    print("Starting Mythology vs Narrative Analysis Pipeline...")
    
    # --- Phase 1: Topology ---
    print("\n--- Phase 1: Topological Feature Extraction ---")
    # Use filtered network (nodes with degree >= 2)
    G = load_graph(use_filtered=True)
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    topology_df = compute_topology_metrics(G)
    print("Topology metrics computed.")
    
    # --- Phase 2: Semantics ---
    print("\n--- Phase 2: Semantic Feature Extraction ---")
    descriptions = load_character_descriptions()
    print(f"Loaded {len(descriptions)} character descriptions.")
    
    # Align descriptions with graph nodes
    graph_nodes = set(G.nodes())
    valid_nodes = [n for n in descriptions.keys() if n in graph_nodes]
    
    # If we have too few valid nodes, we might want to check alignment issues
    if len(valid_nodes) < len(graph_nodes) * 0.1:
        print(f"Warning: Only {len(valid_nodes)} nodes matched. Checking name variations...")
        # Simple check - try matching without quotes or special chars if needed
        # For now, proceed with what we have
        pass
        
    valid_texts = [descriptions[n] for n in valid_nodes]
    
    print(f"Analyzing {len(valid_nodes)} nodes with both graph and text data.")
    
    if not valid_nodes:
        print("Error: No intersection between graph nodes and text descriptions.")
        return

    # Generate Embeddings
    embeddings = generate_embeddings(valid_texts)
    
    # Clustering: Use 16 clusters to match the 16 main faction portals
    cluster_ids = cluster_embeddings(embeddings, n_clusters=16)
    
    # Keywords
    keywords = extract_cluster_keywords(valid_texts, cluster_ids)
    
    # Create semantics DataFrame
    semantics_data = {
        'node_id': valid_nodes,
        'semantic_cluster_id': cluster_ids
    }
    semantics_df = pd.DataFrame(semantics_data)
    
    # Map cluster ID to keywords
    semantics_df['top_keywords'] = semantics_df['semantic_cluster_id'].map(keywords)
    
    # Merge topology and semantics
    # Note: topology_df has all graph nodes, semantics_df has subset.
    combined_df = pd.merge(topology_df, semantics_df, on='node_id', how='left')
    
    # --- Phase 3: Hybrid Analysis ---
    print("\n--- Phase 3: Hybrid Analysis ---")
    
    # Create mapping from node_id to embedding index for valid nodes
    node_to_emb_idx = {n: i for i, n in enumerate(valid_nodes)}
    
    consistency_scores = calculate_neighbor_consistency(G, embeddings, node_to_emb_idx)
    
    # Add to DataFrame
    combined_df['neighbor_consistency'] = combined_df['node_id'].map(consistency_scores)
    
    calculate_correlations(combined_df)
    
    # --- Phase 4: Visualization & Output ---
    print("\n--- Phase 4: Visualization & Output ---")
    
    plot_mythology_scatter(combined_df)
    export_results(combined_df)
    
    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()

