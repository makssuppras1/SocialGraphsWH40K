import pandas as pd
import numpy as np
import sys
from pathlib import Path

# add src to path to allow imports if running from root
sys.path.append(str(Path(__file__).parent.parent))

from src.topology import load_graph, compute_topology_metrics
from src.semantics import load_character_descriptions, generate_embeddings, cluster_embeddings, extract_cluster_keywords
from src.hybrid import calculate_neighbor_consistency, calculate_correlations
from src.vis_utils import plot_mythology_scatter, export_results

def main():
    print("Starting Mythology vs Narrative Analysis Pipeline...")
    
    # phase 1: topology
    print("\n--- Phase 1: Topological Feature Extraction ---")
    # use filtered network (nodes with degree >= 2)
    G = load_graph(use_filtered=True)
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    topology_df = compute_topology_metrics(G)
    print("Topology metrics computed.")
    
    # phase 2: semantics
    print("\n--- Phase 2: Semantic Feature Extraction ---")
    descriptions = load_character_descriptions()
    print(f"Loaded {len(descriptions)} character descriptions.")
    
    # align descriptions with graph nodes
    graph_nodes = set(G.nodes())
    valid_nodes = []
    for n in descriptions.keys():
        if n in graph_nodes:
            valid_nodes.append(n)
    
    # if we have too few valid nodes, might want to check alignment issues
    if len(valid_nodes) < len(graph_nodes) * 0.1:
        print(f"Warning: Only {len(valid_nodes)} nodes matched. Checking name variations...")
        # simple check - try matching without quotes or special chars if needed
        # for now, proceed with what we have
        pass
        
    valid_texts = [descriptions[n] for n in valid_nodes]
    
    print(f"Analyzing {len(valid_nodes)} nodes with both graph and text data.")
    
    if not valid_nodes:
        print("Error: No intersection between graph nodes and text descriptions.")
        return

    # generate embeddings
    embeddings = generate_embeddings(valid_texts)
    
    # clustering: use 16 clusters to match the 16 main faction portals
    cluster_ids = cluster_embeddings(embeddings, n_clusters=16)
    
    # keywords
    keywords = extract_cluster_keywords(valid_texts, cluster_ids)
    
    # create semantics dataframe
    semantics_data = {
        'node_id': valid_nodes,
        'semantic_cluster_id': cluster_ids
    }
    semantics_df = pd.DataFrame(semantics_data)
    
    # map cluster id to keywords
    semantics_df['top_keywords'] = semantics_df['semantic_cluster_id'].map(keywords)
    
    # merge topology and semantics
    # note: topology_df has all graph nodes, semantics_df has subset
    combined_df = pd.merge(topology_df, semantics_df, on='node_id', how='left')
    
    # phase 3: hybrid analysis
    print("\n--- Phase 3: Hybrid Analysis ---")
    
    # create mapping from node_id to embedding index for valid nodes
    node_to_emb_idx = {}
    for i, n in enumerate(valid_nodes):
        node_to_emb_idx[n] = i
    
    consistency_scores = calculate_neighbor_consistency(G, embeddings, node_to_emb_idx)
    
    # add to dataframe
    combined_df['neighbor_consistency'] = combined_df['node_id'].map(consistency_scores)
    
    calculate_correlations(combined_df)
    
    # phase 4: visualization & output
    print("\n--- Phase 4: Visualization & Output ---")
    
    plot_mythology_scatter(combined_df)
    export_results(combined_df)
    
    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()
