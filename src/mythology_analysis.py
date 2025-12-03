import os
import pandas as pd
import sys
from pathlib import Path

# Suppress tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.append(str(Path(__file__).parent.parent))

from src.topology import load_graph, compute_topology_metrics
from src.semantics import load_character_descriptions, generate_embeddings, cluster_embeddings, extract_cluster_keywords
from src.hybrid import calculate_neighbor_consistency, calculate_correlations
from src.vis_utils import plot_mythology_scatter#, export_results

def main():
    # phase 1: topology
    G = load_graph(use_filtered=True)
    topology_df = compute_topology_metrics(G)
    
    # phase 2: semantics
    # Load descriptions only for characters in the filtered network
    graph_nodes = set(G.nodes())
    descriptions = load_character_descriptions(valid_nodes=graph_nodes)
    
    # Get intersection (in case some network nodes don't have descriptions)
    valid_nodes = [n for n in descriptions.keys() if n in graph_nodes]
    
    if not valid_nodes:
        print("Error: No intersection between graph nodes and text descriptions.")
        return
    
    valid_texts = [descriptions[n] for n in valid_nodes]
    embeddings = generate_embeddings(valid_texts)
    cluster_ids = cluster_embeddings(embeddings, n_clusters=16)
    keywords = extract_cluster_keywords(valid_texts, cluster_ids)
    
    semantics_df = pd.DataFrame({
        'node_id': valid_nodes,
        'semantic_cluster_id': cluster_ids
    })
    semantics_df['top_keywords'] = semantics_df['semantic_cluster_id'].map(keywords)
    
    # merge topology and semantics
    combined_df = pd.merge(topology_df, semantics_df, on='node_id', how='left')
    
    # phase 3: hybrid analysis
    node_to_emb_idx = {n: i for i, n in enumerate(valid_nodes)}
    consistency_scores = calculate_neighbor_consistency(G, embeddings, node_to_emb_idx)
    combined_df['neighbor_consistency'] = combined_df['node_id'].map(consistency_scores)
    calculate_correlations(combined_df)
    
    # phase 4: visualization & output
    plot_mythology_scatter(combined_df)
    #export_results(combined_df)
    
    print("done")

if __name__ == "__main__":
    main()
