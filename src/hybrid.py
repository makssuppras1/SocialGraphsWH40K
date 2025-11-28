import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def calculate_neighbor_consistency(G, embeddings, node_to_idx):
    """
    Calculate Neighbor Semantic Consistency: Average cosine similarity between
    a node's embedding and its neighbors' embeddings.
    
    G: NetworkX Graph
    embeddings: numpy array of shape (n_nodes, embedding_dim)
    node_to_idx: dict mapping node name to row index in embeddings
    """
    consistency_scores = {}
    nodes = list(G.nodes())
    
    print(f"Calculating Neighbor Semantic Consistency for {len(nodes)} nodes...")
    count = 0
    
    for node in nodes:
        if node not in node_to_idx:
            consistency_scores[node] = np.nan
            continue
            
        node_idx = node_to_idx[node]
        node_emb = embeddings[node_idx].reshape(1, -1)
        
        # Using successors (outgoing links) as per "links to" description
        neighbors = list(G.neighbors(node))
        
        if not neighbors:
            consistency_scores[node] = np.nan # No neighbors -> undefined consistency
            continue
            
        neighbor_indices = [node_to_idx[n] for n in neighbors if n in node_to_idx]
        
        if not neighbor_indices:
            consistency_scores[node] = np.nan
            continue
            
        neighbor_embs = embeddings[neighbor_indices]
        
        # Compute cosine similarity
        # cosine_similarity returns matrix [1, n_neighbors]
        sims = cosine_similarity(node_emb, neighbor_embs)[0]
        avg_sim = np.mean(sims)
        
        consistency_scores[node] = avg_sim
        
        count += 1
        if count % 1000 == 0:
            print(f"Processed {count} nodes...")
            
    return consistency_scores

def calculate_correlations(df):
    """Calculate correlations between metrics."""
    print("Calculating correlations...")
    # Focus: Betweenness vs Neighbor Consistency
    cols = ['betweenness', 'neighbor_consistency']
    if all(col in df.columns for col in cols):
        # Drop NaNs
        clean_df = df.dropna(subset=cols)
        if len(clean_df) > 0:
            correlation = clean_df['betweenness'].corr(clean_df['neighbor_consistency'], method='pearson')
            print(f"Correlation (Betweenness vs Consistency): {correlation:.4f}")
            return correlation
        else:
            print("Not enough data for correlation.")
    else:
        print(f"Missing columns for correlation. Available: {df.columns}")
    return None

