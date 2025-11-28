import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def calculate_neighbor_consistency(G, embeddings, node_to_idx):
    # calculates neighbor semantic consistency: average cosine similarity between
    # a node's embedding and its neighbors' embeddings
    consistency_scores = {}
    
    for node in G.nodes():
        if node not in node_to_idx:
            consistency_scores[node] = np.nan
            continue
            
        node_idx = node_to_idx[node]
        node_emb = embeddings[node_idx].reshape(1, -1)
        neighbors = list(G.neighbors(node))
        
        if not neighbors:
            consistency_scores[node] = np.nan
            continue
            
        neighbor_indices = [node_to_idx[n] for n in neighbors if n in node_to_idx]
        
        if not neighbor_indices:
            consistency_scores[node] = np.nan
            continue
            
        neighbor_embs = embeddings[neighbor_indices]
        sims = cosine_similarity(node_emb, neighbor_embs)[0]
        consistency_scores[node] = np.mean(sims)
            
    return consistency_scores

def calculate_correlations(df):
    # calculates correlations between metrics
    cols = ['betweenness', 'neighbor_consistency']
    if all(col in df.columns for col in cols):
        clean_df = df.dropna(subset=cols)
        if len(clean_df) > 0:
            correlation = clean_df['betweenness'].corr(clean_df['neighbor_consistency'], method='pearson')
            print(f"Correlation (Betweenness vs Consistency): {correlation:.4f}")
            return correlation
    return None
