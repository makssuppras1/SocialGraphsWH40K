#!/usr/bin/env python3
# compares portal-based factions (16) vs semantic clusters (16)

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

def load_network_with_portals():
    from config import PICKLE_FILTERED_FILE
    path = Path(PICKLE_FILTERED_FILE)
    if not path.exists():
        path = Path(__file__).parent.parent / PICKLE_FILTERED_FILE
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_results():
    csv_path = Path("data/mythology_vs_narrative_analysis.csv")
    if not csv_path.exists():
        csv_path = Path(__file__).parent.parent / "data/mythology_vs_narrative_analysis.csv"
    return pd.read_csv(csv_path)

def get_portal_from_node(G, node_id):
    node_data = G.nodes.get(node_id, {})
    portals = node_data.get('portals', '')
    if portals:
        portal_list = [p.strip() for p in str(portals).split(',') if p.strip()]
        if portal_list:
            return portal_list[0]
    return None

def create_portal_cluster_comparison(df, G):
    # creates comparison between portals and semantic clusters
    df['portal'] = df['node_id'].apply(lambda x: get_portal_from_node(G, x))
    valid_df = df.dropna(subset=['portal', 'semantic_cluster_id']).copy()
    
    portals = sorted(valid_df['portal'].unique())
    clusters = sorted(valid_df['semantic_cluster_id'].unique())
    confusion_matrix = np.zeros((len(portals), len(clusters)))
    
    for idx, row in valid_df.iterrows():
        portal = row['portal']
        cluster = int(row['semantic_cluster_id'])
        if portal in portals and cluster in clusters:
            confusion_matrix[portals.index(portal), clusters.index(cluster)] += 1
    
    # calculate statistics
    portal_stats = []
    for portal in portals:
        portal_chars = valid_df[valid_df['portal'] == portal]
        n_chars = len(portal_chars)
        if n_chars > 0:
            dominant_cluster = portal_chars['semantic_cluster_id'].mode()[0]
            dominant_count = len(portal_chars[portal_chars['semantic_cluster_id'] == dominant_cluster])
            dominant_pct = (dominant_count / n_chars * 100)
        else:
            dominant_cluster = None
            dominant_pct = 0
        
        portal_stats.append({
            'Portal': portal,
            'Characters': n_chars,
            'Clusters': portal_chars['semantic_cluster_id'].nunique(),
            'Dominant Cluster': int(dominant_cluster) if dominant_cluster is not None else None,
            'Dominant %': f"{dominant_pct:.1f}%"
        })
    
    portal_stats_df = pd.DataFrame(portal_stats)
    
    # visualize confusion matrix
    plt.figure(figsize=(14, 10))
    sns.heatmap(confusion_matrix, 
                annot=True, 
                fmt='.0f', 
                cmap='YlOrRd',
                xticklabels=[f'C{int(c)}' for c in clusters],
                yticklabels=portals,
                cbar_kws={'label': 'Number of characters'},
                linewidths=0.5)
    plt.xlabel('Semantic Clusters', fontsize=12, fontweight='bold')
    plt.ylabel('Portals (Factions)', fontsize=12, fontweight='bold')
    plt.title('Portal vs Semantic Cluster Comparison\n(16 Factions vs 16 Text-Based Clusters)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = Path("images/portal_cluster_confusion_matrix.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # calculate alignment metrics
    portal_alignment = []
    for portal in portals:
        portal_chars = valid_df[valid_df['portal'] == portal]
        if len(portal_chars) > 0:
            dominant_cluster = portal_chars['semantic_cluster_id'].mode()[0]
            alignment = len(portal_chars[portal_chars['semantic_cluster_id'] == dominant_cluster]) / len(portal_chars) * 100
            portal_alignment.append({
                'Portal': portal,
                'Alignment %': f"{alignment:.1f}%",
                'Dominant Cluster': int(dominant_cluster)
            })
    
    alignment_df = pd.DataFrame(portal_alignment)
    alignment_df = alignment_df.sort_values('Alignment %', key=lambda x: x.str.rstrip('%').astype(float), ascending=False)
    
    print(f"Nodes with both portal and cluster: {len(valid_df)}")
    print("\nPortal Statistics:")
    print(portal_stats_df.to_string(index=False))
    print("\nPortal Alignment:")
    print(alignment_df.to_string(index=False))
    
    return pd.DataFrame(confusion_matrix, index=portals, columns=[f'Cluster {int(c)}' for c in clusters]), portal_stats_df, None

def main():
    G = load_network_with_portals()
    df = load_results()
    create_portal_cluster_comparison(df, G)

if __name__ == "__main__":
    main()
