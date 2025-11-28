#!/usr/bin/env python3
"""
Compare Portal-based Factions (16) vs Semantic Clusters (16).
This allows direct comparison between affiliation-based grouping and text-based grouping.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

def load_network_with_portals():
    """Load the network to get portal information."""
    from config import PICKLE_FILTERED_FILE
    
    path = Path(PICKLE_FILTERED_FILE)
    if not path.exists():
        path = Path(__file__).parent.parent / PICKLE_FILTERED_FILE
    
    with open(path, 'rb') as f:
        G = pickle.load(f)
    
    return G

def load_results():
    """Load the mythology analysis results."""
    csv_path = Path("data/mythology_vs_narrative_analysis.csv")
    if not csv_path.exists():
        csv_path = Path(__file__).parent.parent / "data/mythology_vs_narrative_analysis.csv"
    
    df = pd.read_csv(csv_path)
    return df

def get_portal_from_node(G, node_id):
    """Extract portal from node attributes."""
    node_data = G.nodes.get(node_id, {})
    portals = node_data.get('portals', '')
    if portals:
        # Portals are stored as comma-separated string
        portal_list = [p.strip() for p in str(portals).split(',') if p.strip()]
        if portal_list:
            return portal_list[0]  # Use first portal
    return None

def create_portal_cluster_comparison(df, G):
    """Create comparison between portals and semantic clusters."""
    print("="*80)
    print("PORTAL vs SEMANTIC CLUSTER COMPARISON")
    print("="*80)
    
    # Add portal information to dataframe
    df['portal'] = df['node_id'].apply(lambda x: get_portal_from_node(G, x))
    
    # Filter to nodes with both portal and cluster
    valid_df = df.dropna(subset=['portal', 'semantic_cluster_id']).copy()
    
    print(f"\nNodes with both portal and semantic cluster: {len(valid_df)}")
    print(f"Nodes without portal: {len(df) - len(valid_df)}")
    
    # Create confusion matrix: Portal (rows) vs Semantic Cluster (columns)
    portals = sorted(valid_df['portal'].unique())
    clusters = sorted(valid_df['semantic_cluster_id'].unique())
    
    confusion_matrix = np.zeros((len(portals), len(clusters)))
    
    for idx, row in valid_df.iterrows():
        portal = row['portal']
        cluster = int(row['semantic_cluster_id'])
        
        if portal in portals and cluster in clusters:
            portal_idx = portals.index(portal)
            cluster_idx = clusters.index(cluster)
            confusion_matrix[portal_idx, cluster_idx] += 1
    
    # Create DataFrame for easier viewing
    confusion_df = pd.DataFrame(
        confusion_matrix,
        index=portals,
        columns=[f'Cluster {int(c)}' for c in clusters]
    )
    
    print("\n" + "="*80)
    print("CONFUSION MATRIX: Portals (Rows) vs Semantic Clusters (Columns)")
    print("="*80)
    print("\n(Values show number of characters in each Portal-Cluster combination)")
    print(confusion_df.to_string())
    
    # Calculate statistics
    print("\n" + "="*80)
    print("PORTAL STATISTICS")
    print("="*80)
    
    portal_stats = []
    for portal in portals:
        portal_chars = valid_df[valid_df['portal'] == portal]
        n_chars = len(portal_chars)
        n_clusters = portal_chars['semantic_cluster_id'].nunique()
        dominant_cluster = portal_chars['semantic_cluster_id'].mode()[0] if len(portal_chars) > 0 else None
        dominant_count = len(portal_chars[portal_chars['semantic_cluster_id'] == dominant_cluster]) if dominant_cluster is not None else 0
        dominant_pct = (dominant_count / n_chars * 100) if n_chars > 0 else 0
        
        portal_stats.append({
            'Portal': portal,
            'Characters': n_chars,
            'Clusters': n_clusters,
            'Dominant Cluster': int(dominant_cluster) if dominant_cluster is not None else None,
            'Dominant %': f"{dominant_pct:.1f}%"
        })
    
    portal_stats_df = pd.DataFrame(portal_stats)
    print(portal_stats_df.to_string(index=False))
    
    # Calculate cluster statistics
    print("\n" + "="*80)
    print("SEMANTIC CLUSTER STATISTICS")
    print("="*80)
    
    cluster_stats = []
    for cluster in clusters:
        cluster_chars = valid_df[valid_df['semantic_cluster_id'] == cluster]
        n_chars = len(cluster_chars)
        n_portals = cluster_chars['portal'].nunique()
        dominant_portal = cluster_chars['portal'].mode()[0] if len(cluster_chars) > 0 else None
        dominant_count = len(cluster_chars[cluster_chars['portal'] == dominant_portal]) if dominant_portal is not None else 0
        dominant_pct = (dominant_count / n_chars * 100) if n_chars > 0 else 0
        
        cluster_stats.append({
            'Cluster': int(cluster),
            'Characters': n_chars,
            'Portals': n_portals,
            'Dominant Portal': dominant_portal,
            'Dominant %': f"{dominant_pct:.1f}%"
        })
    
    cluster_stats_df = pd.DataFrame(cluster_stats)
    print(cluster_stats_df.to_string(index=False))
    
    # Visualize confusion matrix
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
    print(f"\nSaved confusion matrix to {output_path}")
    plt.close()
    
    # Calculate alignment metrics
    print("\n" + "="*80)
    print("ALIGNMENT METRICS")
    print("="*80)
    
    # For each portal, what % of characters are in the dominant cluster?
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
    print("\nPortal Alignment (how well each portal maps to a single semantic cluster):")
    print(alignment_df.to_string(index=False))
    
    return confusion_df, portal_stats_df, cluster_stats_df

def main():
    """Main function."""
    print("="*80)
    print("PORTAL vs SEMANTIC CLUSTER COMPARISON ANALYSIS")
    print("="*80)
    
    # Load data
    G = load_network_with_portals()
    df = load_results()
    
    # Create comparison
    confusion_df, portal_stats, cluster_stats = create_portal_cluster_comparison(df, G)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Insights:")
    print("- High alignment % = Portal characters are semantically similar (text matches faction)")
    print("- Low alignment % = Portal characters are semantically diverse (text doesn't match faction)")
    print("- Multiple clusters per portal = Faction spans multiple semantic themes")
    print("- Multiple portals per cluster = Different factions share semantic themes")

if __name__ == "__main__":
    main()

