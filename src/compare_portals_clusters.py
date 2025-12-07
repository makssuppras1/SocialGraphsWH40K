#!/usr/bin/env python3
# compares portal-based factions (16) vs semantic clusters (16)

import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from config import load_network, load_results
from helpers import filter_missing_values

def get_portal_from_node(G, node_id):
    # Extract the first portal from a node's portal attribute
    node_data = G.nodes.get(node_id, {})
    portals = node_data.get('portals', '')
    
    if not portals:
        return None
    
    # Portals are stored as comma-separated string, split and get first one
    portals_str = str(portals)
    portal_parts = portals_str.split(',')
    
    # Clean up portal parts and get first valid one
    portal_list = []
    for portal_part in portal_parts:
        cleaned = portal_part.strip()
        if cleaned:
            portal_list.append(cleaned)
    
    # Just return the first portal if we have any
    if len(portal_list) > 0:
        return portal_list[0]
    
    return None

def create_portal_cluster_comparison(df, G):
    # creates comparison between portals and semantic clusters
    # Filter to only nodes with semantic data (inner join ensures this)
    df = df.dropna(subset=['semantic_cluster_id']).copy()
    
    # add portal column to dataframe
    portal_list = []
    for node_id in df['node_id']:
        portal = get_portal_from_node(G, node_id)
        portal_list.append(portal)
    df['portal'] = portal_list
    
    # keep only rows with both portal and cluster
    valid_df = filter_missing_values(df, ['portal', 'semantic_cluster_id'])
    
    # Get unique portals and clusters manually
    portal_set = set()
    cluster_set = set()
    for idx in valid_df.index:
        portal_set.add(valid_df.loc[idx, 'portal'])
        cluster_set.add(valid_df.loc[idx, 'semantic_cluster_id'])
    portals = sorted(list(portal_set))
    clusters = sorted(list(cluster_set))
    confusion_matrix = np.zeros((len(portals), len(clusters)))
    
    for idx in valid_df.index:
        row = valid_df.loc[idx]
        portal = row['portal']
        cluster = int(row['semantic_cluster_id'])
        if portal in portals and cluster in clusters:
            portal_idx = portals.index(portal)
            cluster_idx = clusters.index(cluster)
            confusion_matrix[portal_idx, cluster_idx] += 1
    
    # calculate statistics
    portal_stats = []
    for portal in portals:
        portal_chars = valid_df[valid_df['portal'] == portal]
        n_chars = len(portal_chars)
        if n_chars > 0:
            # Find most common cluster manually
            cluster_counts = {}
            for idx in portal_chars.index:
                cluster_id = portal_chars.loc[idx, 'semantic_cluster_id']
                if cluster_id not in cluster_counts:
                    cluster_counts[cluster_id] = 0
                cluster_counts[cluster_id] += 1
            
            # Find the cluster with highest count
            max_count = 0
            dominant_cluster = None
            for cluster_id in cluster_counts.keys():
                if cluster_counts[cluster_id] > max_count:
                    max_count = cluster_counts[cluster_id]
                    dominant_cluster = cluster_id
            
            dominant_count = max_count
            dominant_pct = (dominant_count / n_chars * 100)
        else:
            dominant_cluster = None
            dominant_pct = 0
        
        # Build stats dict
        stats_dict = {
            'Portal': portal,
            'Characters': n_chars,
            'Clusters': len(set(portal_chars['semantic_cluster_id'].tolist()))
        }
        if dominant_cluster is not None:
            stats_dict['Dominant Cluster'] = int(dominant_cluster)
        else:
            stats_dict['Dominant Cluster'] = None
        stats_dict['Dominant %'] = f"{dominant_pct:.1f}%"
        portal_stats.append(stats_dict)
    
    portal_stats_df = pd.DataFrame(portal_stats)
    
    # Create normalized matrix (row-normalized: percentage of each portal in each cluster)
    num_portals = len(portals)
    num_clusters = len(clusters)
    normalized_matrix = np.zeros((num_portals, num_clusters), dtype=float)
    for i in range(num_portals):
        row_sum = confusion_matrix[i, :].sum()
        if row_sum > 0:
            for j in range(num_clusters):
                normalized_matrix[i, j] = (confusion_matrix[i, j] / row_sum) * 100
        else:
            for j in range(num_clusters):
                normalized_matrix[i, j] = 0
    
    # visualize raw count confusion matrix
    # create cluster labels
    cluster_labels = []
    for c in clusters:
        cluster_labels.append(f'C{int(c)}')
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(confusion_matrix, 
                annot=True, 
                fmt='.0f', 
                cmap='YlOrRd',
                xticklabels=cluster_labels,
                yticklabels=portals,
                cbar_kws={'label': 'Number of characters'},
                linewidths=0.5)
    plt.xlabel('Semantic Clusters', fontsize=12, fontweight='bold')
    plt.ylabel('Portals (Factions)', fontsize=12, fontweight='bold')
    plt.title('Portal vs Semantic Cluster Comparison (Raw Counts)\n(16 Factions vs 16 Text-Based Clusters)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = Path("images/portal_cluster_confusion_matrix.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # visualize normalized confusion matrix (row-normalized percentages)
    plt.figure(figsize=(14, 10))
    sns.heatmap(normalized_matrix, 
                annot=True, 
                fmt='.1f', 
                cmap='YlOrRd',
                xticklabels=cluster_labels,
                yticklabels=portals,
                cbar_kws={'label': 'Percentage of portal characters'},
                linewidths=0.5,
                vmin=0,
                vmax=100)
    plt.xlabel('Semantic Clusters', fontsize=12, fontweight='bold')
    plt.ylabel('Portals (Factions)', fontsize=12, fontweight='bold')
    plt.title('Portal vs Semantic Cluster Comparison (Normalized)\n(% of each faction in each semantic cluster)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path_norm = Path("images/portal_cluster_confusion_matrix_normalized.png")
    output_path_norm.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path_norm, dpi=300, bbox_inches='tight')
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
    # sort by alignment percentage (convert string to float)
    alignment_values = []
    for val in alignment_df['Alignment %']:
        val_clean = val.rstrip('%')
        alignment_values.append(float(val_clean))
    alignment_df['_sort_key'] = alignment_values
    alignment_df = alignment_df.sort_values('_sort_key', ascending=False)
    alignment_df = alignment_df.drop('_sort_key', axis=1)
    
    print(f"Portal-Cluster Comparison:")
    print(f"  Nodes with both portal and cluster: {len(valid_df)}")
    print(f"  Portals: {len(portals)}")
    print(f"  Semantic clusters: {len(clusters)}")
    print()
    print("Portal Statistics:")
    print(portal_stats_df.to_string(index=False))
    print()
    print("Portal Alignment (sorted by alignment %):")
    print(alignment_df.to_string(index=False))
    
    # create column labels for dataframes
    cluster_col_labels = []
    for c in clusters:
        cluster_col_labels.append(f'Cluster {int(c)}')
    
    confusion_df = pd.DataFrame(confusion_matrix, index=portals, columns=cluster_col_labels)
    normalized_df = pd.DataFrame(normalized_matrix, index=portals, columns=cluster_col_labels)
    return confusion_df, portal_stats_df, normalized_df

def main():
    G_full = load_network(use_filtered=True)
    df = load_results()
    
    # Filter to only nodes with semantic data
    semantic_nodes = set(df.dropna(subset=['semantic_cluster_id'])['node_id'])
    G = G_full.subgraph(semantic_nodes).copy()
    
    create_portal_cluster_comparison(df, G)

if __name__ == "__main__":
    main()
