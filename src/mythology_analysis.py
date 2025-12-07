import os
import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
import multiprocessing

# Suppress tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from config import load_network, load_results
from semantics import load_character_descriptions, generate_embeddings, extract_cluster_keywords
from helpers import filter_missing_values, calculate_percentile, simple_sort_descending

def compute_topology_metrics(G, approximate_betweenness=True, k_samples=200):
    # Calculate network topology metrics: betweenness, PageRank, clustering, communities
    # Using approximate betweenness for speed - full calculation takes forever on this network
    if approximate_betweenness:
        print(f"  Computing approximate betweenness (sampling {k_samples} nodes)...")
        betweenness = nx.betweenness_centrality(G, k=k_samples)
    else:
        print("  Computing exact betweenness (this may take a while)...")
        betweenness = nx.betweenness_centrality(G)

    print("  Computing PageRank...")
    pagerank = nx.pagerank(G)

    print("  Computing clustering coefficients...")
    clustering = nx.clustering(G)

    print("  Detecting communities (Louvain algorithm)...")
    G_undirected = G.to_undirected()
    partition = nx.community.louvain_communities(G_undirected)
    print(f"{len(partition)} communities")
    
    node_to_community = {}
    for comm_id, community in enumerate(partition):
        for node in community:
            node_to_community[node] = comm_id
    
    nodes = list(G.nodes())
    betweenness_values = []
    pagerank_values = []
    clustering_values = []
    community_ids = []
    
    for n in nodes:
        betweenness_values.append(betweenness.get(n, 0))
        pagerank_values.append(pagerank.get(n, 0))
        clustering_values.append(clustering.get(n, 0))
        community_ids.append(node_to_community.get(n, -1))
    
    return pd.DataFrame({
        'node_id': nodes,
        'betweenness': betweenness_values,
        'pagerank': pagerank_values,
        'clustering_coefficient': clustering_values,
        'network_community_id': community_ids
    })

def _calculate_node_consistency(node, G, embeddings, node_to_idx):
    # Helper for parallel processing
    if node not in node_to_idx:
        return node, np.nan
        
    node_idx = node_to_idx[node]
    node_emb = embeddings[node_idx].reshape(1, -1)
    neighbors = list(G.neighbors(node))
    
    if not neighbors:
        return node, np.nan
        
    neighbor_indices = []
    for n in neighbors:
        if n in node_to_idx:
            neighbor_indices.append(node_to_idx[n])
    
    if not neighbor_indices:
        return node, np.nan
        
    neighbor_embs = embeddings[neighbor_indices]
    sims = cosine_similarity(node_emb, neighbor_embs)[0]
    return node, np.mean(sims)

def calculate_neighbor_consistency(G, embeddings, node_to_idx, n_jobs=None):
    # Calculate neighbor semantic consistency: average cosine similarity between
    # a node's embedding and its neighbors' embeddings
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    
    nodes = list(G.nodes())
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_calculate_node_consistency)(node, G, embeddings, node_to_idx)
        for node in nodes
    )
    
    # Convert to dict
    consistency_scores = {}
    for node, score in results:
        consistency_scores[node] = score
    
    return consistency_scores

def calculate_correlations(df):
    # Calculate correlation between betweenness and neighbor consistency
    cols = ['betweenness', 'neighbor_consistency']
    has_both_cols = True
    for col in cols:
        if col not in df.columns:
            has_both_cols = False
            break
    
    if has_both_cols:
        clean_df = filter_missing_values(df, ['betweenness', 'neighbor_consistency'])
        
        if len(clean_df) > 0:
            correlation = clean_df['betweenness'].corr(clean_df['neighbor_consistency'], method='pearson')
            print(f"Correlation (Betweenness vs Consistency): {correlation:.4f}")
            if correlation < -0.1:
                print("  Negative correlation: High betweenness tends to have lower consistency (mythological anchors)")
            elif correlation > 0.1:
                print("  Positive correlation: High betweenness tends to have higher consistency")
            else:
                print("  Weak correlation: Betweenness and consistency are largely independent")
            return correlation
    return None

def plot_mythology_scatter(df, output_path='images/mythology_scatter.png'):
    # Scatter plot: x-axis = betweenness centrality (log scale), y-axis = neighbor semantic consistency
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    # Tried different styles, whitegrid looks cleanest for this
    
    fig, ax = plt.subplots(figsize=(14, 9))
    # Remove rows with missing values
    plot_df = filter_missing_values(df, ['betweenness', 'neighbor_consistency', 'semantic_cluster_id'])
    
    scatter = ax.scatter(
        plot_df['betweenness'] + 1e-8,
        plot_df['neighbor_consistency'],
        c=plot_df['semantic_cluster_id'],
        cmap='tab20',
        alpha=0.7,
        s=60,
        edgecolors='white',
        linewidth=0.5,
        zorder=3
    )
        
    ax.set_xscale('log')
    
    # Add trend line - log scale makes this a bit tricky
    x_data = plot_df['betweenness'] + 1e-8
    y_data = plot_df['neighbor_consistency']
    log_x = np.log10(x_data)
    coeffs = np.polyfit(log_x, y_data, 1)
    poly = np.poly1d(coeffs)
    # Generate points for trend line
    x_min = x_data.min()
    x_max = x_data.max()
    x_trend = np.logspace(np.log10(x_min), np.log10(x_max), 100)
    y_trend = poly(np.log10(x_trend))
    ax.plot(x_trend, y_trend, 'r--', linewidth=2, alpha=0.8, label='Trend line', zorder=2)
    
    ax.set_xlabel('Betweenness Centrality (log scale)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Neighbor Semantic Consistency', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title('Mythology vs Narrative: Structural Centrality vs Semantic Coherence\n' +
                '(Colored by Semantic Cluster)', fontsize=15, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Semantic Cluster ID', fontsize=11, fontweight='bold')
    
    x_median = plot_df['betweenness'].median()
    y_median = plot_df['neighbor_consistency'].median()
    ax.axvline(x=x_median, color='gray', linestyle=':', alpha=0.5, linewidth=1, zorder=1)
    ax.axhline(y=y_median, color='gray', linestyle=':', alpha=0.5, linewidth=1, zorder=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def load_optimal_clustering():
    # Load pre-computed semantic clustering (k=26)
    data_path = Path(__file__).parent.parent / "data" / "optimal_semantic_clusters.json"
    
    if not data_path.exists():
        print(f"Error: Optimal clustering file not found at {data_path}")
        return None
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    assignments = data['assignments']
    
    # Convert to DataFrame for easier merging
    results = []
    for node_id in assignments.keys():
        info = assignments[node_id]
        cluster_id = info['semantic_cluster_id']
        keywords = info['top_keywords']
        results.append({
            'node_id': node_id,
            'semantic_cluster_id': cluster_id,
            'top_keywords': keywords
        })
    
    return pd.DataFrame(results)

def export_results(df, output_path='data/mythology_vs_narrative_analysis.csv'):
    # Save analysis results to CSV file
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent.parent / output_path
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Select columns to export
    columns = [
        'node_id', 'betweenness', 'pagerank', 'network_community_id', 
        'semantic_cluster_id', 'top_keywords', 'neighbor_consistency'
    ]
    export_cols = []
    for c in columns:
        if c in df.columns:
            export_cols.append(c)
    
    df[export_cols].to_csv(output_path, index=False)
    print(f"  Saved results to {output_path.name}")

def identify_mythological_anchors(df, betweenness_threshold=None, consistency_threshold=None):
    # Identify mythological anchors: characters with high betweenness but low semantic consistency
    # These characters bridge different narrative arcs but don't share semantic themes with neighbors
    
    if betweenness_threshold is None:
        betweenness_values = df['betweenness'].tolist()
        betweenness_threshold = calculate_percentile(betweenness_values, 0.90)
    if consistency_threshold is None:
        consistency_values = df['neighbor_consistency'].tolist()
        consistency_threshold = calculate_percentile(consistency_values, 0.20)
    
    # Filter for high betweenness, low consistency
    # Doing this in two steps because chaining was confusing
    candidates = df.copy()
    candidates = candidates[candidates['betweenness'] >= betweenness_threshold]
    candidates = candidates[candidates['neighbor_consistency'] <= consistency_threshold]
    
    if len(candidates) > 0:
        # Sort by betweenness descending
        candidate_list = []
        for idx in candidates.index:
            row = candidates.loc[idx]
            candidate_list.append((idx, row['betweenness']))
        candidate_list = simple_sort_descending(candidate_list, key_index=1)
        
        print(f"\nFound {len(candidates)} mythological anchor candidates:")
        print(f"  (Betweenness >= {betweenness_threshold:.6f}, Consistency <= {consistency_threshold:.4f})")
        # Show top 10
        top_10_indices = []
        for i in range(min(10, len(candidate_list))):
            top_10_indices.append(candidate_list[i][0])
        
        for idx in top_10_indices:
            row = candidates.loc[idx]
            node_name = row['node_id']
            bet_val = row['betweenness']
            cons_val = row['neighbor_consistency']
            print(f"  {node_name:40s} | Betweenness: {bet_val:8.6f} | "
                  f"Consistency: {cons_val:.4f}")
    
    return candidates

def display_cluster_summary(df, top_n=10):
    # Display summary statistics for semantic clusters
    # Do grouping manually instead of using groupby
    cluster_data = {}
    for idx in df.index:
        row = df.loc[idx]
        cluster_id = row['semantic_cluster_id']
        if cluster_id not in cluster_data:
            cluster_data[cluster_id] = {
                'count': 0,
                'keywords': row['top_keywords'],
                'betweenness_sum': 0.0,
                'consistency_sum': 0.0
            }
        cluster_data[cluster_id]['count'] += 1
        cluster_data[cluster_id]['betweenness_sum'] += row['betweenness']
        cluster_data[cluster_id]['consistency_sum'] += row['neighbor_consistency']
    
    # Calculate averages
    clusters_list = []
    for cluster_id in cluster_data.keys():
        data = cluster_data[cluster_id]
        avg_bet = data['betweenness_sum'] / data['count']
        avg_cons = data['consistency_sum'] / data['count']
        clusters_list.append({
            'cluster_id': cluster_id,
            'count': data['count'],
            'keywords': data['keywords'],
            'avg_betweenness': round(avg_bet, 4),
            'avg_consistency': round(avg_cons, 4)
        })
    
    # Sort by count descending
    for i in range(len(clusters_list)):
        for j in range(i + 1, len(clusters_list)):
            if clusters_list[i]['count'] < clusters_list[j]['count']:
                temp = clusters_list[i]
                clusters_list[i] = clusters_list[j]
                clusters_list[j] = temp
    
    print(f"\nTop {top_n} Semantic Clusters (by size):")
    for i in range(min(top_n, len(clusters_list))):
        cluster = clusters_list[i]
        print(f"  Cluster {int(cluster['cluster_id'])}: {cluster['count']} characters")
        print(f"    Keywords: {cluster['keywords']}")
        print(f"    Avg Betweenness: {cluster['avg_betweenness']:.6f}, Avg Consistency: {cluster['avg_consistency']:.4f}")
        print()

def compare_characters(df, char_names):
    # Compare specific characters side-by-side
    # Find matching characters manually
    found_chars = df.copy()
    matching_indices = []
    for idx in found_chars.index:
        node_id = found_chars.loc[idx, 'node_id']
        if node_id in char_names:
            matching_indices.append(idx)
    
    if len(matching_indices) == 0:
        print("No matching characters found.")
        return
    
    found_chars = found_chars.loc[matching_indices]

    for idx in found_chars.index:
        char = found_chars.loc[idx]
        node_name = char['node_id']
        bet_val = char['betweenness']
        cons_val = char['neighbor_consistency']
        cluster_id = int(char['semantic_cluster_id'])
        keywords = str(char['top_keywords'])
        if len(keywords) > 60:
            keywords = keywords[:60]
        print(f"  {node_name:40s}")
        print(f"    Betweenness: {bet_val:8.6f} | "
              f"Consistency: {cons_val:.4f} | "
              f"Cluster: {cluster_id}")
        print(f"    Keywords: {keywords}")
        print()

def show_statistics(df):
    # Display overall statistics from the analysis
    print("\nANALYSIS STATISTICS")
    print(f"Total characters analyzed: {len(df)}")
    # Count unique clusters manually
    unique_clusters = set()
    for idx in df.index:
        unique_clusters.add(df.loc[idx, 'semantic_cluster_id'])
    print(f"Semantic clusters: {len(unique_clusters)}")
    
    # Count unique communities manually
    unique_communities = set()
    for idx in df.index:
        unique_communities.add(df.loc[idx, 'network_community_id'])
    print(f"Network communities: {len(unique_communities)}")
    print(f"\nBetweenness Centrality:")
    print(f"  Mean: {df['betweenness'].mean():.6f}")
    print(f"  Max: {df['betweenness'].max():.6f}")
    print(f"  Min: {df['betweenness'].min():.6f}")
    print(f"\nNeighbor Semantic Consistency:")
    print(f"  Mean: {df['neighbor_consistency'].mean():.4f}")
    print(f"  Min: {df['neighbor_consistency'].min():.4f}")
    print(f"  Max: {df['neighbor_consistency'].max():.4f}")

def run_full_analysis():
    
    # Phase 1: Load semantic data first to filter network
    semantics_df = load_optimal_clustering()
    
    semantic_nodes = set(semantics_df['node_id'])
    print(f"Loaded clustering for {len(semantic_nodes)} characters")

    G = load_network(use_filtered=True)
    
    # Filter network to only nodes with semantic data
    G_filtered = G.subgraph(semantic_nodes).copy()
    print(f"  - Filtered network to {G_filtered.number_of_nodes()} nodes with semantic data")
    print("  - Calculating network metrics (this may take a few minutes)...")
    topology_df = compute_topology_metrics(G_filtered)
    print(f"Network: {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")
    print(f"Found {topology_df['network_community_id'].nunique()} network communities")
    print()
    
    # Merge topology and semantics - should be 1:1 match now
    # Using pandas merge - tried doing it manually but it was getting messy
    print("Merging topology and semantic data...")
    combined_df = pd.merge(topology_df, semantics_df, on='node_id', how='inner')
    print(f"Combined data for {len(combined_df)} characters")
    print()
    
    # Load descriptions for characters - only use nodes in combined_df (already filtered to semantic data)
    print("  - Loading character descriptions...")
    descriptions = load_character_descriptions(valid_nodes=set(combined_df['node_id']), remove_names=True)
    
    # All nodes in combined_df should have descriptions (they have semantic data)
    valid_nodes = list(combined_df['node_id'])
    
    if not valid_nodes:
        print("Error: No intersection between graph nodes and text descriptions.")
        return None
    
    # Generate embeddings for consistency calculation
    print(f"  - Generating embeddings for {len(valid_nodes)} characters...")
    valid_texts = []
    for n in valid_nodes:
        valid_texts.append(descriptions[n])
    embeddings = generate_embeddings(valid_texts)
    
    # Map node names to embedding indices - need this for the consistency calc
    node_to_emb_idx = {}
    for i in range(len(valid_nodes)):
        node_to_emb_idx[valid_nodes[i]] = i
    
    # Calculate neighbor semantic consistency using filtered network
    print(f"  - Calculating consistency scores for {len(valid_nodes)} nodes (using parallel processing)")
    consistency_scores = calculate_neighbor_consistency(G_filtered, embeddings, node_to_emb_idx)
    combined_df['neighbor_consistency'] = combined_df['node_id'].map(consistency_scores)
    
    calculate_correlations(combined_df)
    print()
    
    plot_mythology_scatter(combined_df)
    export_results(combined_df)
    
    return combined_df

def explore_results():
    
    df = load_results()
    show_statistics(df)
    print()
    identify_mythological_anchors(df)
    print()
    display_cluster_summary(df, top_n=10)

    interesting_chars = [
        "Roboute Guilliman", "Emperor of Mankind", "Ibram Gaunt",
        "Rogal Dorn", "Sanguinius", "Lion El'Jonson", "Angron", "Ezekyle Abaddon"
    ]
    compare_characters(df, interesting_chars)

if __name__ == "__main__":
    df = run_full_analysis()
    if df is not None:
        show_statistics(df)
        identify_mythological_anchors(df)
        display_cluster_summary(df, top_n=5)
