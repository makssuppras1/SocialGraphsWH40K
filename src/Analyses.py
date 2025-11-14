#!/usr/bin/env python3
"""
Network Analysis Script

Performs comprehensive analysis on the Warhammer 40k character network:
- Degree distribution analysis
- Centrality measures (degree, betweenness, eigenvector)
- Centrality correlations
- Degree assortativity analysis
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import pearsonr

from config import DATA_PATH, PICKLE_FILE, PICKLE_FILTERED_FILE


def load_network(use_filtered=True):
    """Load the network from pickle file.
    
    Args:
        use_filtered: If True, loads the filtered network (undirected, no isolated nodes).
                     If False, loads the original network.
    """
    network_file = PICKLE_FILTERED_FILE if use_filtered else PICKLE_FILE
    
    if not network_file.exists():
        raise FileNotFoundError(f"Network file not found: {network_file}")
    
    with open(network_file, 'rb') as f:
        G = pickle.load(f)
    
    return G


def analyze_degree_distribution(network):
    """Analyze and plot degree distribution."""
    print("="*60)
    print("DEGREE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Get all node degrees
    degrees = [network.degree(node) for node in network.nodes()]
    
    # Print basic statistics
    print("\nDegree Statistics:")
    print(f"  Average: {np.mean(degrees):.2f}")
    print(f"  Median: {np.median(degrees):.2f}")
    print(f"  Maximum: {max(degrees)}")
    print(f"  Minimum: {min(degrees)}")
    print(f"  Nodes with degree 0: {sum(1 for d in degrees if d == 0)}")
    
    # Create two plots: regular and log-log
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Regular histogram
    axes[0].hist(degrees, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Degree')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Degree Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Log-log plot
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    axes[1].loglog(unique_degrees, counts, 'o', markersize=6)
    axes[1].set_xlabel('Degree (log scale)')
    axes[1].set_ylabel('Frequency (log scale)')
    axes[1].set_title('Degree Distribution (Log-Log)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_centrality(network, network_directed=None):
    """Analyze centrality measures and their correlations."""
    print("\n" + "="*60)
    print("CENTRALITY ANALYSIS")
    print("="*60)
    
    # Use directed network for eigenvector centrality if available, otherwise convert
    if network_directed is None:
        network_directed = network
        if not isinstance(network_directed, nx.DiGraph):
            # Convert to directed for eigenvector (will use undirected if needed)
            network_directed = network.to_directed()
    
    # Calculate all centralities
    print("\nCalculating centralities...")
    centrality_degree = nx.degree_centrality(network)
    centrality_betweenness = nx.betweenness_centrality(network)
    
    # Eigenvector centrality requires connected graph or max_iter
    try:
        centrality_eigenvector = nx.eigenvector_centrality(network, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        print("Warning: Eigenvector centrality failed to converge, using undirected version")
        network_undir = network.to_undirected() if isinstance(network, nx.DiGraph) else network
        centrality_eigenvector = nx.eigenvector_centrality(network_undir, max_iter=1000)
    
    # Show top 5 nodes for each centrality measure
    print("\nTop 5 by Degree Centrality:")
    for node, value in sorted(centrality_degree.items(), key=lambda x: x[1], reverse=True)[:5]:
        node_name = network.nodes[node].get('name', node)
        print(f"  {node_name}: {value:.4f}")
    
    print("\nTop 5 by Betweenness Centrality:")
    for node, value in sorted(centrality_betweenness.items(), key=lambda x: x[1], reverse=True)[:5]:
        node_name = network.nodes[node].get('name', node)
        print(f"  {node_name}: {value:.4f}")
    
    print("\nTop 5 by Eigenvector Centrality:")
    for node, value in sorted(centrality_eigenvector.items(), key=lambda x: x[1], reverse=True)[:5]:
        node_name = network.nodes[node].get('name', node)
        print(f"  {node_name}: {value:.4f}")
    
    # Get centrality values in same order for all nodes
    nodes = list(network.nodes())
    degree_values = [centrality_degree[n] for n in nodes]
    betweenness_values = [centrality_betweenness[n] for n in nodes]
    eigenvector_values = [centrality_eigenvector[n] for n in nodes]
    
    # Calculate correlations
    corr_deg_bet, _ = pearsonr(degree_values, betweenness_values)
    corr_deg_eig, _ = pearsonr(degree_values, eigenvector_values)
    corr_bet_eig, _ = pearsonr(betweenness_values, eigenvector_values)
    
    print("\n=== Correlations Between Centralities ===")
    print(f"Degree vs Betweenness: r = {corr_deg_bet:.3f}")
    print(f"Degree vs Eigenvector: r = {corr_deg_eig:.3f}")
    print(f"Betweenness vs Eigenvector: r = {corr_bet_eig:.3f}")
    
    # Plot correlations
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].scatter(degree_values, betweenness_values, alpha=0.6, s=30)
    axes[0].set_xlabel('Degree Centrality')
    axes[0].set_ylabel('Betweenness Centrality')
    axes[0].set_title(f'Degree vs Betweenness (r = {corr_deg_bet:.3f})')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(degree_values, eigenvector_values, alpha=0.6, s=30)
    axes[1].set_xlabel('Degree Centrality')
    axes[1].set_ylabel('Eigenvector Centrality')
    axes[1].set_title(f'Degree vs Eigenvector (r = {corr_deg_eig:.3f})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_assortativity(network):
    """Analyze degree assortativity."""
    print("\n" + "="*60)
    print("DEGREE ASSORTATIVITY ANALYSIS")
    print("="*60)
    
    # Calculate assortativity coefficient
    assortativity = nx.degree_assortativity_coefficient(network)
    print(f"\nDegree assortativity coefficient: {assortativity:.3f}")
    
    if assortativity > 0:
        print("The network is assortative: high-degree nodes connect to other high-degree nodes.")
    elif assortativity < 0:
        print("The network is disassortative: high-degree nodes connect to low-degree nodes.")
    else:
        print("The network shows no degree assortativity.")
    
    # Get node degrees and average neighbor degrees
    node_degrees = [network.degree(n) for n in network.nodes()]
    avg_neighbor_degrees = nx.average_neighbor_degree(network)
    neighbor_degrees = [avg_neighbor_degrees[n] for n in network.nodes()]
    
    # Plot degree vs neighbor degree
    plt.figure(figsize=(10, 6))
    plt.scatter(node_degrees, neighbor_degrees, alpha=0.5, s=30)
    plt.xlabel('Node Degree')
    plt.ylabel('Average Neighbor Degree')
    plt.title(f'Degree Assortativity (r = {assortativity:.3f})')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    coefficients = np.polyfit(node_degrees, neighbor_degrees, 1)
    trendline = np.poly1d(coefficients)
    plt.plot(sorted(node_degrees), trendline(sorted(node_degrees)), "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main execution function."""
    # Load filtered network (undirected, no isolated nodes)
    print("Loading filtered network (undirected, no isolated nodes)...")
    G = load_network(use_filtered=True)
    
    # The filtered network is already undirected
    if isinstance(G, nx.DiGraph):
        print("Converting directed graph to undirected for analysis...")
        G_undirected = G.to_undirected()
    else:
        G_undirected = G
    
    print(f"\nNetwork loaded:")
    print(f"  Nodes: {G_undirected.number_of_nodes()}")
    print(f"  Edges: {G_undirected.number_of_edges()}")
    print(f"  Type: {'Undirected' if not isinstance(G, nx.DiGraph) else 'Directed'}")
    
    # Run analyses
    analyze_degree_distribution(G_undirected)
    analyze_centrality(G_undirected, G if isinstance(G, nx.DiGraph) else G_undirected)
    analyze_assortativity(G_undirected)


if __name__ == "__main__":
    main()
