#!/usr/bin/env python3
"""
Network Analysis Script

Performs comprehensive analysis on the Warhammer 40k character network:
- Degree distribution analysis
- Centrality measures (degree, betweenness, eigenvector)
- Centrality correlations
- Degree assortativity analysis
- Network backbone extraction
- Modularity and community detection

All plots are saved to the 'images' folder.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import pearsonr
from collections import defaultdict, Counter
from networkx.algorithms import community

from config import DATA_PATH, PICKLE_FILE, PICKLE_FILTERED_FILE, IMAGES_PATH
from helpers import MAIN_PORTALS


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
    plt.savefig(IMAGES_PATH / "degree_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {IMAGES_PATH / 'degree_distribution.png'}")


def analyze_centrality(network, network_directed=None):
    """Analyze centrality measures and their correlations."""
    print("\n" + "="*60)
    print("CENTRALITY ANALYSIS")
    print("="*60)
    
    # Use directed network for eigenvector centrality if available, otherwise convert
    if network_directed is None:
        network_directed = network
        if not isinstance(network_directed, nx.DiGraph):
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
    plt.savefig(IMAGES_PATH / "centrality_correlations.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {IMAGES_PATH / 'centrality_correlations.png'}")


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
    plt.savefig(IMAGES_PATH / "assortativity.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {IMAGES_PATH / 'assortativity.png'}")


def extract_backbone(network, threshold=0.05):
    """Extract network backbone using disparity filter.
    
    Args:
        network: NetworkX graph
        threshold: Minimum relative edge weight to keep (default: 0.05)
    
    Returns:
        Backbone network with weak edges removed
    """
    print("\n" + "="*60)
    print("NETWORK BACKBONE EXTRACTION")
    print("="*60)
    
    # Calculate edge betweenness centrality
    print("Calculating edge betweenness centrality...")
    edge_betweenness = nx.edge_betweenness_centrality(network)
    
    # Normalize edge betweenness to range [1, 100] for weights
    if not edge_betweenness:
        print("Warning: No edges found in network")
        return network.copy()
    
    min_eb = min(edge_betweenness.values())
    max_eb = max(edge_betweenness.values())
    
    if max_eb == min_eb:
        print("Warning: All edges have same betweenness, using original network")
        return network.copy()
    
    edge_weights = {}
    for edge, eb in edge_betweenness.items():
        edge_weights[edge] = 1 + 99 * (eb - min_eb) / (max_eb - min_eb)
    
    # Create weighted network
    network_weighted = network.copy()
    nx.set_edge_attributes(network_weighted, edge_weights, 'weight')
    
    # Filter out weak edges
    edges_to_remove = []
    for node in network_weighted.nodes():
        node_edges = list(network_weighted.edges(node))
        
        if len(node_edges) > 1:
            weights = [network_weighted[edge[0]][edge[1]]['weight'] for edge in node_edges]
            total_weight = sum(weights)
            
            for edge, weight in zip(node_edges, weights):
                relative_weight = weight / total_weight if total_weight > 0 else 0
                if relative_weight < threshold:
                    edges_to_remove.append(edge)
    
    # Create backbone
    backbone = network_weighted.copy()
    backbone.remove_edges_from(edges_to_remove)
    backbone.remove_nodes_from(list(nx.isolates(backbone)))
    
    print(f"\nOriginal network: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
    print(f"Backbone network: {backbone.number_of_nodes()} nodes, {backbone.number_of_edges()} edges")
    print(f"Edges removed: {(1 - backbone.number_of_edges()/network.number_of_edges())*100:.1f}%")
    
    # Get largest component for visualization
    if backbone.number_of_nodes() > 0:
        components = list(nx.connected_components(backbone))
        largest_component = max(components, key=len)
        backbone_viz = backbone.subgraph(largest_component).copy()
        print(f"Largest component: {backbone_viz.number_of_nodes()} nodes, {backbone_viz.number_of_edges()} edges")
        
        # Visualize backbone (if not too large)
        if backbone_viz.number_of_nodes() <= 500:
            try:
                pos = nx.spring_layout(backbone_viz, k=1, iterations=50, seed=42)
                
                # Calculate node sizes based on degree
                degrees = [backbone_viz.degree(n) for n in backbone_viz.nodes()]
                min_degree = min(degrees) if degrees else 1
                max_degree = max(degrees) if degrees else 1
                
                node_sizes = []
                for degree in degrees:
                    if max_degree > min_degree:
                        size = 20 + 180 * (degree - min_degree) / (max_degree - min_degree)
                    else:
                        size = 20
                    node_sizes.append(size)
                
                plt.figure(figsize=(14, 12))
                nx.draw_networkx_nodes(backbone_viz, pos, 
                                     node_size=node_sizes, 
                                     node_color='steelblue', 
                                     alpha=0.8, 
                                     linewidths=0.5, 
                                     edgecolors='black')
                
                nx.draw_networkx_edges(backbone_viz, pos, 
                                     alpha=0.4, 
                                     width=0.5, 
                                     edge_color='gray')
                
                plt.title(f"Network Backbone\n({backbone_viz.number_of_nodes()} nodes, {backbone_viz.number_of_edges()} edges)", 
                         fontsize=16, pad=20)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(IMAGES_PATH / "network_backbone.png", dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved: {IMAGES_PATH / 'network_backbone.png'}")
            except Exception as e:
                print(f"Warning: Could not visualize backbone: {e}")
        else:
            print(f"Skipping visualization (network too large: {backbone_viz.number_of_nodes()} nodes)")
    
    return backbone


def calculate_modularity(graph, groups):
    """Calculate modularity: M = Σ[Lc/L - (kc/2L)²]
    
    Args:
        graph: NetworkX graph
        groups: Dictionary mapping node -> group/community
    
    Returns:
        Modularity value
    """
    total_edges = graph.number_of_edges()
    if total_edges == 0:
        return 0.0
    
    # Group nodes by their community
    communities = defaultdict(set)
    for node, group in groups.items():
        if node in graph:
            communities[group].add(node)
    
    # Calculate modularity
    modularity = 0.0
    for group_name, group_nodes in communities.items():
        # Count edges within this group
        edges_in_group = 0
        for node1 in group_nodes:
            for node2 in group_nodes:
                if node1 < node2 and graph.has_edge(node1, node2):
                    edges_in_group += 1
        
        # Sum of degrees in this group
        total_degree = sum(graph.degree(node) for node in group_nodes if node in graph)
        
        # Add to modularity
        modularity += (edges_in_group / total_edges) - (total_degree / (2 * total_edges))**2
    
    return modularity


def analyze_communities(network):
    """Analyze communities using Louvain algorithm and compare with faction affiliations."""
    print("\n" + "="*60)
    print("COMMUNITY DETECTION AND MODULARITY ANALYSIS")
    print("="*60)
    
    # Detect communities using Louvain algorithm
    print("\nDetecting communities using Louvain algorithm...")
    detected_communities = community.louvain_communities(network, seed=42)
    
    # Create mapping from node to community
    node_to_community = {}
    for i, comm in enumerate(detected_communities):
        for node in comm:
            node_to_community[node] = i
    
    # Calculate modularity for detected communities
    modularity_communities = calculate_modularity(network, node_to_community)
    
    print(f"\nNumber of detected communities: {len(detected_communities)}")
    print(f"Modularity for detected communities: {modularity_communities:.4f}")
    
    # Show community sizes
    community_sizes = sorted([len(comm) for comm in detected_communities], reverse=True)
    print(f"\nTop 10 largest communities:")
    for i, size in enumerate(community_sizes[:10], 1):
        print(f"  Community {i}: {size:3d} nodes")
    
    # Compare with faction affiliations (if available)
    # Extract portals from nodes (portals attribute is stored as string)
    node_to_faction = {}
    for node in network.nodes():
        portal = network.nodes[node].get('portals', '')
        if portal:
            # Portals are stored as string (already mapped to main portal)
            portal_str = str(portal).strip()
            # Verify it's one of the 16 main portals
            if portal_str in MAIN_PORTALS:
                node_to_faction[node] = portal_str
    
    if node_to_faction:
        modularity_factions = calculate_modularity(network, node_to_faction)
        print(f"\nModularity for faction partition: {modularity_factions:.4f}")
        print(f"Improvement (communities vs factions): {modularity_communities - modularity_factions:.4f}")
        
        # Use all 16 main portals (sorted alphabetically for consistency)
        all_factions = sorted(list(MAIN_PORTALS))
        
        # Use top 16 communities (or all if fewer than 16)
        num_communities = min(16, len(detected_communities))
        top_communities = sorted(range(len(detected_communities)), 
                                key=lambda i: len(detected_communities[i]), 
                                reverse=True)[:num_communities]
        
        # Create 16x16 (or 16xN) confusion matrix
        confusion_matrix = np.zeros((len(all_factions), len(top_communities)))
        
        for node in network.nodes():
            faction = node_to_faction.get(node)
            comm = node_to_community.get(node)
            
            if faction and faction in all_factions and comm in top_communities:
                faction_idx = all_factions.index(faction)
                comm_idx = top_communities.index(comm)
                confusion_matrix[faction_idx, comm_idx] += 1
        
        # Visualize confusion matrix
        try:
            import seaborn as sns
            # Adjust figure size for 16x16 matrix
            fig_size = (max(14, len(top_communities) * 0.8), max(12, len(all_factions) * 0.6))
            plt.figure(figsize=fig_size)
            sns.heatmap(confusion_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                       xticklabels=[f'Comm {i+1}' for i in range(len(top_communities))],
                       yticklabels=all_factions,
                       cbar_kws={'label': 'Number of characters'},
                       linewidths=0.5)
            plt.xlabel('Detected Communities (Top 16)', fontsize=11)
            plt.ylabel('Factions (16 Main Portals)', fontsize=11)
            plt.title('Confusion Matrix: 16 Main Factions vs Top 16 Communities', fontsize=13, pad=15)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(IMAGES_PATH / "faction_community_confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {IMAGES_PATH / 'faction_community_confusion_matrix.png'}")
            print(f"  Matrix size: {len(all_factions)} factions × {len(top_communities)} communities")
        except ImportError:
            print("Warning: seaborn not available, skipping confusion matrix visualization")
    
    # Visualize communities (if network is not too large)
    if network.number_of_nodes() <= 1000:
        try:
            # Use largest component for visualization
            components = list(nx.connected_components(network))
            if components:
                largest_component = max(components, key=len)
                network_viz = network.subgraph(largest_component).copy()
                
                pos = nx.spring_layout(network_viz, k=1, iterations=50, seed=42)
                
                # Color nodes by community
                colors = plt.cm.tab20(np.linspace(0, 1, len(detected_communities)))
                node_colors = []
                for node in network_viz.nodes():
                    comm = node_to_community.get(node, 0)
                    node_colors.append(colors[comm % len(colors)])
                
                plt.figure(figsize=(14, 12))
                nx.draw_networkx_nodes(network_viz, pos, 
                                     node_color=node_colors,
                                     node_size=30,
                                     alpha=0.8,
                                     linewidths=0.5)
                
                nx.draw_networkx_edges(network_viz, pos,
                                     alpha=0.1,
                                     width=0.3,
                                     edge_color='gray')
                
                plt.title(f"Community Detection\n({len(detected_communities)} communities, modularity={modularity_communities:.4f})",
                         fontsize=16, pad=20)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(IMAGES_PATH / "community_detection.png", dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved: {IMAGES_PATH / 'community_detection.png'}")
        except Exception as e:
            print(f"Warning: Could not visualize communities: {e}")
    else:
        print(f"Skipping community visualization (network too large: {network.number_of_nodes()} nodes)")
    
    return detected_communities, modularity_communities


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
    print(f"\nSaving images to: {IMAGES_PATH.resolve()}")
    
    # Run analyses
    analyze_degree_distribution(G_undirected)
    analyze_centrality(G_undirected, G if isinstance(G, nx.DiGraph) else G_undirected)
    analyze_assortativity(G_undirected)
    
    # Extract backbone
    backbone = extract_backbone(G_undirected, threshold=0.05)
    
    # Community detection
    communities, modularity = analyze_communities(G_undirected)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nAll images saved to: {IMAGES_PATH.resolve()}")


if __name__ == "__main__":
    main()
