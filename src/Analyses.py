#!/usr/bin/env python3
# network analysis script - calculates degree dist, centrality, correlations, assortativity, backbone, communities

import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import pearsonr
from collections import defaultdict
from networkx.algorithms import community

from config import IMAGES_PATH, load_network
from helpers import MAIN_PORTALS, simple_sort_descending

def analyze_degree_distribution(network):
    degrees = []
    for node in network.nodes():
        degrees.append(network.degree(node))
    
    print(f"Degree distribution analysis:")
    print(f"  Total nodes: {len(degrees)}")
    total_degree = sum(degrees)
    mean_degree = total_degree / len(degrees) if len(degrees) > 0 else 0
    print(f"  Mean degree: {mean_degree:.2f}")
    print(f"  Max degree: {max(degrees)}")
    print(f"  Min degree: {min(degrees)}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Regular histogram
    axes[0].hist(degrees, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Degree')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Degree Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Log-log scale
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    axes[1].loglog(unique_degrees, counts, 'o', markersize=6)
    axes[1].set_xlabel('Degree (log scale)')
    axes[1].set_ylabel('Frequency (log scale)')
    axes[1].set_title('Degree Distribution (Log-Log)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = IMAGES_PATH / "degree_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_centrality(network, network_directed=None):
    # Calculate different centrality measures and their correlations
    print("Centrality analysis:")
    if network_directed is None:
        network_directed = network
        if not isinstance(network_directed, nx.DiGraph):
            network_directed = network.to_directed()
    
    # calculate three types of centrality
    centrality_degree = nx.degree_centrality(network)
    centrality_betweenness = nx.betweenness_centrality(network)
    
    if isinstance(network, nx.DiGraph):
        network_undir = network.to_undirected()
    else:
        network_undir = network
    centrality_eigenvector = nx.eigenvector_centrality(network_undir, max_iter=1000)
    
    # collect all values for each node
    nodes = list(network.nodes())
    degree_values = []
    betweenness_values = []
    eigenvector_values = []
    
    # Build lists for correlation calculation
    for n in nodes:
        deg_val = centrality_degree[n]
        bet_val = centrality_betweenness[n]
        eig_val = centrality_eigenvector[n]
        degree_values.append(deg_val)
        betweenness_values.append(bet_val)
        eigenvector_values.append(eig_val)
    
    # Calculate correlations between measures
    corr_deg_bet, _ = pearsonr(degree_values, betweenness_values)
    corr_deg_eig, _ = pearsonr(degree_values, eigenvector_values)
    corr_bet_eig, _ = pearsonr(betweenness_values, eigenvector_values)
    
    print(f"  Degree-Betweenness correlation: {corr_deg_bet:.3f}")
    print(f"  Degree-Eigenvector correlation: {corr_deg_eig:.3f}")
    print(f"  Betweenness-Eigenvector correlation: {corr_bet_eig:.3f}")
    
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


def analyze_assortativity(network):
    # Check if high degree nodes connect to other high degree nodes
    assortativity = nx.degree_assortativity_coefficient(network)
    print(f"Assortativity: {assortativity:.3f}")
    if assortativity > 0:
        print("  Network is assortative (high-degree nodes connect to high-degree nodes)")
    else:
        print("  Network is disassortative (high-degree nodes connect to low-degree nodes)")
    
    # collect node degrees
    node_degrees = []
    for n in network.nodes():
        deg = network.degree(n)
        node_degrees.append(deg)
    
    # get average neighbor degrees for each node
    avg_neighbor_degrees = nx.average_neighbor_degree(network)
    neighbor_degrees = []
    for n in network.nodes():
        neighbor_deg = avg_neighbor_degrees[n]
        neighbor_degrees.append(neighbor_deg)
    
    # create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(node_degrees, neighbor_degrees, alpha=0.5, s=30)
    plt.xlabel('Node Degree')
    plt.ylabel('Average Neighbor Degree')
    plt.title(f'Degree Assortativity (r = {assortativity:.3f})')
    plt.grid(True, alpha=0.3)
    
    # add trend line
    coefficients = np.polyfit(node_degrees, neighbor_degrees, 1)
    trendline = np.poly1d(coefficients)
    sorted_degrees = sorted(node_degrees)
    trend_y = trendline(sorted_degrees)
    plt.plot(sorted_degrees, trend_y, "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    output_file = IMAGES_PATH / "assortativity.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def extract_backbone(network, threshold=0.05):
    # Extract important edges, remove weak ones
    print(f"Extracting network backbone (threshold={threshold})...")
    edge_betweenness = nx.edge_betweenness_centrality(network)
    
    if not edge_betweenness:
        return network.copy()
    
    min_eb = min(edge_betweenness.values())
    max_eb = max(edge_betweenness.values())
    
    if max_eb == min_eb:
        return network.copy()
    
    edge_weights = {}
    for edge in edge_betweenness.keys():
        eb = edge_betweenness[edge]
        # Normalize edge betweenness to 1-100 range
        weight = 1 + 99 * (eb - min_eb) / (max_eb - min_eb)
        edge_weights[edge] = weight
    
    network_weighted = network.copy()
    nx.set_edge_attributes(network_weighted, edge_weights, 'weight')
    # print(f"Edge weight range: {min(edge_weights.values()):.2f} to {max(edge_weights.values()):.2f}")  # debugging
    
    # find edges to remove based on threshold
    edges_to_remove = []
    for node in network_weighted.nodes():
        node_edges = list(network_weighted.edges(node))
        if len(node_edges) > 1:
            # get weights for all edges from this node
            weights = []
            for edge in node_edges:
                source = edge[0]
                target = edge[1]
                edge_weight = network_weighted[source][target]['weight']
                weights.append(edge_weight)
            
            total_weight = sum(weights)
            
            # check each edge and mark weak ones for removal
            edge_index = 0
            for edge in node_edges:
                if total_weight > 0:
                    relative_weight = weights[edge_index] / total_weight
                else:
                    relative_weight = 0
                
                if relative_weight < threshold:
                    edges_to_remove.append(edge)
                edge_index = edge_index + 1
    
    backbone = network_weighted.copy()
    backbone.remove_edges_from(edges_to_remove)
    backbone.remove_nodes_from(list(nx.isolates(backbone)))
    
    print(f"  Backbone: {backbone.number_of_nodes()} nodes, {backbone.number_of_edges()} edges")
    print(f"  Removed {network.number_of_edges() - backbone.number_of_edges()} edges")
    
    # Visualize if not too big
    if backbone.number_of_nodes() > 0:
        components = list(nx.connected_components(backbone))
        largest_component = max(components, key=len)
        backbone_viz = backbone.subgraph(largest_component).copy()
        
        if backbone_viz.number_of_nodes() <= 500:
            pos = nx.spring_layout(backbone_viz, k=1, iterations=50, seed=42)
            degrees = []
            for n in backbone_viz.nodes():
                degrees.append(backbone_viz.degree(n))
            
            if degrees:
                min_degree = min(degrees)
                max_degree = max(degrees)
                node_sizes = []
                for d in degrees:
                    if max_degree > min_degree:
                        size = 20 + 180 * (d - min_degree) / (max_degree - min_degree)
                    else:
                        size = 20
                    node_sizes.append(size)
            else:
                num_nodes = len(backbone_viz.nodes())
                node_sizes = [20] * num_nodes
            
            plt.figure(figsize=(14, 12))
            nx.draw_networkx_nodes(backbone_viz, pos, node_size=node_sizes, node_color='steelblue', 
                                 alpha=0.8, linewidths=0.5, edgecolors='black')
            nx.draw_networkx_edges(backbone_viz, pos, alpha=0.4, width=0.5, edge_color='gray')
            plt.title(f"Network Backbone\n({backbone_viz.number_of_nodes()} nodes, {backbone_viz.number_of_edges()} edges)", 
                     fontsize=16, pad=20)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(IMAGES_PATH / "network_backbone.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    return backbone


def calculate_modularity(graph, groups):
    # Custom modularity calculation (networkx has one but wanted to understand it better)
    total_edges = graph.number_of_edges()
    if total_edges == 0:
        return 0.0
    
    communities = defaultdict(set)
    for node in groups.keys():
        group = groups[node]
        if node in graph:
            communities[group].add(node)
    
    modularity = 0.0
    for group_name in communities.keys():
        group_nodes = communities[group_name]
        edges_in_group = 0
        for node1 in group_nodes:
            for node2 in group_nodes:
                if node1 < node2 and graph.has_edge(node1, node2):
                    edges_in_group += 1
        
        total_degree = 0
        for node in group_nodes:
            if node in graph:
                total_degree += graph.degree(node)
        
        part1 = edges_in_group / total_edges
        part2 = (total_degree / (2 * total_edges)) ** 2
        modularity += part1 - part2
    
    return modularity


def analyze_communities(network):
    # Find communities using Louvain algorithm and compare with factions
    # Ensure network is undirected for Louvain algorithm
    if isinstance(network, nx.DiGraph):
        network = network.to_undirected()
    elif not isinstance(network, nx.Graph):
        network = network.to_undirected()
    
    print("Community detection:")
    detected_communities = community.louvain_communities(network, seed=42)
    
    node_to_community = {}
    # Assign community IDs - using index as ID
    comm_index = 0
    for comm in detected_communities:
        for node in comm:
            node_to_community[node] = comm_index
        comm_index = comm_index + 1
    
    modularity_communities = calculate_modularity(network, node_to_community)
    print(f"  Found {len(detected_communities)} communities")
    print(f"  Modularity: {modularity_communities:.4f}")
    
    # compare with factions
    node_to_faction = {}
    for node in network.nodes():
        portal = network.nodes[node].get('portals', '')
        if portal:
            portal_str = str(portal).strip()
            if portal_str in MAIN_PORTALS:
                node_to_faction[node] = portal_str
    
    if node_to_faction:
        modularity_factions = calculate_modularity(network, node_to_faction)
        print(f"  Faction modularity: {modularity_factions:.4f}")
        
        # Sort factions for display
        all_factions = list(MAIN_PORTALS)
        all_factions.sort()
        
        num_communities = 16
        if len(detected_communities) < 16:
            num_communities = len(detected_communities)
        
        # sort communities by size (largest first)
        community_sizes = []
        for i in range(len(detected_communities)):
            size = len(detected_communities[i])
            community_sizes.append((i, size))
        
        # Sort by size descending - using helper function
        community_sizes = simple_sort_descending(community_sizes, key_index=1)
        
        # get top communities
        top_communities = []
        for i in range(num_communities):
            top_communities.append(community_sizes[i][0])
        
        confusion_matrix = np.zeros((len(all_factions), len(top_communities)))
        for node in network.nodes():
            faction = node_to_faction.get(node)
            comm = node_to_community.get(node)
            if faction and faction in all_factions and comm in top_communities:
                confusion_matrix[all_factions.index(faction), top_communities.index(comm)] += 1
        
        # Create normalized matrix (row-normalized: percentage of each faction in each community)
        num_factions = len(all_factions)
        num_communities = len(top_communities)
        normalized_matrix = np.zeros((num_factions, num_communities), dtype=float)
        for i in range(num_factions):
            row_sum = confusion_matrix[i, :].sum()
            if row_sum > 0:
                for j in range(num_communities):
                    normalized_matrix[i, j] = (confusion_matrix[i, j] / row_sum) * 100
            else:
                for j in range(num_communities):
                    normalized_matrix[i, j] = 0
        
        # visualize raw count confusion matrix
        try:
            import seaborn as sns
            fig_size_x = max(14, len(top_communities) * 0.8)
            fig_size_y = max(12, len(all_factions) * 0.6)
            plt.figure(figsize=(fig_size_x, fig_size_y))
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
            
            # visualize normalized confusion matrix (row-normalized percentages)
            plt.figure(figsize=(fig_size_x, fig_size_y))
            sns.heatmap(normalized_matrix, annot=True, fmt='.1f', cmap='YlOrRd',
                       xticklabels=[f'Comm {i+1}' for i in range(len(top_communities))],
                       yticklabels=all_factions,
                       cbar_kws={'label': 'Percentage of faction characters'},
                       linewidths=0.5,
                       vmin=0,
                       vmax=100)
            plt.xlabel('Detected Communities (Top 16)', fontsize=11)
            plt.ylabel('Factions (16 Main Portals)', fontsize=11)
            plt.title('Faction vs Community Comparison (Normalized)\n(% of each faction in each community)', fontsize=13, pad=15)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(IMAGES_PATH / "faction_community_confusion_matrix_normalized.png", dpi=300, bbox_inches='tight')
            plt.close()
        except ImportError:
            pass
    
    # visualize if not too big
    if network.number_of_nodes() <= 1000:
        try:
            components = list(nx.connected_components(network))
            if components:
                largest_component = max(components, key=len)
                network_viz = network.subgraph(largest_component).copy()
                pos = nx.spring_layout(network_viz, k=1, iterations=50, seed=42)
                # assign colors to communities
                num_colors = len(detected_communities)
                color_values = np.linspace(0, 1, num_colors)
                colors = plt.cm.tab20(color_values)
                
                # assign color to each node based on its community
                node_colors = []
                for node in network_viz.nodes():
                    comm_id = node_to_community.get(node, 0)
                    color_index = comm_id % len(colors)
                    node_colors.append(colors[color_index])
                
                plt.figure(figsize=(14, 12))
                nx.draw_networkx_nodes(network_viz, pos, node_color=node_colors, node_size=30,
                                     alpha=0.8, linewidths=0.5)
                nx.draw_networkx_edges(network_viz, pos, alpha=0.1, width=0.3, edge_color='gray')
                plt.title(f"Community Detection\n({len(detected_communities)} communities, modularity={modularity_communities:.4f})",
                         fontsize=16, pad=20)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(IMAGES_PATH / "community_detection.png", dpi=300, bbox_inches='tight')
                plt.close()
        except Exception:
            pass
    
    return detected_communities, modularity_communities


def main():
    # Run all network analyses
    print("Loading network...")
    G = load_network(use_filtered=True)
    
    if isinstance(G, nx.DiGraph):
        G_undirected = G.to_undirected()
    else:
        G_undirected = G
    
    print(f"Network: {G_undirected.number_of_nodes()} nodes, {G_undirected.number_of_edges()} edges")
    print()
    
    analyze_degree_distribution(G_undirected)
    print()
    
    if isinstance(G, nx.DiGraph):
        analyze_centrality(G_undirected, G)
    else:
        analyze_centrality(G_undirected, G_undirected)
    print()
    
    analyze_assortativity(G_undirected)
    print()
    
    extract_backbone(G_undirected, threshold=0.05)
    print()
    
    analyze_communities(G_undirected)
    print()
    
    print("Analysis complete.")


if __name__ == "__main__":
    main()
