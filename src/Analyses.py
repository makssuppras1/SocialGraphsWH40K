#!/usr/bin/env python3
# network analysis script - calculates degree dist, centrality, correlations, assortativity, backbone, communities

import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import pearsonr
from collections import defaultdict
from networkx.algorithms import community

from config import PICKLE_FILE, PICKLE_FILTERED_FILE, IMAGES_PATH
from helpers import MAIN_PORTALS


def load_network(use_filtered=True):
    # loads network from pickle file
    network_file = PICKLE_FILTERED_FILE if use_filtered else PICKLE_FILE
    
    if not network_file.exists():
        raise FileNotFoundError(f"Network file not found: {network_file}")
    
    with open(network_file, 'rb') as f:
        return pickle.load(f)

def analyze_degree_distribution(network):
    # looks at how many connections each node has and makes plots
    degrees = [network.degree(node) for node in network.nodes()]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].hist(degrees, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Degree')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Degree Distribution')
    axes[0].grid(True, alpha=0.3)
    
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    axes[1].loglog(unique_degrees, counts, 'o', markersize=6)
    axes[1].set_xlabel('Degree (log scale)')
    axes[1].set_ylabel('Frequency (log scale)')
    axes[1].set_title('Degree Distribution (Log-Log)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(IMAGES_PATH / "degree_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()


def analyze_centrality(network, network_directed=None):
    # calculates different centrality measures and sees how they correlate
    if network_directed is None:
        network_directed = network
        if not isinstance(network_directed, nx.DiGraph):
            network_directed = network.to_directed()
    
    centrality_degree = nx.degree_centrality(network)
    centrality_betweenness = nx.betweenness_centrality(network)
    
    try:
        centrality_eigenvector = nx.eigenvector_centrality(network, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        network_undir = network.to_undirected() if isinstance(network, nx.DiGraph) else network
        centrality_eigenvector = nx.eigenvector_centrality(network_undir, max_iter=1000)
    
    nodes = list(network.nodes())
    degree_values = [centrality_degree[n] for n in nodes]
    betweenness_values = [centrality_betweenness[n] for n in nodes]
    eigenvector_values = [centrality_eigenvector[n] for n in nodes]
    
    corr_deg_bet, _ = pearsonr(degree_values, betweenness_values)
    corr_deg_eig, _ = pearsonr(degree_values, eigenvector_values)
    corr_bet_eig, _ = pearsonr(betweenness_values, eigenvector_values)
    
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
    # checks if high degree nodes connect to other high degree nodes
    assortativity = nx.degree_assortativity_coefficient(network)
    
    node_degrees = [network.degree(n) for n in network.nodes()]
    avg_neighbor_degrees = nx.average_neighbor_degree(network)
    neighbor_degrees = [avg_neighbor_degrees[n] for n in network.nodes()]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(node_degrees, neighbor_degrees, alpha=0.5, s=30)
    plt.xlabel('Node Degree')
    plt.ylabel('Average Neighbor Degree')
    plt.title(f'Degree Assortativity (r = {assortativity:.3f})')
    plt.grid(True, alpha=0.3)
    
    coefficients = np.polyfit(node_degrees, neighbor_degrees, 1)
    trendline = np.poly1d(coefficients)
    plt.plot(sorted(node_degrees), trendline(sorted(node_degrees)), "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig(IMAGES_PATH / "assortativity.png", dpi=300, bbox_inches='tight')
    plt.close()


def extract_backbone(network, threshold=0.05):
    # extracts the important edges, removes weak ones
    edge_betweenness = nx.edge_betweenness_centrality(network)
    
    if not edge_betweenness:
        return network.copy()
    
    min_eb = min(edge_betweenness.values())
    max_eb = max(edge_betweenness.values())
    
    if max_eb == min_eb:
        return network.copy()
    
    edge_weights = {}
    for edge, eb in edge_betweenness.items():
        edge_weights[edge] = 1 + 99 * (eb - min_eb) / (max_eb - min_eb)
    
    network_weighted = network.copy()
    nx.set_edge_attributes(network_weighted, edge_weights, 'weight')
    
    edges_to_remove = []
    for node in network_weighted.nodes():
        node_edges = list(network_weighted.edges(node))
        if len(node_edges) > 1:
            weights = [network_weighted[edge[0]][edge[1]]['weight'] for edge in node_edges]
            total_weight = sum(weights)
            
            for i, edge in enumerate(node_edges):
                relative_weight = weights[i] / total_weight if total_weight > 0 else 0
                if relative_weight < threshold:
                    edges_to_remove.append(edge)
    
    backbone = network_weighted.copy()
    backbone.remove_edges_from(edges_to_remove)
    backbone.remove_nodes_from(list(nx.isolates(backbone)))
    
    # visualize if not too big
    if backbone.number_of_nodes() > 0:
        components = list(nx.connected_components(backbone))
        largest_component = max(components, key=len)
        backbone_viz = backbone.subgraph(largest_component).copy()
        
        if backbone_viz.number_of_nodes() <= 500:
            try:
                pos = nx.spring_layout(backbone_viz, k=1, iterations=50, seed=42)
                degrees = [backbone_viz.degree(n) for n in backbone_viz.nodes()]
                
                if degrees:
                    min_degree, max_degree = min(degrees), max(degrees)
                    node_sizes = [20 + 180 * (d - min_degree) / (max_degree - min_degree) if max_degree > min_degree else 20 for d in degrees]
                else:
                    node_sizes = [20] * len(backbone_viz.nodes())
                
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
            except Exception:
                pass
    
    return backbone


def calculate_modularity(graph, groups):
    # calculates modularity score
    total_edges = graph.number_of_edges()
    if total_edges == 0:
        return 0.0
    
    communities = defaultdict(set)
    for node, group in groups.items():
        if node in graph:
            communities[group].add(node)
    
    modularity = 0.0
    for group_name, group_nodes in communities.items():
        edges_in_group = sum(1 for node1 in group_nodes for node2 in group_nodes 
                            if node1 < node2 and graph.has_edge(node1, node2))
        total_degree = sum(graph.degree(node) for node in group_nodes if node in graph)
        modularity += (edges_in_group / total_edges) - (total_degree / (2 * total_edges))**2
    
    return modularity


def analyze_communities(network):
    # finds communities using louvain and compares with factions
    detected_communities = community.louvain_communities(network, seed=42)
    
    node_to_community = {}
    for i, comm in enumerate(detected_communities):
        for node in comm:
            node_to_community[node] = i
    
    modularity_communities = calculate_modularity(network, node_to_community)
    
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
        
        all_factions = sorted(list(MAIN_PORTALS))
        num_communities = min(16, len(detected_communities))
        sorted_indices = sorted(range(len(detected_communities)), 
                                key=lambda i: len(detected_communities[i]), 
                                reverse=True)
        top_communities = sorted_indices[:num_communities]
        
        confusion_matrix = np.zeros((len(all_factions), len(top_communities)))
        for node in network.nodes():
            faction = node_to_faction.get(node)
            comm = node_to_community.get(node)
            if faction and faction in all_factions and comm in top_communities:
                confusion_matrix[all_factions.index(faction), top_communities.index(comm)] += 1
        
        # visualize
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
                colors = plt.cm.tab20(np.linspace(0, 1, len(detected_communities)))
                node_colors = [colors[node_to_community.get(node, 0) % len(colors)] for node in network_viz.nodes()]
                
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
    # main function that runs everything
    G = load_network(use_filtered=True)
    
    if isinstance(G, nx.DiGraph):
        G_undirected = G.to_undirected()
    else:
        G_undirected = G
    
    analyze_degree_distribution(G_undirected)
    
    if isinstance(G, nx.DiGraph):
        analyze_centrality(G_undirected, G)
    else:
        analyze_centrality(G_undirected, G_undirected)
    
    analyze_assortativity(G_undirected)
    extract_backbone(G_undirected, threshold=0.05)
    analyze_communities(G_undirected)
    
    print("Analysis complete.")


if __name__ == "__main__":
    main()
