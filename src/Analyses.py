#!/usr/bin/env python3
# network analysis script - does stuff on the warhammer network
# calculates degree dist, centrality stuff, correlations, assortativity, backbone, communities

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
    # loads network from pickle file
    # use_filtered = True means use the filtered one (undirected, no isolated nodes)
    # False means use original
    
    if use_filtered:
        network_file = PICKLE_FILTERED_FILE
    else:
        network_file = PICKLE_FILE
    
    if not network_file.exists():
        raise FileNotFoundError(f"Network file not found: {network_file}")
    
    with open(network_file, 'rb') as f:
        G = pickle.load(f)
    
    return G


def analyze_degree_distribution(network):
    # looks at how many connections each node has and makes plots
    print("="*60)
    print("DEGREE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # get all the degrees
    degrees = []
    for node in network.nodes():
        degrees.append(network.degree(node))
    
    # print some stats
    print("\nDegree Statistics:")
    print(f"  Average: {np.mean(degrees):.2f}")
    print(f"  Median: {np.median(degrees):.2f}")
    print(f"  Maximum: {max(degrees)}")
    print(f"  Minimum: {min(degrees)}")
    print(f"  Nodes with degree 0: {sum(1 for d in degrees if d == 0)}")
    
    # make two plots side by side
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # regular histogram
    axes[0].hist(degrees, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Degree')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Degree Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # log log plot
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
    # calculates different centrality measures and sees how they correlate
    print("\n" + "="*60)
    print("CENTRALITY ANALYSIS")
    print("="*60)
    
    # need directed network for eigenvector sometimes
    if network_directed is None:
        network_directed = network
        if not isinstance(network_directed, nx.DiGraph):
            network_directed = network.to_directed()
    
    # calculate all the centralities
    print("\nCalculating centralities...")
    centrality_degree = nx.degree_centrality(network)
    centrality_betweenness = nx.betweenness_centrality(network)
    
    # eigenvector can fail sometimes so try catch
    try:
        centrality_eigenvector = nx.eigenvector_centrality(network, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        print("Warning: Eigenvector centrality failed to converge, using undirected version")
        if isinstance(network, nx.DiGraph):
            network_undir = network.to_undirected()
        else:
            network_undir = network
        centrality_eigenvector = nx.eigenvector_centrality(network_undir, max_iter=1000)
    
    # show top 5 for each
    print("\nTop 5 by Degree Centrality:")
    sorted_degree = sorted(centrality_degree.items(), key=lambda x: x[1], reverse=True)
    for i in range(min(5, len(sorted_degree))):
        node, value = sorted_degree[i]
        node_name = network.nodes[node].get('name', node)
        print(f"  {node_name}: {value:.4f}")
    
    print("\nTop 5 by Betweenness Centrality:")
    sorted_between = sorted(centrality_betweenness.items(), key=lambda x: x[1], reverse=True)
    for i in range(min(5, len(sorted_between))):
        node, value = sorted_between[i]
        node_name = network.nodes[node].get('name', node)
        print(f"  {node_name}: {value:.4f}")
    
    print("\nTop 5 by Eigenvector Centrality:")
    sorted_eigen = sorted(centrality_eigenvector.items(), key=lambda x: x[1], reverse=True)
    for i in range(min(5, len(sorted_eigen))):
        node, value = sorted_eigen[i]
        node_name = network.nodes[node].get('name', node)
        print(f"  {node_name}: {value:.4f}")
    
    # get values in same order for correlation
    nodes = list(network.nodes())
    degree_values = [centrality_degree[n] for n in nodes]
    betweenness_values = [centrality_betweenness[n] for n in nodes]
    eigenvector_values = [centrality_eigenvector[n] for n in nodes]
    
    # calculate correlations
    corr_deg_bet, _ = pearsonr(degree_values, betweenness_values)
    corr_deg_eig, _ = pearsonr(degree_values, eigenvector_values)
    corr_bet_eig, _ = pearsonr(betweenness_values, eigenvector_values)
    
    print("\n=== Correlations Between Centralities ===")
    print(f"Degree vs Betweenness: r = {corr_deg_bet:.3f}")
    print(f"Degree vs Eigenvector: r = {corr_deg_eig:.3f}")
    print(f"Betweenness vs Eigenvector: r = {corr_bet_eig:.3f}")
    
    # make scatter plots
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
    # checks if high degree nodes connect to other high degree nodes
    print("\n" + "="*60)
    print("DEGREE ASSORTATIVITY ANALYSIS")
    print("="*60)
    
    # calculate the coefficient
    assortativity = nx.degree_assortativity_coefficient(network)
    print(f"\nDegree assortativity coefficient: {assortativity:.3f}")
    
    if assortativity > 0:
        print("The network is assortative: high-degree nodes connect to other high-degree nodes.")
    elif assortativity < 0:
        print("The network is disassortative: high-degree nodes connect to low-degree nodes.")
    else:
        print("The network shows no degree assortativity.")
    
    # get degrees and neighbor degrees
    node_degrees = []
    for n in network.nodes():
        node_degrees.append(network.degree(n))
    
    avg_neighbor_degrees = nx.average_neighbor_degree(network)
    neighbor_degrees = [avg_neighbor_degrees[n] for n in network.nodes()]
    
    # plot
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
    plt.plot(sorted_degrees, trendline(sorted_degrees), "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig(IMAGES_PATH / "assortativity.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {IMAGES_PATH / 'assortativity.png'}")


def extract_backbone(network, threshold=0.05):
    # extracts the important edges, removes weak ones
    # threshold is minimum relative weight to keep (default 0.05)
    print("\n" + "="*60)
    print("NETWORK BACKBONE EXTRACTION")
    print("="*60)
    
    # calculate edge betweenness
    print("Calculating edge betweenness centrality...")
    edge_betweenness = nx.edge_betweenness_centrality(network)
    
    # normalize to weights between 1 and 100
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
    
    # make weighted network
    network_weighted = network.copy()
    nx.set_edge_attributes(network_weighted, edge_weights, 'weight')
    
    # find edges to remove
    edges_to_remove = []
    for node in network_weighted.nodes():
        node_edges = list(network_weighted.edges(node))
        
        if len(node_edges) > 1:
            weights = []
            for edge in node_edges:
                weights.append(network_weighted[edge[0]][edge[1]]['weight'])
            total_weight = sum(weights)
            
            for i, edge in enumerate(node_edges):
                weight = weights[i]
                if total_weight > 0:
                    relative_weight = weight / total_weight
                else:
                    relative_weight = 0
                if relative_weight < threshold:
                    edges_to_remove.append(edge)
    
    # make backbone
    backbone = network_weighted.copy()
    backbone.remove_edges_from(edges_to_remove)
    isolated = list(nx.isolates(backbone))
    backbone.remove_nodes_from(isolated)
    
    print(f"\nOriginal network: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
    print(f"Backbone network: {backbone.number_of_nodes()} nodes, {backbone.number_of_edges()} edges")
    edges_removed_pct = (1 - backbone.number_of_edges()/network.number_of_edges())*100
    print(f"Edges removed: {edges_removed_pct:.1f}%")
    
    # visualize if not too big
    if backbone.number_of_nodes() > 0:
        components = list(nx.connected_components(backbone))
        largest_component = max(components, key=len)
        backbone_viz = backbone.subgraph(largest_component).copy()
        print(f"Largest component: {backbone_viz.number_of_nodes()} nodes, {backbone_viz.number_of_edges()} edges")
        
        if backbone_viz.number_of_nodes() <= 500:
            try:
                pos = nx.spring_layout(backbone_viz, k=1, iterations=50, seed=42)
                
                # calculate node sizes from degree
                degrees = []
                for n in backbone_viz.nodes():
                    degrees.append(backbone_viz.degree(n))
                
                if degrees:
                    min_degree = min(degrees)
                    max_degree = max(degrees)
                else:
                    min_degree = 1
                    max_degree = 1
                
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
                
                title = f"Network Backbone\n({backbone_viz.number_of_nodes()} nodes, {backbone_viz.number_of_edges()} edges)"
                plt.title(title, fontsize=16, pad=20)
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
    # calculates modularity score
    # groups is dict mapping node -> group/community
    total_edges = graph.number_of_edges()
    if total_edges == 0:
        return 0.0
    
    # group nodes by community
    communities = defaultdict(set)
    for node, group in groups.items():
        if node in graph:
            communities[group].add(node)
    
    # calculate modularity
    modularity = 0.0
    for group_name, group_nodes in communities.items():
        # count edges within this group
        edges_in_group = 0
        for node1 in group_nodes:
            for node2 in group_nodes:
                if node1 < node2 and graph.has_edge(node1, node2):
                    edges_in_group += 1
        
        # sum of degrees in this group
        total_degree = 0
        for node in group_nodes:
            if node in graph:
                total_degree += graph.degree(node)
        
        # add to modularity
        part1 = edges_in_group / total_edges
        part2 = (total_degree / (2 * total_edges))**2
        modularity += part1 - part2
    
    return modularity


def analyze_communities(network):
    # finds communities using louvain and compares with factions
    print("\n" + "="*60)
    print("COMMUNITY DETECTION AND MODULARITY ANALYSIS")
    print("="*60)
    
    # detect communities
    print("\nDetecting communities using Louvain algorithm...")
    detected_communities = community.louvain_communities(network, seed=42)
    
    # map nodes to communities
    node_to_community = {}
    for i, comm in enumerate(detected_communities):
        for node in comm:
            node_to_community[node] = i
    
    # calculate modularity
    modularity_communities = calculate_modularity(network, node_to_community)
    
    print(f"\nNumber of detected communities: {len(detected_communities)}")
    print(f"Modularity for detected communities: {modularity_communities:.4f}")
    
    # show sizes
    community_sizes = []
    for comm in detected_communities:
        community_sizes.append(len(comm))
    community_sizes = sorted(community_sizes, reverse=True)
    
    print(f"\nTop 10 largest communities:")
    for i in range(min(10, len(community_sizes))):
        print(f"  Community {i+1}: {community_sizes[i]:3d} nodes")
    
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
        print(f"\nModularity for faction partition: {modularity_factions:.4f}")
        improvement = modularity_communities - modularity_factions
        print(f"Improvement (communities vs factions): {improvement:.4f}")
        
        # use all 16 main portals
        all_factions = sorted(list(MAIN_PORTALS))
        
        # use top 16 communities
        num_communities = min(16, len(detected_communities))
        community_indices = list(range(len(detected_communities)))
        sorted_indices = sorted(community_indices, 
                                key=lambda i: len(detected_communities[i]), 
                                reverse=True)
        top_communities = sorted_indices[:num_communities]
        
        # make confusion matrix
        confusion_matrix = np.zeros((len(all_factions), len(top_communities)))
        
        for node in network.nodes():
            faction = node_to_faction.get(node)
            comm = node_to_community.get(node)
            
            if faction and faction in all_factions and comm in top_communities:
                faction_idx = all_factions.index(faction)
                comm_idx = top_communities.index(comm)
                confusion_matrix[faction_idx, comm_idx] += 1
        
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
            print(f"Saved: {IMAGES_PATH / 'faction_community_confusion_matrix.png'}")
            print(f"  Matrix size: {len(all_factions)} factions × {len(top_communities)} communities")
        except ImportError:
            print("Warning: seaborn not available, skipping confusion matrix visualization")
    
    # visualize if not too big
    if network.number_of_nodes() <= 1000:
        try:
            components = list(nx.connected_components(network))
            if components:
                largest_component = max(components, key=len)
                network_viz = network.subgraph(largest_component).copy()
                
                pos = nx.spring_layout(network_viz, k=1, iterations=50, seed=42)
                
                # color by community
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
                
                title = f"Community Detection\n({len(detected_communities)} communities, modularity={modularity_communities:.4f})"
                plt.title(title, fontsize=16, pad=20)
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
    # main function that runs everything
    print("Loading filtered network (undirected, no isolated nodes)...")
    G = load_network(use_filtered=True)
    
    # convert to undirected if needed
    if isinstance(G, nx.DiGraph):
        print("Converting directed graph to undirected for analysis...")
        G_undirected = G.to_undirected()
    else:
        G_undirected = G
    
    print(f"\nNetwork loaded:")
    print(f"  Nodes: {G_undirected.number_of_nodes()}")
    print(f"  Edges: {G_undirected.number_of_edges()}")
    if isinstance(G, nx.DiGraph):
        print(f"  Type: Directed")
    else:
        print(f"  Type: Undirected")
    print(f"\nSaving images to: {IMAGES_PATH.resolve()}")
    
    # run all analyses
    analyze_degree_distribution(G_undirected)
    
    if isinstance(G, nx.DiGraph):
        analyze_centrality(G_undirected, G)
    else:
        analyze_centrality(G_undirected, G_undirected)
    
    analyze_assortativity(G_undirected)
    
    # extract backbone
    backbone = extract_backbone(G_undirected, threshold=0.05)
    
    # community detection
    communities, modularity = analyze_communities(G_undirected)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nAll images saved to: {IMAGES_PATH.resolve()}")


if __name__ == "__main__":
    main()
