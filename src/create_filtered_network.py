#!/usr/bin/env python3
# creates filtered undirected network by keeping only the largest connected component
# this script:
# 1. loads the original network from pickle file
# 2. converts it to an undirected graph
# 3. keeps only the largest connected component (removes all nodes not connected to main network)
# 4. removes nodes with degree < 2 (iteratively until no more can be removed)
# 5. saves the filtered network in multiple formats

import pickle
import networkx as nx
from pathlib import Path

from config import DATA_PATH, PICKLE_FILE, GEXF_FILTERED_FILE, PICKLE_FILTERED_FILE


def load_network():
    # loads the network from pickle file
    print("="*60)
    print("Loading Network")
    print("="*60)
    
    if not PICKLE_FILE.exists():
        raise FileNotFoundError(f"Network file not found: {PICKLE_FILE}")
    
    with open(PICKLE_FILE, 'rb') as f:
        G = pickle.load(f)
    
    print(f"Loaded network:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Type: {type(G).__name__}")
    
    return G


def convert_to_undirected(G):
    # converts directed graph to undirected graph
    print("\n" + "="*60)
    print("Converting to Undirected Graph")
    print("="*60)
    
    # convert to undirected (combines parallel edges)
    G_undirected = G.to_undirected()
    
    print(f"Converted to undirected graph:")
    print(f"  Nodes: {G_undirected.number_of_nodes()}")
    print(f"  Edges: {G_undirected.number_of_edges()}")
    
    return G_undirected


def keep_largest_component(G):
    # keeps only the largest connected component, removing all other nodes
    print("\n" + "="*60)
    print("Keeping Only Largest Connected Component")
    print("="*60)
    
    # find all connected components
    components = list(nx.connected_components(G))
    print(f"Found {len(components)} connected components")
    
    if not components:
        print("Warning: No connected components found!")
        return G.copy(), []
    
    # find the largest component
    largest_component = max(components, key=len)
    print(f"Largest component size: {len(largest_component)} nodes")
    
    # show component size distribution
    component_sizes = sorted([len(c) for c in components], reverse=True)
    print(f"\nComponent size distribution:")
    print(f"  Largest: {component_sizes[0]} nodes")
    if len(component_sizes) > 1:
        print(f"  Second largest: {component_sizes[1]} nodes")
    if len(component_sizes) > 2:
        print(f"  Total components: {len(component_sizes)}")
        print(f"  Other components: {component_sizes[2:]} nodes each")
    
    # find nodes to remove (all nodes not in largest component)
    all_nodes = set(G.nodes())
    nodes_to_remove = all_nodes - largest_component
    
    print(f"\nRemoving {len(nodes_to_remove)} nodes not in largest component")
    
    if nodes_to_remove:
        print(f"\nSample removed nodes (first 10):")
        removed_list = list(nodes_to_remove)[:10]
        for node in removed_list:
            node_name = G.nodes[node].get('name', node)
            print(f"  {node_name}")
        if len(nodes_to_remove) > 10:
            print(f"  ... and {len(nodes_to_remove) - 10} more")
    
    # create a subgraph with only the largest component
    G_filtered = G.subgraph(largest_component).copy()
    
    print(f"\nAfter filtering:")
    print(f"  Original nodes: {G.number_of_nodes()}")
    print(f"  Removed nodes: {len(nodes_to_remove)}")
    print(f"  Remaining nodes: {G_filtered.number_of_nodes()}")
    print(f"  Edges: {G_filtered.number_of_edges()}")
    
    return G_filtered, list(nodes_to_remove)


def remove_low_degree_nodes(G, min_degree=2):
    # removes nodes with degree < min_degree iteratively until no more can be removed
    print("\n" + "="*60)
    print(f"Removing Nodes with Degree < {min_degree}")
    print("="*60)
    
    G_filtered = G.copy()
    total_removed = 0
    iteration = 0
    
    while True:
        iteration += 1
        # find nodes with degree < min_degree
        nodes_to_remove = []
        for n in G_filtered.nodes():
            if G_filtered.degree(n) < min_degree:
                nodes_to_remove.append(n)
        
        if not nodes_to_remove:
            break
        
        if iteration == 1:
            print(f"Iteration {iteration}: Found {len(nodes_to_remove)} nodes with degree < {min_degree}")
            if nodes_to_remove:
                print(f"\nSample nodes to remove (first 10):")
                for node in nodes_to_remove[:10]:
                    node_name = G_filtered.nodes[node].get('name', node)
                    degree = G_filtered.degree(node)
                    print(f"  {node_name} (degree: {degree})")
                if len(nodes_to_remove) > 10:
                    print(f"  ... and {len(nodes_to_remove) - 10} more")
        
        # remove the nodes
        G_filtered.remove_nodes_from(nodes_to_remove)
        total_removed += len(nodes_to_remove)
        
        if iteration > 1:
            print(f"Iteration {iteration}: Removed {len(nodes_to_remove)} nodes (total removed: {total_removed})")
    
    print(f"\nRemoved {total_removed} nodes with degree < {min_degree} in {iteration} iteration(s)")
    print(f"\nAfter filtering:")
    print(f"  Original nodes: {G.number_of_nodes()}")
    print(f"  Removed nodes: {total_removed}")
    print(f"  Remaining nodes: {G_filtered.number_of_nodes()}")
    print(f"  Edges: {G_filtered.number_of_edges()}")
    
    return G_filtered, total_removed


def save_filtered_network(G_filtered):
    # saves the filtered network in multiple formats
    print("\n" + "="*60)
    print("Saving Filtered Network")
    print("="*60)
    
    # prepare graph for export (convert lists to strings)
    G_export = prepare_graph_for_export(G_filtered)
    
    # save gexf
    try:
        if GEXF_FILTERED_FILE.exists():
            GEXF_FILTERED_FILE.unlink()
        nx.write_gexf(G_export, GEXF_FILTERED_FILE)
        if GEXF_FILTERED_FILE.exists() and GEXF_FILTERED_FILE.stat().st_size > 0:
            size = GEXF_FILTERED_FILE.stat().st_size
            print(f"Saved filtered graph to {GEXF_FILTERED_FILE} ({size:,} bytes)")
        else:
            print(f"Warning: GEXF file was created but appears empty")
    except Exception as e:
        print(f"Warning: Could not save GEXF format: {e}")
        print(f"  Error type: {type(e).__name__}")
    
    # save pickle (preserves all data types)
    with open(PICKLE_FILTERED_FILE, 'wb') as f:
        pickle.dump(G_filtered, f)
    print(f"Saved filtered graph to {PICKLE_FILTERED_FILE} (preserves all data types including lists)")
    
    print("\n" + "="*60)
    print("Filtered network creation complete!")
    print("="*60)
    print(f"\nFiltered network ready for analysis!")
    print(f"  - Nodes: {G_filtered.number_of_nodes()}")
    print(f"  - Edges: {G_filtered.number_of_edges()}")
    print(f"\nFiles saved to: {DATA_PATH.resolve()}")


def prepare_graph_for_export(G):
    # prepares graph for export by converting lists to strings
    G_export = G.copy()
    
    for node in G_export.nodes():
        node_data = G_export.nodes[node]
        
        # convert all_affiliations list to string
        if 'all_affiliations' in node_data:
            if isinstance(node_data['all_affiliations'], list):
                affil_strs = []
                for x in node_data['all_affiliations']:
                    if x is not None:
                        affil_strs.append(str(x))
                node_data['all_affiliations'] = ', '.join(affil_strs)
            elif node_data['all_affiliations'] is None:
                node_data['all_affiliations'] = ''
        
        # remove portal_list (keep only portals string)
        if 'portal_list' in node_data:
            del node_data['portal_list']
        
        # convert None and non-serializable types
        for key, value in list(node_data.items()):
            if value is None:
                node_data[key] = ''
            elif isinstance(value, (list, tuple, dict)):
                node_data[key] = str(value)
    
    # convert edge attributes
    for u, v, edge_data in G_export.edges(data=True):
        for key, value in list(edge_data.items()):
            if value is None:
                edge_data[key] = ''
            elif isinstance(value, (list, tuple, dict)):
                edge_data[key] = str(value)
    
    return G_export


def print_network_statistics(G, G_filtered):
    # prints comparison statistics
    print("\n" + "="*60)
    print("NETWORK STATISTICS COMPARISON")
    print("="*60)
    
    print(f"\nOriginal Network (Directed):")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.6f}")
    
    print(f"\nFiltered Network (Undirected, largest component, degree >= 2):")
    print(f"  Nodes: {G_filtered.number_of_nodes()}")
    print(f"  Edges: {G_filtered.number_of_edges()}")
    print(f"  Density: {nx.density(G_filtered):.6f}")
    
    if G_filtered.number_of_nodes() > 0:
        # connected components (should be 1 after filtering)
        components = list(nx.connected_components(G_filtered))
        print(f"\nConnected Components:")
        print(f"  Number of components: {len(components)}")
        if len(components) == 1:
            print(f"  ✓ Network is fully connected (single component)")
        else:
            print(f"  Warning: Multiple components still present")
            if components:
                largest_component = max(components, key=len)
                pct = 100*len(largest_component)/G_filtered.number_of_nodes()
                print(f"  Largest component size: {len(largest_component)} nodes ({pct:.2f}%)")
        
        # degree statistics
        degrees = [d for n, d in G_filtered.degree()]
        import numpy as np
        print(f"\nDegree Statistics (filtered network):")
        print(f"  Average degree: {np.mean(degrees):.2f}")
        print(f"  Median degree: {np.median(degrees):.2f}")
        if degrees:
            print(f"  Max degree: {max(degrees)}")
            print(f"  Min degree: {min(degrees)}")
        else:
            print(f"  Max degree: 0")
            print(f"  Min degree: 0")


def main():
    # main execution function
    # step 1: load network
    G = load_network()
    
    # step 2: convert to undirected
    G_undirected = convert_to_undirected(G)
    
    # step 3: keep only largest connected component
    G_filtered, removed_nodes_component = keep_largest_component(G_undirected)
    
    # step 4: remove nodes with degree < 2
    G_filtered, removed_nodes_degree = remove_low_degree_nodes(G_filtered, min_degree=2)
    
    # step 5: print statistics
    print_network_statistics(G, G_filtered)
    
    # step 6: save filtered network
    save_filtered_network(G_filtered)


if __name__ == "__main__":
    main()
