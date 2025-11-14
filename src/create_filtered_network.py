#!/usr/bin/env python3
"""
Create filtered undirected network by removing isolated nodes (degree 0).

This script:
1. Loads the original network from pickle file
2. Converts it to an undirected graph
3. Removes all nodes with degree 0 (isolated nodes)
4. Saves the filtered network in multiple formats
"""

import pickle
import networkx as nx
from pathlib import Path

from config import DATA_PATH, PICKLE_FILE, GEXF_FILTERED_FILE, PICKLE_FILTERED_FILE


def load_network():
    """Load the network from pickle file."""
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
    """Convert directed graph to undirected graph."""
    print("\n" + "="*60)
    print("Converting to Undirected Graph")
    print("="*60)
    
    # Convert to undirected (combines parallel edges)
    G_undirected = G.to_undirected()
    
    print(f"Converted to undirected graph:")
    print(f"  Nodes: {G_undirected.number_of_nodes()}")
    print(f"  Edges: {G_undirected.number_of_edges()}")
    
    return G_undirected


def remove_isolated_nodes(G):
    """Remove all nodes with degree 0."""
    print("\n" + "="*60)
    print("Removing Isolated Nodes (degree 0)")
    print("="*60)
    
    # Find nodes with degree 0
    isolated_nodes = [n for n in G.nodes() if G.degree(n) == 0]
    
    print(f"Found {len(isolated_nodes)} isolated nodes (degree 0)")
    
    if isolated_nodes:
        print(f"\nSample isolated nodes (first 10):")
        for node in isolated_nodes[:10]:
            node_name = G.nodes[node].get('name', node)
            print(f"  {node_name}")
        if len(isolated_nodes) > 10:
            print(f"  ... and {len(isolated_nodes) - 10} more")
    
    # Create a copy and remove isolated nodes
    G_filtered = G.copy()
    G_filtered.remove_nodes_from(isolated_nodes)
    
    print(f"\nAfter filtering:")
    print(f"  Original nodes: {G.number_of_nodes()}")
    print(f"  Removed nodes: {len(isolated_nodes)}")
    print(f"  Remaining nodes: {G_filtered.number_of_nodes()}")
    print(f"  Edges: {G_filtered.number_of_edges()}")
    
    return G_filtered, isolated_nodes


def save_filtered_network(G_filtered):
    """Save the filtered network in multiple formats."""
    print("\n" + "="*60)
    print("Saving Filtered Network")
    print("="*60)
    
    # Prepare graph for export (convert lists to strings)
    G_export = prepare_graph_for_export(G_filtered)
    
    # Save GEXF
    try:
        if GEXF_FILTERED_FILE.exists():
            GEXF_FILTERED_FILE.unlink()
        nx.write_gexf(G_export, GEXF_FILTERED_FILE)
        if GEXF_FILTERED_FILE.exists() and GEXF_FILTERED_FILE.stat().st_size > 0:
            print(f"Saved filtered graph to {GEXF_FILTERED_FILE} ({GEXF_FILTERED_FILE.stat().st_size:,} bytes)")
        else:
            print(f"Warning: GEXF file was created but appears empty")
    except Exception as e:
        print(f"Warning: Could not save GEXF format: {e}")
        print(f"  Error type: {type(e).__name__}")
    
    # Save Pickle (preserves all data types)
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
    """Prepare graph for export by converting lists to strings."""
    G_export = G.copy()
    
    for node in G_export.nodes():
        node_data = G_export.nodes[node]
        
        # Convert all_affiliations list to string
        if 'all_affiliations' in node_data:
            if isinstance(node_data['all_affiliations'], list):
                node_data['all_affiliations'] = ', '.join(
                    str(x) for x in node_data['all_affiliations'] if x is not None
                )
            elif node_data['all_affiliations'] is None:
                node_data['all_affiliations'] = ''
        
        # Remove portal_list (keep only portals string)
        if 'portal_list' in node_data:
            del node_data['portal_list']
        
        # Convert None and non-serializable types
        for key, value in list(node_data.items()):
            if value is None:
                node_data[key] = ''
            elif isinstance(value, (list, tuple, dict)):
                node_data[key] = str(value)
    
    # Convert edge attributes
    for u, v, edge_data in G_export.edges(data=True):
        for key, value in list(edge_data.items()):
            if value is None:
                edge_data[key] = ''
            elif isinstance(value, (list, tuple, dict)):
                edge_data[key] = str(value)
    
    return G_export


def print_network_statistics(G, G_filtered):
    """Print comparison statistics."""
    print("\n" + "="*60)
    print("NETWORK STATISTICS COMPARISON")
    print("="*60)
    
    print(f"\nOriginal Network (Directed):")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.6f}")
    
    print(f"\nFiltered Network (Undirected, no isolated nodes):")
    print(f"  Nodes: {G_filtered.number_of_nodes()}")
    print(f"  Edges: {G_filtered.number_of_edges()}")
    print(f"  Density: {nx.density(G_filtered):.6f}")
    
    if G_filtered.number_of_nodes() > 0:
        # Connected components
        components = list(nx.connected_components(G_filtered))
        print(f"\nConnected Components:")
        print(f"  Number of components: {len(components)}")
        if components:
            largest_component = max(components, key=len)
            print(f"  Largest component size: {len(largest_component)} nodes "
                  f"({100*len(largest_component)/G_filtered.number_of_nodes():.2f}%)")
        
        # Degree statistics
        degrees = [d for n, d in G_filtered.degree()]
        import numpy as np
        print(f"\nDegree Statistics (filtered network):")
        print(f"  Average degree: {np.mean(degrees):.2f}")
        print(f"  Median degree: {np.median(degrees):.2f}")
        print(f"  Max degree: {max(degrees) if degrees else 0}")
        print(f"  Min degree: {min(degrees) if degrees else 0}")


def main():
    """Main execution function."""
    # Step 1: Load network
    G = load_network()
    
    # Step 2: Convert to undirected
    G_undirected = convert_to_undirected(G)
    
    # Step 3: Remove isolated nodes
    G_filtered, isolated_nodes = remove_isolated_nodes(G_undirected)
    
    # Step 4: Print statistics
    print_network_statistics(G, G_filtered)
    
    # Step 5: Save filtered network
    save_filtered_network(G_filtered)


if __name__ == "__main__":
    main()

