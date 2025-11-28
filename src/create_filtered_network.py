#!/usr/bin/env python3
# creates filtered undirected network: largest component, nodes with degree >= 2

import pickle
import networkx as nx

from config import PICKLE_FILE, GEXF_FILTERED_FILE, PICKLE_FILTERED_FILE


def load_network():
    if not PICKLE_FILE.exists():
        raise FileNotFoundError(f"Network file not found: {PICKLE_FILE}")
    
    with open(PICKLE_FILE, 'rb') as f:
        return pickle.load(f)


def filter_network(G):
    # convert to undirected
    G_undirected = G.to_undirected()
    
    # keep only largest connected component
    components = list(nx.connected_components(G_undirected))
    if not components:
        raise ValueError("No connected components found")
    
    largest_component = max(components, key=len)
    G_filtered = G_undirected.subgraph(largest_component).copy()
    
    # remove nodes with degree < 2 iteratively
    while True:
        nodes_to_remove = [n for n in G_filtered.nodes() if G_filtered.degree(n) < 2]
        if not nodes_to_remove:
            break
        G_filtered.remove_nodes_from(nodes_to_remove)
    
    return G_filtered


def prepare_graph_for_export(G):
    # prepares graph for export by converting lists to strings
    G_export = G.copy()
    
    for node in G_export.nodes():
        node_data = G_export.nodes[node]
        if 'all_affiliations' in node_data and isinstance(node_data['all_affiliations'], list):
            node_data['all_affiliations'] = ', '.join(str(x) for x in node_data['all_affiliations'] if x is not None)
        elif 'all_affiliations' in node_data and node_data['all_affiliations'] is None:
            node_data['all_affiliations'] = ''
        
        if 'portal_list' in node_data:
            del node_data['portal_list']
        
        for key, value in list(node_data.items()):
            if value is None:
                node_data[key] = ''
            elif isinstance(value, (list, tuple, dict)):
                node_data[key] = str(value)
    
    for u, v, edge_data in G_export.edges(data=True):
        for key, value in list(edge_data.items()):
            if value is None:
                edge_data[key] = ''
            elif isinstance(value, (list, tuple, dict)):
                edge_data[key] = str(value)
    
    return G_export


def save_filtered_network(G_filtered):
    G_export = prepare_graph_for_export(G_filtered)
    
    try:
        if GEXF_FILTERED_FILE.exists():
            GEXF_FILTERED_FILE.unlink()
        nx.write_gexf(G_export, GEXF_FILTERED_FILE)
    except Exception as e:
        print(f"Warning: Could not save GEXF format: {e}")
    
    with open(PICKLE_FILTERED_FILE, 'wb') as f:
        pickle.dump(G_filtered, f)
    
    print(f"Filtered network created: {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")


def main():
    G = load_network()
    G_filtered = filter_network(G)
    save_filtered_network(G_filtered)


if __name__ == "__main__":
    main()
