import json
import pickle
import networkx as nx
#import community.community_louvain as community_louvain
import pandas as pd
from pathlib import Path

def load_graph(edges_file='data/lexicanum_edges.json', use_filtered=True):
    # loads graph from edges json file, optionally filtered
    # NOTE: Always uses filtered network by default. Set use_filtered=False only if you need the unfiltered network.
    valid_nodes = None
    if use_filtered:
        try:
            from config import PICKLE_FILTERED_FILE
            path = Path(PICKLE_FILTERED_FILE)
            if not path.exists():
                path = Path(__file__).parent.parent / PICKLE_FILTERED_FILE
            
            if path.exists():
                with open(path, 'rb') as f:
                    G_filtered = pickle.load(f)
                valid_nodes = set(G_filtered.nodes())
        except Exception:
            pass
    else:
        import warnings
        warnings.warn("Using unfiltered network. All analysis should use the filtered network (use_filtered=True).", UserWarning)
    
    path = Path(edges_file)
    if not path.exists():
        path = Path(__file__).parent.parent / edges_file
        
    with open(path, 'r', encoding='utf-8') as f:
        edges_data = json.load(f)
    
    G = nx.DiGraph()
    for edge in edges_data:
        source, target = edge['source'], edge['target']
        if valid_nodes is None or (source in valid_nodes and target in valid_nodes):
            G.add_edge(source, target)
    
    return G

def compute_topology_metrics(G):

    betweenness = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G)
    clustering = nx.clustering(G)
    G_undirected = G.to_undirected()
    partition = nx.community.louvain_communities(G_undirected)
    
    # Convert partition (list of sets) to node->community_id mapping
    node_to_community = {}
    for comm_id, community in enumerate(partition):
        for node in community:
            node_to_community[node] = comm_id
    
    nodes = list(G.nodes())
    return pd.DataFrame({
        'node_id': nodes,
        'betweenness': [betweenness.get(n, 0) for n in nodes],
        'pagerank': [pagerank.get(n, 0) for n in nodes],
        'clustering_coefficient': [clustering.get(n, 0) for n in nodes],
        'network_community_id': [node_to_community.get(n, -1) for n in nodes]
    })

if __name__ == "__main__":
    try:
        G = load_graph(use_filtered=True)
        metrics_df = compute_topology_metrics(G)
        metrics_df.to_csv("data/topology_metrics.csv", index=False)
    except Exception as e:
        print(f"Error: {e}")
