import json
import pickle
import networkx as nx
import community.community_louvain as community_louvain
import pandas as pd
from pathlib import Path

def load_graph(edges_file='data/lexicanum_edges.json', use_filtered=True):
    # loads graph from edges json file, optionally filtered
    # use_filtered = True means only include nodes from filtered network
    
    valid_nodes = None
    if use_filtered:
        try:
            from config import PICKLE_FILTERED_FILE
            path = Path(PICKLE_FILTERED_FILE)
            if not path.exists():
                path = Path(__file__).parent.parent / PICKLE_FILTERED_FILE
            
            if path.exists():
                print(f"Loading filtered network nodes from {path}...")
                with open(path, 'rb') as f:
                    G_filtered = pickle.load(f)
                valid_nodes = set(G_filtered.nodes())
                print(f"Filtered network has {len(valid_nodes)} nodes")
            else:
                print(f"Warning: Filtered network file not found at {path}. Using full network.")
        except Exception as e:
            print(f"Warning: Could not load filtered network: {e}. Using full network.")
    
    # load edges
    path = Path(edges_file)
    if not path.exists():
        path = Path(__file__).parent.parent / edges_file
        
    with open(path, 'r', encoding='utf-8') as f:
        edges_data = json.load(f)
    
    G = nx.DiGraph()
    
    # add edges, filtering if needed
    edges_added = 0
    for edge in edges_data:
        source = edge['source']
        target = edge['target']
        
        # if using filtered, only add edges between valid nodes
        if valid_nodes is not None:
            if source not in valid_nodes or target not in valid_nodes:
                continue
        
        G.add_edge(source, target)
        edges_added += 1
    
    print(f"Built directed graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    if use_filtered and valid_nodes:
        print(f"  (Filtered to match {len(valid_nodes)} nodes from filtered network)")
    
    return G

def compute_topology_metrics(G):
    # computes topological metrics: betweenness, pagerank, clustering, communities
    # returns dataframe with these metrics
    
    # betweenness centrality - might take a while for big networks
    print("Calculating Betweenness Centrality...")
    betweenness = nx.betweenness_centrality(G)
    
    print("Calculating PageRank...")
    pagerank = nx.pagerank(G)
    
    print("Calculating Clustering Coefficient...")
    clustering = nx.clustering(G)
    
    # community detection (louvain) - needs undirected graph
    print("Calculating Louvain Communities...")
    G_undirected = G.to_undirected()
    partition = community_louvain.best_partition(G_undirected)
    
    # combine into dataframe
    nodes = list(G.nodes())
    data = {
        'node_id': nodes,
        'betweenness': [betweenness.get(n, 0) for n in nodes],
        'pagerank': [pagerank.get(n, 0) for n in nodes],
        'clustering_coefficient': [clustering.get(n, 0) for n in nodes],
        'network_community_id': [partition.get(n, 0) for n in nodes]
    }
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    try:
        G = load_graph(use_filtered=True)
        print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        metrics_df = compute_topology_metrics(G)
        print(metrics_df.head())
        metrics_df.to_csv("data/topology_metrics.csv", index=False)
        print("Topology metrics saved to data/topology_metrics.csv")
    except Exception as e:
        print(f"Error: {e}")
