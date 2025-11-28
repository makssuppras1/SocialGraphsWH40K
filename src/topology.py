import json
import networkx as nx
import community.community_louvain as community_louvain
import pandas as pd
from pathlib import Path

def load_graph(edges_file='data/lexicanum_edges.json'):
    """Load the graph from the edges JSON file."""
    path = Path(edges_file)
    if not path.exists():
        # Try relative to project root if not found
        path = Path(__file__).parent.parent / edges_file
        
    with open(path, 'r', encoding='utf-8') as f:
        edges_data = json.load(f)
    
    G = nx.DiGraph()
    for edge in edges_data:
        G.add_edge(edge['source'], edge['target'])
        
    return G

def compute_topology_metrics(G):
    """
    Compute topological metrics: Betweenness, PageRank, Clustering, Communities.
    Returns a DataFrame with these metrics.
    """
    # Betweenness Centrality
    # For 12k nodes, this might take a few minutes.
    print("Calculating Betweenness Centrality...")
    # Using k=None for exact calculation as per requirements, but could be approximated with k=1000
    betweenness = nx.betweenness_centrality(G)
    
    print("Calculating PageRank...")
    pagerank = nx.pagerank(G)
    
    print("Calculating Clustering Coefficient...")
    clustering = nx.clustering(G)
    
    # Community Detection (Louvain)
    # Louvain requires undirected graph.
    print("Calculating Louvain Communities...")
    G_undirected = G.to_undirected()
    partition = community_louvain.best_partition(G_undirected)
    
    # Combine into DataFrame
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
        G = load_graph()
        print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        metrics_df = compute_topology_metrics(G)
        print(metrics_df.head())
        metrics_df.to_csv("data/topology_metrics.csv", index=False)
        print("Topology metrics saved to data/topology_metrics.csv")
    except Exception as e:
        print(f"Error: {e}")

