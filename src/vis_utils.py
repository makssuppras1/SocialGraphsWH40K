import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def plot_mythology_scatter(df, output_path='images/mythology_scatter.png'):
    # scatter plot: x-axis = betweenness centrality (log scale), y-axis = neighbor semantic consistency
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 8))
    plot_df = df.dropna(subset=['betweenness', 'neighbor_consistency', 'semantic_cluster_id']).copy()
    
    if len(plot_df) == 0:
        return

    try:
        sns.scatterplot(
            data=plot_df,
            x='betweenness',
            y='neighbor_consistency',
            hue='semantic_cluster_id',
            palette='tab20',
            alpha=0.6,
            s=20
        )
        plt.xscale('log')
        plt.xlabel('Betweenness Centrality (Log Scale)')
        plt.ylabel('Neighbor Semantic Consistency')
        plt.title('Narrative Independence vs Structural Centrality\n(Highlighting Mythological Anchors)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Semantic Cluster', ncol=2)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error plotting: {e}")

def export_results(df, output_path='data/mythology_vs_narrative_analysis.csv'):
    # saves a csv file with specific columns
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    columns = [
        'node_id', 'betweenness', 'pagerank', 'network_community_id', 
        'semantic_cluster_id', 'top_keywords', 'neighbor_consistency'
    ]
    export_cols = [c for c in columns if c in df.columns]
    df[export_cols].to_csv(output_path, index=False)
