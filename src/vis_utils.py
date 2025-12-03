import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

def plot_mythology_scatter(df, output_path='images/mythology_scatter.png'):
    # scatter plot: x-axis = betweenness centrality (log scale), y-axis = neighbor semantic consistency
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Set style for better aesthetics
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9
    
    fig, ax = plt.subplots(figsize=(14, 9))
    plot_df = df.dropna(subset=['betweenness', 'neighbor_consistency', 'semantic_cluster_id']).copy()
    

    # Create scatter plot with edge colors for better visibility
    scatter = ax.scatter(
        plot_df['betweenness'] + 1e-8,  # Add small value to avoid log(0)
        plot_df['neighbor_consistency'],
        c=plot_df['semantic_cluster_id'],
        cmap='tab20',
        alpha=0.7,
        s=60,
        edgecolors='white',
        linewidth=0.5,
        zorder=3
    )
        
    # Set log scale for x-axis
    ax.set_xscale('log')
    
    # Add trend line (fit on log-transformed x values)
    x_data = plot_df['betweenness'] + 1e-8
    y_data = plot_df['neighbor_consistency']
    log_x = np.log10(x_data)
    
    # Fit linear regression on log(x) vs y
    coeffs = np.polyfit(log_x, y_data, 1)
    poly = np.poly1d(coeffs)
    
    # Generate x values for trend line (in original scale, then log-transform for fitting)
    x_trend = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 100)
    log_x_trend = np.log10(x_trend)
    y_trend = poly(log_x_trend)
    
    # Plot trend line
    ax.plot(x_trend, y_trend, 'r--', linewidth=2, alpha=0.8, label='Trend line', zorder=2)
    
    # Better labels with units/descriptions
    ax.set_xlabel('Betweenness Centrality (log scale)', 
                    fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Neighbor Semantic Consistency', 
                    fontsize=13, fontweight='bold', labelpad=10)
    
    # Enhanced title
    ax.set_title('Mythology vs Narrative: Structural Centrality vs Semantic Coherence\n' +
                '(Colored by Semantic Cluster)', 
                fontsize=15, fontweight='bold', pad=20)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add colorbar instead of legend for better space usage
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Semantic Cluster ID', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    
    # Add reference lines for key regions
    # Vertical line for high betweenness (mythological anchors)
    x_median = plot_df['betweenness'].median()
    ax.axvline(x=x_median, color='gray', linestyle=':', alpha=0.5, linewidth=1, zorder=1)
    
    # Horizontal line for consistency threshold
    y_median = plot_df['neighbor_consistency'].median()
    ax.axhline(y=y_median, color='gray', linestyle=':', alpha=0.5, linewidth=1, zorder=1)
    
    # Add annotations for key regions
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # # Top-right: High betweenness + High consistency (important within coherent narrative)
    # ax.text(x_max * 0.7, y_max * 0.9, 
    #         'High Centrality\n+ High Consistency\n(Coherent Narrative)', 
    #         fontsize=10, ha='center', va='top',
    #         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.6, edgecolor='darkgreen', linewidth=1.5),
    #         zorder=4)
    
    # # Top-left: Low betweenness + High consistency (narrative agents)
    # ax.text(x_min * 1.0, y_max * 0.8, 
    #         'Low Centrality\n+ High Consistency\n(Narrative Agents)', 
    #         fontsize=10, ha='center', va='top',
    #         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.6, edgecolor='darkblue', linewidth=1.5),
    #         zorder=4)
    
    # # Bottom-right: High betweenness + Low consistency (mythological anchors)
    # ax.text(x_max * 0.8, y_min * 1.25, 
    #         'High Centrality\n+ Low Consistency\n(Mythological Anchors)', 
    #         fontsize=10, ha='center', va='bottom',
    #         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7, edgecolor='orange', linewidth=1.5),
    #         zorder=4)
    
    # Improve layout
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()

# def export_results(df, output_path='data/mythology_vs_narrative_analysis.csv'):
#     # saves a csv file with specific columns
#     Path(output_path).parent.mkdir(parents=True, exist_ok=True)

#     columns = [
#         'node_id', 'betweenness', 'pagerank', 'network_community_id', 
#         'semantic_cluster_id', 'top_keywords', 'neighbor_consistency'
#     ]
#     export_cols = [c for c in columns if c in df.columns]
#     df[export_cols].to_csv(output_path, index=False)
