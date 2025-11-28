#!/usr/bin/env python3
# explores the mythology analysis results:
# - identifies mythological anchor candidates
# - displays all semantic clusters
# - shows interesting comparisons

import pandas as pd
import numpy as np
from pathlib import Path

def load_results():
    # loads the analysis results
    csv_path = Path("data/mythology_vs_narrative_analysis.csv")
    if not csv_path.exists():
        csv_path = Path(__file__).parent.parent / "data/mythology_vs_narrative_analysis.csv"
    
    df = pd.read_csv(csv_path)
    return df

def identify_mythological_anchors(df, betweenness_threshold=0.01, consistency_threshold=0.50):
    # identifies potential mythological anchors:
    # high betweenness + low neighbor consistency
    print("="*80)
    print("MYTHOLOGICAL ANCHOR CANDIDATES")
    print("="*80)
    print(f"\nCriteria:")
    print(f"  - Betweenness >= {betweenness_threshold}")
    print(f"  - Neighbor Consistency <= {consistency_threshold}")
    print(f"  (High structural importance + Low semantic similarity to neighbors)")
    
    # filter candidates
    candidates = df[
        (df['betweenness'] >= betweenness_threshold) & 
        (df['neighbor_consistency'] <= consistency_threshold)
    ].copy()
    
    if len(candidates) == 0:
        print("\nNo characters meet the strict criteria. Relaxing thresholds...")
        # use top 10% by betweenness, bottom 20% by consistency
        betweenness_threshold = df['betweenness'].quantile(0.90)
        consistency_threshold = df['neighbor_consistency'].quantile(0.20)
        candidates = df[
            (df['betweenness'] >= betweenness_threshold) & 
            (df['neighbor_consistency'] <= consistency_threshold)
        ].copy()
        print(f"  Using: Betweenness >= {betweenness_threshold:.6f} (top 10%)")
        print(f"         Consistency <= {consistency_threshold:.4f} (bottom 20%)")
    
    if len(candidates) > 0:
        candidates = candidates.sort_values('betweenness', ascending=False)
        print(f"\nFound {len(candidates)} candidates:\n")
        
        for idx, row in candidates.iterrows():
            print(f"{row['node_id']:40s} | Betweenness: {row['betweenness']:8.6f} | "
                  f"Consistency: {row['neighbor_consistency']:.4f} | "
                  f"Cluster: {int(row['semantic_cluster_id'])}")
    else:
        print("\nNo candidates found even with relaxed criteria.")
    
    return candidates

def display_all_clusters(df):
    # displays all semantic clusters with their keywords and example characters
    print("\n" + "="*80)
    print("ALL SEMANTIC CLUSTERS (20 Archetypes)")
    print("="*80)
    
    # group by cluster
    clusters = df.groupby('semantic_cluster_id').agg({
        'node_id': 'count',
        'top_keywords': 'first',
        'betweenness': ['mean', 'max'],
        'neighbor_consistency': 'mean'
    }).round(4)
    
    clusters.columns = ['count', 'keywords', 'avg_betweenness', 'max_betweenness', 'avg_consistency']
    clusters = clusters.sort_values('count', ascending=False)
    
    print(f"\n{'Cluster':<8} {'Count':<8} {'Avg Betweenness':<18} {'Avg Consistency':<18} {'Keywords'}")
    print("-"*80)
    
    for cluster_id, row in clusters.iterrows():
        print(f"{int(cluster_id):<8} {int(row['count']):<8} {row['avg_betweenness']:<18.6f} "
              f"{row['avg_consistency']:<18.4f} {row['keywords']}")
    
    # show example characters from each cluster
    print("\n" + "="*80)
    print("EXAMPLE CHARACTERS FROM EACH CLUSTER")
    print("="*80)
    
    for cluster_id in sorted(df['semantic_cluster_id'].dropna().unique()):
        cluster_chars = df[df['semantic_cluster_id'] == cluster_id].copy()
        keywords = cluster_chars['top_keywords'].iloc[0]
        
        print(f"\n{'='*80}")
        print(f"CLUSTER {int(cluster_id)}: {keywords}")
        print(f"{'='*80}")
        print(f"Total characters: {len(cluster_chars)}")
        print(f"Average consistency: {cluster_chars['neighbor_consistency'].mean():.4f}")
        print(f"Average betweenness: {cluster_chars['betweenness'].mean():.6f}")
        
        # show top characters by betweenness
        top_by_betweenness = cluster_chars.nlargest(5, 'betweenness')
        print(f"\nTop 5 by Betweenness:")
        for idx, char in top_by_betweenness.iterrows():
            print(f"  {char['node_id']:40s} | Betweenness: {char['betweenness']:8.6f} | "
                  f"Consistency: {char['neighbor_consistency']:.4f}")
        
        # show characters with lowest consistency (most diverse neighbors)
        low_consistency = cluster_chars.nsmallest(3, 'neighbor_consistency')
        if len(low_consistency) > 0:
            print(f"\nLowest Consistency (Most Diverse Neighbors):")
            for idx, char in low_consistency.iterrows():
                print(f"  {char['node_id']:40s} | Consistency: {char['neighbor_consistency']:.4f}")

def compare_characters(df, char_names):
    # compares specific characters side by side
    print("\n" + "="*80)
    print("CHARACTER COMPARISON")
    print("="*80)
    
    found_chars = df[df['node_id'].isin(char_names)].copy()
    
    if len(found_chars) == 0:
        print("No matching characters found.")
        return
    
    print(f"\n{'Character':<40} {'Betweenness':<15} {'Consistency':<15} {'Cluster':<10} {'Keywords'}")
    print("-"*120)
    
    for idx, char in found_chars.iterrows():
        keywords = str(char['top_keywords'])[:50]  # truncate long keywords
        print(f"{char['node_id']:<40} {char['betweenness']:<15.6f} "
              f"{char['neighbor_consistency']:<15.4f} {int(char['semantic_cluster_id']):<10} {keywords}")

def show_statistics(df):
    # shows overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    print(f"\nTotal characters analyzed: {len(df)}")
    print(f"\nBetweenness Centrality:")
    print(f"  Mean: {df['betweenness'].mean():.6f}")
    print(f"  Median: {df['betweenness'].median():.6f}")
    max_betweenness_idx = df['betweenness'].idxmax()
    print(f"  Max: {df['betweenness'].max():.6f} ({df.loc[max_betweenness_idx, 'node_id']})")
    
    print(f"\nNeighbor Semantic Consistency:")
    print(f"  Mean: {df['neighbor_consistency'].mean():.4f}")
    print(f"  Median: {df['neighbor_consistency'].median():.4f}")
    min_consistency_idx = df['neighbor_consistency'].idxmin()
    max_consistency_idx = df['neighbor_consistency'].idxmax()
    print(f"  Min: {df['neighbor_consistency'].min():.4f} ({df.loc[min_consistency_idx, 'node_id']})")
    print(f"  Max: {df['neighbor_consistency'].max():.4f} ({df.loc[max_consistency_idx, 'node_id']})")
    
    print(f"\nSemantic Clusters:")
    print(f"  Number of clusters: {df['semantic_cluster_id'].nunique()}")
    print(f"  Largest cluster: {df['semantic_cluster_id'].value_counts().max()} characters")
    print(f"  Smallest cluster: {df['semantic_cluster_id'].value_counts().min()} characters")

def main():
    # main function
    print("="*80)
    print("MYTHOLOGY VS NARRATIVE ANALYSIS - RESULTS EXPLORATION")
    print("="*80)
    
    # load data
    df = load_results()
    
    # show statistics
    show_statistics(df)
    
    # identify mythological anchors
    candidates = identify_mythological_anchors(df)
    
    # display all clusters
    display_all_clusters(df)
    
    # compare some interesting characters
    print("\n" + "="*80)
    print("INTERESTING CHARACTER COMPARISONS")
    print("="*80)
    
    interesting_chars = [
        "Roboute Guilliman",
        "Emperor of Mankind",
        "Ibram Gaunt",
        "Rogal Dorn",
        "Sanguinius",
        "Lion El'Jonson"
    ]
    
    compare_characters(df, interesting_chars)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
