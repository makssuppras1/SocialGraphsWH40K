#!/usr/bin/env python3
# explores the mythology analysis results

import pandas as pd
import numpy as np
from pathlib import Path

def load_results():
    csv_path = Path("data/mythology_vs_narrative_analysis.csv")
    if not csv_path.exists():
        csv_path = Path(__file__).parent.parent / "data/mythology_vs_narrative_analysis.csv"
    return pd.read_csv(csv_path)

def identify_mythological_anchors(df, betweenness_threshold=0.01, consistency_threshold=0.50):
    # identifies potential mythological anchors: high betweenness + low neighbor consistency
    candidates = df[
        (df['betweenness'] >= betweenness_threshold) & 
        (df['neighbor_consistency'] <= consistency_threshold)
    ].copy()
    
    if len(candidates) == 0:
        # relax thresholds
        betweenness_threshold = df['betweenness'].quantile(0.90)
        consistency_threshold = df['neighbor_consistency'].quantile(0.20)
        candidates = df[
            (df['betweenness'] >= betweenness_threshold) & 
            (df['neighbor_consistency'] <= consistency_threshold)
        ].copy()
    
    if len(candidates) > 0:
        candidates = candidates.sort_values('betweenness', ascending=False)
        print(f"Found {len(candidates)} mythological anchor candidates:")
        for idx, row in candidates.iterrows():
            print(f"  {row['node_id']:40s} | Betweenness: {row['betweenness']:8.6f} | "
                  f"Consistency: {row['neighbor_consistency']:.4f}")
    
    return candidates

def display_all_clusters(df):
    # displays all semantic clusters with their keywords
    clusters = df.groupby('semantic_cluster_id').agg({
        'node_id': 'count',
        'top_keywords': 'first',
        'betweenness': 'mean',
        'neighbor_consistency': 'mean'
    }).round(4)
    
    clusters.columns = ['count', 'keywords', 'avg_betweenness', 'avg_consistency']
    clusters = clusters.sort_values('count', ascending=False)
    
    print("\nSemantic Clusters:")
    for cluster_id, row in clusters.iterrows():
        print(f"  Cluster {int(cluster_id)}: {int(row['count'])} chars, "
              f"avg betweenness: {row['avg_betweenness']:.6f}, "
              f"avg consistency: {row['avg_consistency']:.4f}")
        print(f"    Keywords: {row['keywords']}")

def compare_characters(df, char_names):
    found_chars = df[df['node_id'].isin(char_names)].copy()
    if len(found_chars) == 0:
        print("No matching characters found.")
        return
    
    print("\nCharacter Comparison:")
    for idx, char in found_chars.iterrows():
        keywords = str(char['top_keywords'])[:50]
        print(f"  {char['node_id']:40s} | Betweenness: {char['betweenness']:8.6f} | "
              f"Consistency: {char['neighbor_consistency']:.4f} | Cluster: {int(char['semantic_cluster_id'])}")

def show_statistics(df):
    print(f"\nTotal characters: {len(df)}")
    print(f"Betweenness - Mean: {df['betweenness'].mean():.6f}, Max: {df['betweenness'].max():.6f}")
    print(f"Consistency - Mean: {df['neighbor_consistency'].mean():.4f}, "
          f"Min: {df['neighbor_consistency'].min():.4f}, Max: {df['neighbor_consistency'].max():.4f}")
    print(f"Semantic clusters: {df['semantic_cluster_id'].nunique()}")

def main():
    df = load_results()
    show_statistics(df)
    identify_mythological_anchors(df)
    display_all_clusters(df)
    
    interesting_chars = [
        "Roboute Guilliman", "Emperor of Mankind", "Ibram Gaunt",
        "Rogal Dorn", "Sanguinius", "Lion El'Jonson"
    ]
    compare_characters(df, interesting_chars)

if __name__ == "__main__":
    main()
