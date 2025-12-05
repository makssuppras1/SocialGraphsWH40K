# Semantic Clustering Explanation

## Overview

This document explains how we group Warhammer 40K characters into semantic clusters based on their descriptions. Instead of grouping by network connections, we group characters by the **meaning and themes** in their text descriptions.

## What is Semantic Clustering?

Semantic clustering groups items based on **meaning** rather than exact word matches. For example:
- Characters mentioning "knights", "grey", "daemon", "brotherhood" might be grouped together
- Characters mentioning "eldar", "craftworld", "ynnari" might form another group
- The algorithm understands that these words are related thematically

## The Process: Step by Step

### Step 1: Load and Clean Character Descriptions

**What happens:**
- Load character descriptions from Wikipedia-style text
- Remove character names from descriptions (so we focus on themes, not name mentions)
- Clean up wiki markup (remove brackets, links, etc.)

**Why remove names?**
- If we keep names, characters might cluster just because they mention each other
- Removing names helps us find thematic groups (e.g., "Space Marines", "Eldar", "Chaos")

**Result:** Clean text descriptions for 3,232 characters

### Step 2: Generate Embeddings

**What happens:**
- Convert text descriptions into numerical vectors (embeddings)
- Each description becomes a 384-dimensional vector
- Similar meanings → similar vectors

**How it works:**
- Uses a pre-trained model (`all-MiniLM-L6-v2`) that understands language
- The model was trained on millions of text examples
- It captures semantic relationships (e.g., "knight" and "warrior" are close in meaning)

**Result:** 3,232 embeddings (each is a 384-number vector)

**Time:** ~13 seconds

### Step 3: Find Optimal Number of Clusters

**What happens:**
- Test different numbers of clusters (k) from 2 to 40
- For each k, run K-Means clustering and evaluate quality
- Choose the best k using a parsimony-adjusted method

**Why 2-40?**
- 40 is the number of Louvain communities found in the network
- This gives us a reasonable upper bound
- We test all values to find the optimal one

**How we evaluate quality:**
- **Silhouette Score**: Measures how well-separated clusters are
  - Range: -1 to 1 (higher is better)
  - Considers both cohesion (within-cluster similarity) and separation (between-cluster distance)
- **Average Within-Cluster Distance**: How close items are within each cluster
  - Lower is better (tighter clusters)

**Result:** Quality scores for each k value (2 through 40)

**Time:** ~17 seconds (parallelized across 6 CPU cores)

### Step 4: Select Optimal k Using Parsimony Principle

**The Problem:**
- Higher k often gives better silhouette scores (more clusters = better fit)
- But we want **fewer clusters** when quality is similar (simpler model)

**Our Solution:**
1. Find the maximum silhouette score across all k values
2. Set threshold = 95% of maximum
3. Select the **smallest k** where silhouette score ≥ threshold

**Why this works:**
- If k=26 has 95% of the best score, it's "good enough"
- We prefer k=26 over k=36 if quality is similar (parsimony principle)
- Similar to the "elbow method" - find where diminishing returns start

**Result:** Optimal k = 26 clusters

### Step 5: Extract Keywords for Each Cluster

**What happens:**
- For each cluster, find the most distinctive words
- Uses TF-IDF (Term Frequency-Inverse Document Frequency)
- Words that appear often in one cluster but rarely elsewhere are most distinctive

**Result:** Top 5 keywords per cluster (e.g., "knights, grey, daemon, brotherhood, primarch")

## Results Summary

### Optimal Configuration
- **Number of clusters (k):** 26
- **Silhouette score:** 0.0405 (95.5% of maximum 0.0425)
- **Average within-cluster distance:** 1.0145
- **Total characters:** 3,232

### Cluster Sizes
Clusters range from 39 to 237 characters:
- **Smallest cluster (20):** 39 characters (Necromunda-related)
- **Largest cluster (13):** 237 characters (Khorne/World Eaters)
- **Average cluster size:** ~124 characters

### Example Clusters

**Cluster 0 (59 characters):** Grey Knights
- Keywords: knights, grey, daemon, brotherhood, primarch
- Examples: Kaldor Draigo, Garran Crowe, Alaric

**Cluster 6 (127 characters):** Eldar
- Keywords: eldar, craftworld, ynnari, iyanden, commorragh
- Examples: Illic Nightspear, Iyanna Arienal

**Cluster 9 (120 characters):** Chaos Daemons
- Keywords: daemon, tzeentch, khorne, daemons, slaanesh
- Examples: Ka'Bandha, Ingethel

**Cluster 25 (99 characters):** Sisters of Battle
- Keywords: sisters, sororitas, canoness, adepta, sister
- Examples: Imelda Veritas, Jenetia Krole

## Output Files

### 1. `data/optimal_semantic_clusters.json`
Contains the final clustering assignment for each character:
```json
{
  "optimal_k": 26,
  "total_characters": 3232,
  "silhouette_score": 0.0405,
  "assignments": {
    "Character Name": {
      "semantic_cluster_id": 0,
      "top_keywords": "knights, grey, daemon, brotherhood, primarch"
    }
  }
}
```

### 2. `data/clustering_evaluation_results.json`
Contains evaluation metrics for all tested k values (2-40):
```json
{
  "k_range": {"min": 2, "max": 40},
  "optimal_k": 26,
  "evaluations": [
    {
      "k": 2,
      "silhouette_score": 0.0234,
      "avg_within_cluster_distance": 1.1234,
      "is_optimal": false
    }
  ]
}
```

### 3. `data/optimal_cluster_summaries.json`
Contains summaries for each of the 26 clusters:
```json
{
  "optimal_k": 26,
  "clusters": {
    "0": {
      "size": 59,
      "keywords": "knights, grey, daemon, brotherhood, primarch",
      "characters": ["Character1", "Character2", ...]
    }
  }
}
```

### 4. `images/cohesion_analysis.png`
Visualization showing:
- Silhouette scores for all k values (2-40)
- Average within-cluster distances
- Highlighted optimal k=26

### 5. `data/character_descriptions_processed.json`
Cached processed descriptions (names removed) for faster re-runs

## How to Interpret the Results

### Silhouette Score (0.0405)
- **What it means:** Moderate clustering quality
- **Why it's low:** Character descriptions are diverse and overlapping
- **Is it good?** Yes - for this dataset, 0.04 is reasonable. Higher scores (0.5+) are rare in real-world text clustering.

### Average Within-Cluster Distance (1.0145)
- **What it means:** Characters in the same cluster are relatively similar
- **Lower is better:** Tighter clusters = more cohesive themes
- **Context:** In embedding space, distances typically range from 0-2, so 1.01 is reasonable

### Why k=26?
- **Quality:** 95.5% of maximum silhouette score (very close to best)
- **Parsimony:** 10 fewer clusters than k=36 (simpler model)
- **Balance:** Good quality with reasonable complexity

## Technical Details

### Models and Libraries
- **Embeddings:** SentenceTransformer (`all-MiniLM-L6-v2`)
- **Clustering:** scikit-learn K-Means
- **Evaluation:** scikit-learn Silhouette Score
- **Keywords:** scikit-learn TF-IDF

### Performance
- **Total time:** ~30 seconds
- **Embeddings:** ~13 seconds
- **Clustering:** ~17 seconds (parallelized)
- **Parallelization:** 6 cores for outer loop, 5 cores per k for inner operations

### Why This Method?
1. **Semantic understanding:** Captures meaning, not just keywords
2. **Automated selection:** No manual tuning needed
3. **Balanced approach:** Quality + parsimony
4. **Reproducible:** Same input → same output

## Academic References

See `CLUSTERING_METHOD_REFERENCE.md` for detailed references:
- Rousseeuw (1987): Original silhouette coefficient paper
- Elbow Method: Wikipedia reference
- Parsimony Principle: Occam's Razor in clustering

## Usage

To run the clustering analysis:
```bash
uv run python src/find_optimal_clusters.py
```

The script will:
1. Load and process character descriptions
2. Generate embeddings
3. Test k values from 2-40
4. Select optimal k=26
5. Extract keywords
6. Save all results to JSON files

