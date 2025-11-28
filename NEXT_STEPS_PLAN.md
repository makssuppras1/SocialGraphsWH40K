# Revised Next Steps Plan: "Mythology vs. Narrative" Analysis Pipeline

## Executive Summary

This plan pivots the project focus to investigate the **"Mythology vs. Narrative" dichotomy** in the Warhammer 40,000 universe. The core research objective is to test the hypothesis that "Mythological Anchors" (setting-defining figures like The Emperor) differ structurally and semantically from "Narrative Agents" (novel protagonists like Ibram Gaunt).

We will implement a hybrid Network+NLP pipeline to calculate a **"Narrative Independence Score"** and identify these archetypes.

## Research Hypothesis

> **Hypothesis**: "Narrative Agents" will have **high** neighbor similarity (they interact with characters within their specific story arc/thematic cluster). "Mythological Anchors" will have **low** neighbor similarity (they connect disparate parts of the lore and distinct thematic clusters).

## Data Sources

- **Graph Data**: `lexicanum_edges.json` (Source -> Target)
- **Text Data**: `lexicanum_characters.json` (Title, Description/Wikitext)
  - *Note*: We will construct a Pandas DataFrame from these files containing `node_id`, `description`, and `links`.

## Development Phases

### **Phase 1: Topological Feature Extraction (NetworkX)**
*Focus: Characterizing structural roles in the graph.*

**Tasks:**
1. **Graph Construction**:
   - Load `lexicanum_edges.json` and `lexicanum_characters.json`.
   - Construct a **Directed Graph** `G`.
   
2. **Metric Calculation** (Store as node attributes):
   - **Betweenness Centrality**: To identify bridge nodes connecting different parts of the network.
   - **PageRank**: To identify global importance/influence.
   - **Clustering Coefficient**: To measure how tightly knit each character's immediate circle is.

3. **Community Detection**:
   - Apply the **Louvain Algorithm** (using `community` / `python-louvain` or `networkx`) to assign a discrete "Network Community" ID to each node.
   - *Note*: Louvain typically requires an undirected graph; we will assume the undirected version for community detection.

**Deliverable**: Graph `G` enriched with structural attributes.

### **Phase 2: Semantic Feature Extraction (NLP)**
*Focus: Characterizing thematic content and archetypes.*

**Tasks:**
1. **Preprocessing**:
   - Clean character descriptions (from `lexicanum_characters.json` or raw text files).
   - Lowercase, remove stopwords.

2. **Embeddings Generation**:
   - Use `sentence-transformers` (Model: `all-MiniLM-L6-v2`).
   - Generate dense vector embeddings for every character description.
   - *Performance Note*: Implement batching (e.g., batch size 32 or 64) to handle ~12,000 nodes efficiently.

3. **Semantic Clustering (Archetypes)**:
   - Apply **K-Means clustering** to the embeddings.
   - Set $K$ (e.g., $K=20$) to group characters by "Archetype" (e.g., Soldiers, Deities, Politicians).
   - Assign "Semantic Cluster ID" to each node.

4. **Cluster Labeling (Vocabulary Analysis)**:
   - Collect all text for characters in each Semantic Cluster.
   - Apply **TF-IDF** to find top unique keywords for each cluster.
   - Generate labels (e.g., Cluster 0: "God, Throne, Ancient").

**Deliverable**: Node embeddings, Semantic Cluster IDs, and Cluster Keywords.

### **Phase 3: Hybrid Analysis (The Core Logic)**
*Focus: Integrating Structure and Semantics.*

**Tasks:**
1. **Calculate Neighbor Semantic Consistency**:
   - For each node $u$:
     - Identify neighbors $v \in N(u)$.
     - Retrieve embeddings $E_u$ and $E_v$.
     - Calculate **Cosine Similarity** between $E_u$ and each $E_v$.
     - Compute the **Average Cosine Similarity**.
   - *Interpretation*: 
     - High Avg Sim = "Narrative Agent" (Consistent local context).
     - Low Avg Sim = "Mythological Anchor" (Bridges distinct contexts).

2. **Metric Correlation**:
   - Calculate **Pearson correlation** between **Betweenness Centrality** and **Neighbor Semantic Consistency**.
   - Test if high betweenness (bridges) correlates with low semantic consistency (mythological status).

**Deliverable**: "Narrative Independence Score" (Neighbor Semantic Consistency) for all nodes.

### **Phase 4: Visualization & Output**
*Focus: Reporting results.*

**Tasks:**
1. **Scatter Plot Visualization**:
   - **X-axis**: Betweenness Centrality (Log Scale).
   - **Y-axis**: Neighbor Semantic Consistency.
   - **Color**: Semantic Cluster ID.
   - **Goal**: Highlight outliers. High Centrality + Low Consistency should be "Mythological Anchors".

2. **Data Export**:
   - Generate `mythology_vs_narrative_analysis.csv` with columns:
     - `Node`
     - `Betweenness`
     - `PageRank`
     - `Network_Community_ID`
     - `Semantic_Cluster_ID`
     - `Top_Keywords`
     - `Neighbor_Consistency_Score`

---

## Implementation Strategy

### 1. Environment Setup
We need to install the following libraries:
```bash
pip install sentence-transformers scikit-learn python-louvain pandas plotly seaborn
```

### 2. Code Structure
We will create a new main analysis script: `src/mythology_analysis.py`.

**Modules**:
- `src/topology.py`: NetworkX operations (Phase 1).
- `src/semantics.py`: Embeddings, K-Means, TF-IDF (Phase 2).
- `src/hybrid.py`: Consistency calculation, Correlations (Phase 3).
- `src/vis_utils.py`: Plotting and Export (Phase 4).

### 3. Execution Steps
1. **Setup**: Install dependencies.
2. **Refine Data**: Ensure `lexicanum_characters.json` has clean descriptions. If descriptions are empty, fallback to wikitext snippets or file content.
3. **Pipeline Development**: Implement phases sequentially in `src/mythology_analysis.py`.
4. **Run & Tune**: Run on the full 12k dataset (expect ~5-10 mins for embeddings on CPU).

---

## Success Criteria
- A CSV file containing the "Narrative Independence Score" for key characters.
- A Scatter Plot clearly identifying characters like "The Emperor" or "Horus" as outliers compared to standard book characters.
- Validation of the hypothesis via correlation metrics.
