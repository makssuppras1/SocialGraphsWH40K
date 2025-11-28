# Project Description: Warhammer 40,000 Character Network Analysis

## Project Overview

This project constructs and analyzes a social network graph of characters from the Warhammer 40,000 (WH40K) fictional universe, extracted from the Lexicanum wiki. The research applies network science methodologies to understand the structure, relationships, and community organization of characters within this rich fictional narrative ecosystem.

**Research Context**: This is a final assignment for the DTU course "02805 Social graphs and interactions", demonstrating the application of network analysis techniques to a large-scale fictional character network.

## Research Questions and Objectives

The project aims to answer several key questions:
1. What is the structural topology of the WH40K character network?
2. How do characters cluster into communities, and do these align with their faction affiliations?
3. What are the centrality patterns and how do different centrality measures correlate?
4. Does the network exhibit degree assortativity (do high-degree characters connect to other high-degree characters)?
5. What is the degree distribution, and does it follow a power-law or other characteristic distribution?

## Data Source

**Primary Data**: Lexicanum wiki (https://wh40k.lexicanum.com)
- **Raw Data**: Wiki page texts extracted from 13 batch JSON files (`lexicanum_page_texts_batch_*.json`)
- **Character Categories**: Pre-processed character categorization data (`lexicanum_characters_by_category_generated.json`)
- **Portal Mapping**: Comprehensive mapping of faction portals to 16 main factions (`faction_portal_affiliations_comprehensive.json`)

**Network Scale**:
- Original network: ~12,401 nodes (characters), ~15,978 edges (character-to-character links)
- Filtered network: Largest connected component, undirected, nodes with degree ≥ 2

## Methodology

### Phase 1: Network Construction (`create_network.py`)

The network construction follows a 5-step pipeline:

1. **Character-to-Affiliation Mapping**
   - Processes character categories to extract affiliations
   - Maps each character to their primary affiliation and all affiliations
   - Identifies race/species separately from organizational affiliations
   - Output: `character_affiliation_mapping.json`

2. **Character Entry Creation**
   - Processes all wiki pages to identify character pages
   - Filters out non-character pages (lists, formations, categories)
   - Extracts faction portal templates (e.g., `{{AstraMilitarumPortal}}`) from wikitext
   - Maps extracted portals to one of 16 main faction portals using comprehensive mapping
   - Assigns portal based on affiliation if not found in wikitext
   - Output: `lexicanum_characters.json`

3. **Edge Extraction**
   - Parses wikitext to find wiki link patterns `[[Character Name]]`
   - Creates directed edges from source character to target character
   - Filters out portal links, special namespaces, and self-loops
   - Handles character name variations and canonicalization
   - Output: `lexicanum_edges.json`

4. **NetworkX Graph Construction**
   - Creates directed NetworkX graph
   - Adds nodes with attributes: name, affiliation, race, portals, all_affiliations
   - Adds weighted edges (multiple links between same characters increase weight)
   - Calculates basic network statistics

5. **Graph Export**
   - Exports to GEXF format (for Gephi visualization)
   - Exports to Pickle format (preserves all Python data types)

### Phase 2: Network Filtering (`create_filtered_network.py`)

Creates a filtered version for analysis:
1. Converts directed graph to undirected
2. Keeps only the largest connected component
3. Iteratively removes nodes with degree < 2
4. Output: `lexicanum_network_filtered.pkl` and `.gexf`

### Phase 3: Network Analysis (`Analyses.py`)

Comprehensive analysis suite:

1. **Degree Distribution Analysis**
   - Calculates degree statistics (mean, median, max, min)
   - Creates histogram and log-log plots
   - Tests for power-law distribution characteristics

2. **Centrality Measures**
   - **Degree Centrality**: Number of connections
   - **Betweenness Centrality**: Importance as bridge between nodes
   - **Eigenvector Centrality**: Importance based on connections to important nodes
   - Calculates Pearson correlations between centrality measures
   - Identifies top characters by each centrality measure

3. **Degree Assortativity**
   - Calculates degree assortativity coefficient
   - Plots node degree vs. average neighbor degree
   - Determines if network is assortative (high-high connections) or disassortative (high-low connections)

4. **Network Backbone Extraction**
   - Uses disparity filter to extract network backbone
   - Removes weak edges based on relative edge weight threshold (default: 0.05)
   - Visualizes backbone structure (if network size permits)

5. **Community Detection**
   - Applies Louvain algorithm for community detection
   - Calculates modularity for detected communities
   - Compares detected communities with faction-based partitions
   - Creates confusion matrix: 16 main factions vs. top 16 detected communities
   - Calculates modularity for faction-based partition
   - Visualizes communities (if network size permits)

## The 16 Main Faction Portals

Characters are mapped to one of these main factions:
1. Adeptus Custodes
2. Mechanicum
3. Astra Militarum
4. Chaos
5. Drukhari
6. Asuryani
7. Genestealer Cults
8. Imperium
9. Inquisition
10. Leagues of Votann
11. Adepta Sororitas
12. Space Marines
13. Necrons
14. Orks
15. T'au
16. Tyranids

## Key Technical Details

### Character Filtering
- Excludes pages matching patterns: "list of", "known members of", "formation", "category:", etc.
- Only includes individual character pages

### Portal Mapping Logic
- Extracts portal templates from wikitext: `{{NamePortal}}`
- Normalizes portal names (handles plurals, camelCase variations)
- Maps sub-portals to main portals using comprehensive mapping file
- If multiple portals found, uses the last one (conflict resolution)
- Falls back to affiliation-based portal assignment if no portal found in wikitext

### Edge Creation
- Directed edges represent character mentions/links in wiki pages
- Multiple mentions create weighted edges
- Self-loops are excluded
- Portal links and special namespace links are filtered out

### Network Attributes

**Node Attributes**:
- `name`: Display name (title without parenthetical info)
- `title`: Canonical title
- `affiliation`: Primary organizational affiliation
- `race`: Species/race (Human, Ork, Eldar, etc.)
- `all_affiliations`: List of all affiliations
- `affiliation_count`: Number of affiliations
- `portals`: Main faction portal (string, comma-separated if multiple)
- `portal_list`: List of main faction portals
- `pageid`: Lexicanum page ID

**Edge Attributes**:
- `weight`: Number of times the link appears (default: 1)

## Output Files

### Data Files
- `character_affiliation_mapping.json`: Character-to-affiliation mappings
- `lexicanum_characters.json`: All character entries with attributes
- `lexicanum_edges.json`: All character-to-character edges
- `lexicanum_network.pkl`: Original directed network (pickle)
- `lexicanum_network.gexf`: Original network (GEXF format)
- `lexicanum_network_filtered.pkl`: Filtered undirected network (pickle)
- `lexicanum_network_filtered.gexf`: Filtered network (GEXF format)

### Analysis Output (Images)
- `degree_distribution.png`: Degree distribution histogram and log-log plot
- `centrality_correlations.png`: Scatter plots of centrality measure correlations
- `assortativity.png`: Degree assortativity scatter plot
- `faction_community_confusion_matrix.png`: 16×16 heatmap comparing factions vs. communities
- `network_backbone.png`: Visualization of network backbone (if generated)
- `community_detection.png`: Community visualization (if generated)

## Key Findings (Expected Analysis Results)

The analysis script produces quantitative results including:

1. **Network Topology**: Basic statistics (nodes, edges, density, connected components)
2. **Degree Distribution**: Whether the network follows a power-law or other distribution
3. **Centrality Rankings**: Top characters by different centrality measures (e.g., Roboute Guilliman, Emperor of Mankind)
4. **Centrality Correlations**: How degree, betweenness, and eigenvector centralities relate
5. **Assortativity**: Whether the network is assortative or disassortative
6. **Community Structure**: Number of communities, modularity scores, alignment with factions
7. **Faction-Community Alignment**: Confusion matrix showing how well detected communities match faction affiliations

## Technical Stack

- **Python 3.12+**
- **NetworkX**: Graph construction and analysis
- **NumPy**: Numerical operations
- **Matplotlib**: Plotting and visualization
- **SciPy**: Statistical functions (Pearson correlation)
- **Seaborn**: Heatmap visualization (optional, for confusion matrix)

## Project Structure

```
Final Assignment/
├── src/
│   ├── create_network.py          # Network construction pipeline
│   ├── create_filtered_network.py # Network filtering
│   ├── Analyses.py                # Comprehensive network analysis
│   ├── config.py                  # Configuration and paths
│   └── helpers.py                 # Utility functions
├── raw_data/                      # Input data (wiki pages, categories)
├── data/                          # Generated network files
├── images/                        # Analysis output plots
└── README.md                      # Project documentation
```

## Scientific Paper Context

This project demonstrates:
- **Network Science Applications**: Applying real-world network analysis techniques to fictional narrative data
- **Community Detection**: Comparing algorithmic community detection with domain knowledge (faction affiliations)
- **Centrality Analysis**: Understanding character importance through multiple centrality measures
- **Network Topology**: Characterizing the structure of a large-scale character relationship network
- **Data Mining**: Extracting structured network data from unstructured wiki text

The research contributes to understanding how fictional universes can be modeled as networks, and how network science methods can reveal structural patterns in narrative relationships.

## Usage Notes

1. **Network Creation**: Run `create_network.py` to build the network from raw data
2. **Network Filtering**: Run `create_filtered_network.py` to create filtered version
3. **Analysis**: Run `Analyses.py` to perform all analyses and generate plots

All scripts use relative paths configured in `config.py` and assume data files are in the expected locations.

