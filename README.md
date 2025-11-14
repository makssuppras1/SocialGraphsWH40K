# Warhammer 40k Character Network from Lexicanum

This project creates a network graph of Warhammer 40,000 characters from Lexicanum wiki data, extracting character relationships, affiliations, and faction portals.

## Overview

The project processes raw Lexicanum wiki page data to:
1. Extract character information and affiliations
2. Map characters to one of 16 main faction portals
3. Build a directed network graph based on character mentions/links
4. Export the network in multiple formats for analysis

## Project Structure

```
Final Assignment/
├── notebooks/
│   ├── create_network.py    # Main script to build the network
│   ├── config.py             # Configuration and file paths
│   └── helpers.py            # Utility functions
├── raw_data/                 # Input data files
│   ├── lexicanum_page_texts_batch_*.json
│   └── lexicanum_characters_by_category_generated.json
└── data/                     # Generated output files
    ├── character_affiliation_mapping.json
    ├── lexicanum_characters.json
    ├── lexicanum_edges.json
    ├── lexicanum_network.graphml
    ├── lexicanum_network.gexf
    ├── lexicanum_network.pkl
    └── faction_portal_affiliations_filtered.json
```

## Requirements

- Python 3.12+
- Required packages:
  - `networkx` - Graph creation and analysis
  - `numpy` - Numerical operations
  - `pandas` - Data manipulation (optional, for analysis)

## Setup

1. **Install dependencies** (if using a virtual environment):
   ```bash
   pip install networkx numpy pandas
   ```

2. **Ensure data files are in place**:
   - `raw_data/lexicanum_page_texts_batch_*.json` - Wiki page data
   - `raw_data/lexicanum_characters_by_category_generated.json` - Character categories
   - `data/faction_portal_affiliations_comprehensive.json` - Portal mapping file

## Usage

### Running the Network Creation Script

```bash
cd notebooks
python create_network.py
```

Or using pyenv:
```bash
cd notebooks
pyenv local SoGraph
python create_network.py
```

### What the Script Does

The script executes 5 main steps:

1. **Build Character-to-Affiliation Mapping**
   - Loads character categories from `lexicanum_characters_by_category_generated.json`
   - Maps each character to their affiliations
   - Determines primary affiliation based on specificity
   - Outputs: `data/character_affiliation_mapping.json`

2. **Create Character Entries**
   - Loads all wiki pages from batch files
   - Extracts portal templates (e.g., `{{AstraMilitarumPortal}}`)
   - Maps extracted portals to one of 16 main portals using the mapping file
   - Filters out non-character pages (lists, formations, etc.)
   - Outputs: `data/lexicanum_characters.json`

3. **Extract Edges**
   - Processes wikitext to find character-to-character links
   - Filters out portal links and special namespaces
   - Creates directed edges between characters
   - Outputs: `data/lexicanum_edges.json`

4. **Build NetworkX Graph**
   - Creates a directed graph with all characters as nodes
   - Adds node attributes: name, affiliation, portals, etc.
   - Adds edges with weights
   - Calculates network statistics

5. **Save Graph**
   - Exports to GEXF format (for Gephi)
   - Exports to Pickle format (preserves all Python data types)

## Output Files

### `character_affiliation_mapping.json`
Maps each character to their primary and all affiliations:
```json
{
  "Character Name": {
    "primary_affiliation": "Astra Militarum",
    "all_affiliations": ["Astra Militarum", "Human", "Tyrannic Wars"],
    "affiliation_count": 3
  }
}
```

### `lexicanum_characters.json`
Character entries with affiliations and mapped portals:
```json
{
  "title": "Character Name",
  "name": "Display Name",
  "affiliation": "Primary Affiliation",
  "portals": ["Astra Militarum"]
}
```

### `lexicanum_edges.json`
Directed edges between characters:
```json
{
  "source": "Character A",
  "target": "Character B"
}
```

### Network Files
- **`lexicanum_network.gexf`** - GEXF format (for Gephi visualization)
- **`lexicanum_network.pkl`** - Pickle format (preserves all data types including lists)
- **`lexicanum_network.graphml`** - GraphML format (if present, from previous runs)

## The 16 Main Portals

Characters are mapped to one of these main faction portals:
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

## Portal Mapping Logic

The script uses `faction_portal_affiliations_comprehensive.json` to map sub-portals to main portals:
- Extracts portal templates from wikitext (e.g., `{{AstraMilitarumPortal}}`)
- Normalizes portal names (handles plurals, camelCase, etc.)
- Matches to keys in the mapping file
- If multiple portals are listed, selects the **last one** (conflict resolution)
- Returns the mapped main portal name

## Network Statistics

The script outputs network statistics including:
- Basic properties (nodes, edges, density)
- Connected components (weakly and strongly connected)
- Degree statistics (in-degree, out-degree, total degree)
- Top characters by degree centrality
- Top affiliations by node count

## Code Structure

### `create_network.py`
Main script that orchestrates the network creation process:
- `build_affiliation_mapping()` - Step 1
- `create_character_entries()` - Step 2
- `extract_edges()` - Step 3
- `build_network()` - Step 4
- `save_graph()` - Step 5

### `config.py`
Centralized configuration:
- File paths for input/output
- Directory paths
- Batch file patterns

### `helpers.py`
Utility functions:
- `extract_portals()` - Extract and map portals from wikitext
- `normalize_portal_name()` - Handle name variations
- `is_character()` - Filter out non-character pages
- `load_portal_mapping()` - Load portal mapping file
- `match_portal_to_key()` - Match portal names to mapping keys
- `map_portal_to_main_portals()` - Map to main portals
- `load_batch_files()` - Load wiki page data
- `create_character_title_map()` - Create title lookup maps

## Example Output

```
============================================================
NETWORK STATISTICS
============================================================

Basic Properties:
  Nodes: 12401
  Edges: 15978
  Density: 0.000104

Connected Components:
  Weakly connected components: 5811
  Largest component size: 5317 nodes (42.88%)

Top 10 Characters by Degree Centrality:
   1. Roboute Guilliman (High Lords of Terra): degree=202
   2. Emperor of Mankind (Imperial): degree=200
   3. Rogal Dorn (Horus Heresy): degree=188
   ...
```

## Notes

- The script filters out non-character pages (lists, formations, categories)
- Portal links are excluded from edge extraction
- Self-loops are not created
- Multiple edges between the same characters are weighted
- List attributes are converted to strings for GraphML/GEXF export
- Pickle format preserves all original data types

## License

This project is for educational purposes as part of the DTU course "02805 Social graphs and interactions".

