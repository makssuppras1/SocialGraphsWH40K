# Final Project Movie Script: Warhammer 40k Character Network Analysis

## Script Outline (60-90 seconds)

## 1. Central Idea (15 seconds)

**What**: Analyzing the social network of Warhammer 40,000 characters from Lexicanum wiki. Building a character co-occurrence network based on mentions and links, enriched with faction affiliations and portal information.

**Why interesting**: 
- Thousands of interconnected characters across a rich fictional universe
- Combines network structure analysis with text content analysis
- Can identify important characters, narrative communities, and story relationships

## 2. Datasets & Collection (10 seconds)

**Data Source**: Lexicanum wiki (lexicanum.com) - scraped via API/wiki export

**Datasets**:
- 13 batch files of page wikitext (~20 MB raw data)
- Character category mappings for affiliations
- **Total**: 12,454 pages → 12,401 characters after filtering

## 3. Data Analysis (15 seconds)

**Data Size**:
- Raw: ~20 MB (13 JSON batch files)
- Processed: ~7 MB (characters, edges, mappings)
- Network files: 5.5-7 MB (GraphML, GEXF)

**Network**:
- **12,401 nodes** (characters)
- **15,978 edges** (character mentions/links)
- **5,811 components**, largest has 5,317 nodes (43%)
- **Top characters**: Roboute Guilliman (202 links), Emperor of Mankind (200), Rogal Dorn (188)

**Node Attributes**: Affiliation, portals, all affiliations list

## 4. Text Analysis (10 seconds)

**Text Source**: Wikitext from character pages

**Extracted**:
- Wiki links (`[[Character]]`) → network edges
- Portal tags (`{{Portal}}`) → node attributes
- Category tags → affiliations

**Stats**: 6,520 characters (52%) have outgoing links, ~1.5 links per character page

## 5. Combining Networks & Text (10 seconds)

**Integration**:
- Network structure built from text links (each `[[Character]]` = edge)
- Text attributes (portals, categories) enrich nodes
- Combined analysis: network communities vs. affiliation clusters, centrality vs. textual prominence

**Research**: How do network communities align with factions? How does text content predict network position?

## Visual Elements

1. **Network visualization** - Force-directed layout, colored by affiliation
2. **Key stats on screen** - 12,401 nodes, 15,978 edges, top characters
3. **Data pipeline** - Raw → Processed → Network → Analysis
4. **Top characters list** - Roboute Guilliman, Emperor, etc.

## Quick Reference (On-Screen Stats)

- **12,401 characters**
- **15,978 edges**
- **~20 MB raw data**
- **Largest component: 5,317 nodes (43%)**
- **Top: Roboute Guilliman (202 links)**
