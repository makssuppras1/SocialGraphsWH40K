# How Portal-Cluster Comparison Relates to the Research Question

## The Original Research Question

> **"To what extent does the hyperlink-based character network encode the faction structure described in textual lore, and what network mechanisms explain alignment or misalignment between detected communities and faction labels?"**

## Three Layers of Grouping

We now have **three different ways** of grouping characters:

### 1. **Network Communities** (Structural - from hyperlinks)
- **Source**: Hyperlink structure (who links to whom)
- **Method**: Louvain algorithm
- **Represents**: Narrative relationships, story arcs, who appears together in wiki pages
- **Example**: Characters that appear in the same storylines/campaigns cluster together

### 2. **Portal-Based Factions** (Text-derived labels - from wiki templates)
- **Source**: Wiki portal templates (`{{AstraMilitarumPortal}}`)
- **Method**: Template extraction + mapping to 16 main factions
- **Represents**: Official faction affiliations from wiki metadata
- **Example**: Characters explicitly tagged as "Space Marines", "Chaos", etc.

### 3. **Semantic Clusters** (Text-derived content - from descriptions)
- **Source**: Character description text content
- **Method**: Embeddings + K-Means clustering (16 clusters)
- **Represents**: Thematic similarity based on what the text says about characters
- **Example**: Characters with similar descriptions cluster together (e.g., "Primarchs", "Gaunt's Ghosts")

## How They Relate to the Research Question

### The Core Question Breakdown:

**"Does the hyperlink network encode faction structure?"**

This can be answered by comparing:
- **Network Communities** (from hyperlinks) vs **Portal Factions** (from wiki labels)

**"What explains alignment/misalignment?"**

This is where **Semantic Clusters** become crucial:

1. **Portal vs Semantic Cluster** comparison reveals:
   - **High alignment** (e.g., Adepta Sororitas 85.1%) → Faction has coherent semantic identity
   - **Low alignment** (e.g., Space Marines 14.0%) → Faction is semantically diverse

2. **Network Community vs Semantic Cluster** comparison reveals:
   - Do structural communities align with semantic themes?
   - Are communities formed around shared narrative themes?

3. **All Three Together** explain misalignment:
   - If Network Communities ≠ Portal Factions, but Network Communities ≈ Semantic Clusters
   - Then: Characters connect through **narrative themes** (semantics) rather than **faction labels** (portals)
   - This explains why hyperlink structure doesn't perfectly encode faction structure!

## The Complete Analysis Framework

```
┌─────────────────────────────────────────────────────────────┐
│                    RESEARCH QUESTION                        │
│  "Does hyperlink network encode faction structure?"         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │   Network Communities (Louvain)       │
        │   [Structural grouping from links]   │
        └───────────────────────────────────────┘
                    │                    │
                    │                    │
        ┌───────────▼────────┐  ┌───────▼──────────────┐
        │ Portal Factions    │  │ Semantic Clusters    │
        │ [Wiki templates]   │  │ [Text content]       │
        └────────────────────┘  └──────────────────────┘
                │                        │
                └──────────┬─────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │   THREE-WAY COMPARISON                │
        │   1. Network Communities vs Portals   │
        │   2. Network Communities vs Clusters │
        │   3. Portals vs Clusters             │
        └──────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │   EXPLANATION OF MISALIGNMENT        │
        │   - Why don't communities = factions? │
        │   - What mechanisms drive alignment?  │
        └──────────────────────────────────────┘
```

## What the Portal-Cluster Comparison Tells Us

### For the Research Question:

1. **Faction Semantic Coherence**:
   - **High alignment** (Adepta Sororitas 85.1%) → Faction has strong semantic identity
     - Characters in this faction share similar text themes
     - Network communities should align well with this faction
   - **Low alignment** (Space Marines 14.0%) → Faction is semantically diverse
     - Characters span many semantic themes
     - Network communities will likely split this faction across multiple communities

2. **Mechanism Explanation**:
   - If **Portals ≠ Semantic Clusters** → Faction labels don't capture semantic themes
   - If **Network Communities ≈ Semantic Clusters** → Hyperlinks follow semantic themes
   - Therefore: **Network Communities ≠ Portals** because links follow semantics, not labels!

3. **The "Why" Behind Misalignment**:
   - **Space Marines** (14% alignment) spans 14 semantic clusters
     - Different chapters have different narrative themes
     - Network communities will split Space Marines across multiple communities
     - This explains why hyperlink structure doesn't encode Space Marines as a single community
   - **Adepta Sororitas** (85% alignment) is mostly in one semantic cluster
     - Strong semantic coherence
     - Network communities should align better with this faction

## The Complete Answer to the Research Question

### "To what extent does the network encode faction structure?"

**Answer**: Partially, but with important exceptions:
- **High semantic coherence factions** (Adepta Sororitas, Inquisition) → Better alignment
- **Low semantic coherence factions** (Space Marines, Chaos) → Poor alignment

### "What explains alignment/misalignment?"

**Answer**: **Semantic coherence** is the key mechanism:

1. **High Alignment** (Portal ≈ Semantic Cluster):
   - Faction has coherent semantic identity
   - Characters share similar text themes
   - Network communities align with faction because links follow semantic themes

2. **Low Alignment** (Portal ≠ Semantic Cluster):
   - Faction is semantically diverse
   - Characters span multiple semantic themes
   - Network communities split the faction because links follow semantics, not labels

3. **The Mechanism**:
   - Hyperlinks reflect **narrative relationships** (who appears together in stories)
   - Narrative relationships follow **semantic themes** (similar characters appear together)
   - Faction labels are **organizational categories** (may not match narrative themes)
   - Therefore: Network structure encodes **semantic/narrative structure**, not necessarily **faction structure**

## Next Steps for Complete Analysis

To fully answer the research question, we should also compare:

1. **Network Communities vs Portals** (already done in `Analyses.py`)
   - Confusion matrix showing how Louvain communities map to factions
   - Modularity comparison

2. **Network Communities vs Semantic Clusters** (NEW - should be done)
   - Do structural communities align with semantic themes?
   - This would complete the triangle of comparisons

3. **Three-way analysis**:
   - When do all three align? (Portal = Cluster = Community)
   - When do they diverge? And why?

This would provide a complete answer to: "What network mechanisms explain alignment or misalignment?"

