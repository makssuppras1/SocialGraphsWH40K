#!/usr/bin/env python3
"""
Create Warhammer 40k Character Network from Lexicanum Data

This script:
1. Builds character-to-affiliation mapping from category data
2. Creates character entries with affiliations and portals
3. Extracts edges from wikitext links
4. Builds NetworkX graph with all attributes
5. Saves graph in multiple formats (GEXF, Pickle)
"""

import json
import re
import pickle
import numpy as np
import networkx as nx
from collections import Counter
from pathlib import Path

from config import (
    RAW_DATA_PATH, DATA_PATH,
    CHAR_CATEGORIES_FILE, AFFILIATION_MAPPING_FILE,
    CHARACTERS_FILE, EDGES_FILE,
    GEXF_FILE, PICKLE_FILE,
    PORTAL_MAPPING_FILE, BATCH_FILE_PATTERN
)
from helpers import (
    extract_portals, is_character, determine_primary_affiliation, extract_race,
    load_batch_files, create_character_title_map, find_character_title,
    load_portal_mapping, match_portal_to_key
)


def build_affiliation_mapping():
    """Step 1: Build character-to-affiliation mapping from category data."""
    print("="*60)
    print("STEP 1: Building Character-to-Affiliation Mapping")
    print("="*60)
    
    # Load character categories
    with open(CHAR_CATEGORIES_FILE, 'r', encoding='utf-8') as f:
        char_categories = json.load(f)
    
    print(f"Loaded {len(char_categories)} categories from Lexicanum data")
    
    # Map characters to their affiliations
    character_affiliations = {}
    for category, characters in char_categories.items():
        match = re.match(r"Characters\s*\(([^)]+)\)", category)
        if match:
            affil_name = match.group(1).strip()
            for char in characters:
                if char not in character_affiliations:
                    character_affiliations[char] = set()
                character_affiliations[char].add(affil_name)
    
    print(f"Found {len(set(affil for affils in character_affiliations.values() for affil in affils))} unique character affiliations")
    print(f"Mapped {len(character_affiliations)} characters to affiliations")
    
    # Create structured output
    character_affil_mapping = {}
    for char, affil_set in character_affiliations.items():
        affil_list = sorted(list(affil_set))
        character_affil_mapping[char] = {
            "primary_affiliation": determine_primary_affiliation(affil_set),
            "race": extract_race(affil_set),
            "all_affiliations": affil_list,
            "affiliation_count": len(affil_list)
        }
    
    # Save mapping
    with open(AFFILIATION_MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(character_affil_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved character-to-affiliation mapping to {AFFILIATION_MAPPING_FILE}")
    print_sample_entries(character_affil_mapping, 5)
    
    return character_affil_mapping


def create_character_entries(char_affil_mapping):
    """Step 2: Create character entries with affiliations and portals."""
    print("\n" + "="*60)
    print("STEP 2: Creating Character Entries with Affiliations and Portals")
    print("="*60)
    
    print(f"Loaded affiliation mapping for {len(char_affil_mapping)} characters")
    
    # Load portal mapping to map sub-portals to 16 main portals
    portal_mapping = load_portal_mapping(PORTAL_MAPPING_FILE)
    if portal_mapping:
        print(f"Loaded portal mapping with {len(portal_mapping)} entries")
    else:
        print("Warning: Portal mapping file not found, using raw portal extraction")
    
    # Load all pages from batch files
    all_pages, batch_files = load_batch_files(RAW_DATA_PATH, BATCH_FILE_PATTERN)
    print(f"Found {len(batch_files)} batch files")
    print(f"Total pages collected: {len(all_pages)}")
    
    # Create character entries
    entries = []
    stats = {
        "with_affil": 0, 
        "without_affil": 0, 
        "skipped": 0,
        "with_portal": 0,
        "without_portal": 0
    }
    
    for title, page_data in sorted(all_pages.items()):
        if not is_character(title):
            stats["skipped"] += 1
            continue
        
        display_name = re.sub(r"\s*\([^)]*\)$", "", title).strip()
        file_name = title.replace("/", "_").replace("\\", "_")
        wikitext = page_data.get("wikitext", "")
        # Extract portals and map them to main portals using the comprehensive mapping
        portals = extract_portals(wikitext, portal_mapping)
        
        # Look up affiliations
        affil_data = char_affil_mapping.get(title) or char_affil_mapping.get(display_name)
        if affil_data:
            stats["with_affil"] += 1
            affiliation = affil_data.get("primary_affiliation")
            race = affil_data.get("race")
            all_affiliations = affil_data.get("all_affiliations", [])
            affiliation_count = affil_data.get("affiliation_count", 0)
        else:
            stats["without_affil"] += 1
            affiliation = None
            race = None
            all_affiliations = []
            affiliation_count = 0
        
        # If no portal found in wikitext, try to assign from affiliations
        if not portals and portal_mapping:
            assigned_portal = None
            
            # First try primary affiliation
            if affiliation:
                matched_key = match_portal_to_key(affiliation, portal_mapping)
                if matched_key:
                    main_portals = portal_mapping[matched_key]
                    if main_portals:
                        assigned_portal = main_portals[-1] if isinstance(main_portals, list) else main_portals
            
            # If no match, try all affiliations
            if not assigned_portal and all_affiliations:
                for affil in all_affiliations:
                    matched_key = match_portal_to_key(affil, portal_mapping)
                    if matched_key:
                        main_portals = portal_mapping[matched_key]
                        if main_portals:
                            assigned_portal = main_portals[-1] if isinstance(main_portals, list) else main_portals
                            break  # Use first match
            
            if assigned_portal:
                portals = [assigned_portal]
        
        # Track portal statistics
        if portals:
            stats["with_portal"] += 1
        else:
            stats["without_portal"] += 1
        
        entries.append({
            "title": title,
            "name": display_name,
            "file_name": file_name,
            "pageid": page_data.get("pageid"),
            "affiliation": affiliation,
            "race": race,
            "all_affiliations": all_affiliations,
            "affiliation_count": affiliation_count,
            "portals": portals
        })
    
    # Save characters
    with open(CHARACTERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    
    print(f"\nStored {len(entries)} characters in {CHARACTERS_FILE}")
    print(f"Characters with affiliations: {stats['with_affil']}")
    print(f"Characters without affiliations: {stats['without_affil']}")
    print(f"Characters with portals: {stats['with_portal']} ({100*stats['with_portal']/len(entries):.1f}%)")
    print(f"Characters without portals: {stats['without_portal']} ({100*stats['without_portal']/len(entries):.1f}%)")
    print(f"Non-character pages skipped: {stats['skipped']}")
    
    # Show portal distribution
    portal_counts = Counter()
    for entry in entries:
        for portal in entry.get("portals", []):
            portal_counts[portal] += 1
    
    if portal_counts:
        print(f"\nPortal distribution (top 10):")
        for portal, count in portal_counts.most_common(10):
            print(f"  {portal}: {count} characters")
    
    print_sample_characters(entries, 5)
    
    return entries


def extract_edges(characters):
    """Step 3: Extract edges from wikitext links."""
    print("\n" + "="*60)
    print("STEP 3: Extracting Edges from Wikitext")
    print("="*60)
    
    print(f"Loaded {len(characters)} characters")
    
    # Create character title mapping
    character_titles, character_title_map = create_character_title_map(characters)
    print(f"Created lookup set with {len(character_titles)} variations")
    
    # Load all pages
    all_pages, _ = load_batch_files(RAW_DATA_PATH, BATCH_FILE_PATTERN)
    print(f"Processing {len(all_pages)} pages to extract edges...")
    
    # Extract edges
    edges = []
    characters_with_links = 0
    
    for source_title, page_data in all_pages.items():
        if source_title not in character_titles or not is_character(source_title):
            continue
        
        wikitext = page_data.get("wikitext", "")
        if not wikitext:
            continue
        
        # Extract wiki links
        wiki_links = re.findall(r'\[\[([^\]|#:]+)(?:\|[^\]]+)?\]\]', wikitext)
        
        found_links = False
        for link in wiki_links:
            link = link.strip()
            
            # Skip special namespaces and portal links
            if ":" in link or "portal" in link.lower():
                continue
            
            # Find character title
            target_title = find_character_title(link, character_title_map)
            if not target_title or not is_character(target_title):
                continue
            
            # Skip self-loops
            if target_title == source_title:
                continue
            
            edges.append({"source": source_title, "target": target_title})
            found_links = True
        
        if found_links:
            characters_with_links += 1
    
    # Save edges
    with open(EDGES_FILE, 'w', encoding='utf-8') as f:
        json.dump(edges, f, indent=2, ensure_ascii=False)
    
    print(f"\nExtracted {len(edges)} edges")
    print(f"Characters with outgoing links: {characters_with_links}")
    print(f"Saved edges to {EDGES_FILE}")
    print_sample_edges(edges, 10)
    
    return edges


def build_network(edges, characters):
    """Step 4: Build NetworkX graph with all attributes."""
    print("\n" + "="*60)
    print("STEP 4: Building NetworkX Graph")
    print("="*60)
    
    print(f"Loaded {len(edges)} edges from file")
    print(f"Loaded {len(characters)} characters")
    
    # Create character lookup
    char_dict = {char["title"]: char for char in characters}
    
    # Build directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for char in characters:
        title = char["title"]
        if not is_character(title):
            continue
        
        portals = char.get("portals", [])
        portals_str = ', '.join(portals) if portals else ''
        
        G.add_node(
            title,
            name=char["name"],
            file_name=char.get("file_name", ""),
            pageid=char.get("pageid"),
            affiliation=char.get("affiliation"),
            race=char.get("race"),
            all_affiliations=char.get("all_affiliations", []),
            affiliation_count=char.get("affiliation_count", 0),
            portals=portals_str,
            portal_list=portals
        )
    
    print(f"Added {G.number_of_nodes()} nodes to graph")
    
    # Add edges
    for edge in edges:
        source, target = edge["source"], edge["target"]
        if source in G and target in G:
            if G.has_edge(source, target):
                G[source][target]["weight"] += 1
            else:
                G.add_edge(source, target, weight=1)
    
    print(f"Added {G.number_of_edges()} edges to graph")
    
    # Print statistics
    print_network_statistics(G, char_dict)
    
    return G


def save_graph(G):
    """Step 5: Save graph in multiple formats."""
    print("\n" + "="*60)
    print("STEP 5: Saving Graph in Multiple Formats")
    print("="*60)
    
    # Create export copy (convert lists to strings)
    G_export = prepare_graph_for_export(G)
    
    # Save GEXF
    save_gexf(G_export, GEXF_FILE)
    
    # Save Pickle (preserves all data types)
    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump(G, f)
    print(f"Saved graph to {PICKLE_FILE} (preserves all data types including lists)")
    
    print("\n" + "="*60)
    print("Network creation complete!")
    print("="*60)
    print(f"\nGraph ready for analysis!")
    print(f"  - Nodes: {G.number_of_nodes()}")
    print(f"  - Edges: {G.number_of_edges()}")
    print(f"\nFiles saved to: {DATA_PATH.resolve()}")


# ============================================================================
# Helper functions for printing and graph export
# ============================================================================

def print_sample_entries(mapping, n=5):
    """Print sample entries from affiliation mapping."""
    print("\nSample entries:")
    for i, (char, data) in enumerate(list(mapping.items())[:n]):
        print(f"\n  {char}:")
        print(f"    Primary: {data['primary_affiliation']}")
        race = data.get('race')
        if race:
            print(f"    Race: {race}")
        affils = ', '.join(data['all_affiliations'][:5])
        print(f"    All affiliations ({data['affiliation_count']}): {affils}")
        if data['affiliation_count'] > 5:
            print(f"      ... and {data['affiliation_count'] - 5} more")


def print_sample_characters(entries, n=5):
    """Print sample character entries."""
    print("\nSample entries:")
    for entry in entries[:n]:
        portals_str = ', '.join(entry.get('portals', [])) or 'None'
        race_str = f", race: {entry.get('race')}" if entry.get('race') else ""
        print(f"  {entry['name']}: {entry['affiliation']}{race_str} "
              f"({entry['affiliation_count']} affiliations, portals: {portals_str})")


def print_sample_edges(edges, n=10):
    """Print sample edges."""
    print("\nSample edges:")
    for edge in edges[:n]:
        print(f"  {edge['source']} -> {edge['target']}")


def print_network_statistics(G, char_dict):
    """Print network statistics."""
    print("\n" + "="*60)
    print("NETWORK STATISTICS")
    print("="*60)
    
    print(f"\nBasic Properties:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.6f}")
    
    if G.number_of_nodes() > 0:
        # Connected components
        wcc = list(nx.weakly_connected_components(G))
        print(f"\nConnected Components:")
        print(f"  Weakly connected components: {len(wcc)}")
        if wcc:
            largest_wcc = max(wcc, key=len)
            print(f"  Largest component size: {len(largest_wcc)} nodes "
                  f"({100*len(largest_wcc)/G.number_of_nodes():.2f}%)")
        
        scc = list(nx.strongly_connected_components(G))
        print(f"  Strongly connected components: {len(scc)}")
        if scc:
            largest_scc = max(scc, key=len)
            print(f"  Largest strongly connected component: {len(largest_scc)} nodes")
        
        # Degree statistics
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        total_degrees = [d for n, d in G.degree()]
        
        print(f"\nDegree Statistics:")
        print(f"  Average in-degree: {np.mean(in_degrees):.2f}")
        print(f"  Average out-degree: {np.mean(out_degrees):.2f}")
        print(f"  Average total degree: {np.mean(total_degrees):.2f}")
        print(f"  Max in-degree: {max(in_degrees) if in_degrees else 0}")
        print(f"  Max out-degree: {max(out_degrees) if out_degrees else 0}")
        
        # Top characters by degree
        degree_centrality = nx.degree_centrality(G)
        top_10 = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print(f"\nTop 10 Characters by Degree Centrality:")
        for i, (node, centrality) in enumerate(top_10, 1):
            char_info = char_dict.get(node, {})
            name = char_info.get("name", node)
            affiliation = char_info.get("affiliation", "Unknown")
            print(f"  {i:2d}. {name} ({affiliation}): "
                  f"degree={G.degree(node)}, centrality={centrality:.4f}")
        
        # Affiliation distribution
        affiliations = [G.nodes[n].get("affiliation") for n in G.nodes() 
                        if G.nodes[n].get("affiliation")]
        affil_counts = Counter(affiliations)
        
        print(f"\nTop 10 Affiliations (by node count):")
        for affil, count in affil_counts.most_common(10):
            print(f"  {affil}: {count} characters")


def prepare_graph_for_export(G):
    """Prepare graph for export by converting lists to strings."""
    G_export = G.copy()
    
    for node in G_export.nodes():
        node_data = G_export.nodes[node]
        
        # Convert all_affiliations list to string
        if 'all_affiliations' in node_data:
            if isinstance(node_data['all_affiliations'], list):
                node_data['all_affiliations'] = ', '.join(
                    str(x) for x in node_data['all_affiliations'] if x is not None
                )
            elif node_data['all_affiliations'] is None:
                node_data['all_affiliations'] = ''
        
        # Remove portal_list (keep only portals string)
        if 'portal_list' in node_data:
            del node_data['portal_list']
        
        # Convert None and non-serializable types
        for key, value in list(node_data.items()):
            if value is None:
                node_data[key] = ''
            elif isinstance(value, (list, tuple, dict)):
                node_data[key] = str(value)
    
    # Convert edge attributes
    for u, v, edge_data in G_export.edges(data=True):
        for key, value in list(edge_data.items()):
            if value is None:
                edge_data[key] = ''
            elif isinstance(value, (list, tuple, dict)):
                edge_data[key] = str(value)
    
    return G_export


def save_gexf(G_export, path):
    """Save graph as GEXF format."""
    try:
        if path.exists():
            path.unlink()
        nx.write_gexf(G_export, path)
        if path.exists() and path.stat().st_size > 0:
            print(f"Saved graph to {path} ({path.stat().st_size:,} bytes)")
        else:
            print(f"Warning: GEXF file was created but appears empty")
    except Exception as e:
        print(f"Warning: Could not save GEXF format: {e}")
        print(f"  Error type: {type(e).__name__}")


# ============================================================================
# Main execution
# ============================================================================

def main():
    """Main execution function."""
    # Step 1: Build affiliation mapping
    char_affil_mapping = build_affiliation_mapping()
    
    # Step 2: Create character entries
    characters = create_character_entries(char_affil_mapping)
    
    # Step 3: Extract edges
    edges = extract_edges(characters)
    
    # Step 4: Build network
    G = build_network(edges, characters)
    
    # Step 5: Save graph
    save_graph(G)


if __name__ == "__main__":
    main()
