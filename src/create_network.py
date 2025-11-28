#!/usr/bin/env python3
# creates warhammer 40k character network from lexicanum data
# this script:
# 1. builds character-to-affiliation mapping from category data
# 2. creates character entries with affiliations and portals
# 3. extracts edges from wikitext links
# 4. builds networkx graph with all attributes
# 5. saves graph in multiple formats (gexf, pickle)

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
    # step 1: builds character-to-affiliation mapping from category data
    print("="*60)
    print("STEP 1: Building Character-to-Affiliation Mapping")
    print("="*60)
    
    # load character categories
    with open(CHAR_CATEGORIES_FILE, 'r', encoding='utf-8') as f:
        char_categories = json.load(f)
    
    print(f"Loaded {len(char_categories)} categories from Lexicanum data")
    
    # map characters to their affiliations
    character_affiliations = {}
    for category, characters in char_categories.items():
        match = re.match(r"Characters\s*\(([^)]+)\)", category)
        if match:
            affil_name = match.group(1).strip()
            for char in characters:
                if char not in character_affiliations:
                    character_affiliations[char] = set()
                character_affiliations[char].add(affil_name)
    
    # count unique affiliations
    all_affils = set()
    for affils in character_affiliations.values():
        for affil in affils:
            all_affils.add(affil)
    
    print(f"Found {len(all_affils)} unique character affiliations")
    print(f"Mapped {len(character_affiliations)} characters to affiliations")
    
    # create structured output
    character_affil_mapping = {}
    for char, affil_set in character_affiliations.items():
        affil_list = sorted(list(affil_set))
        character_affil_mapping[char] = {
            "primary_affiliation": determine_primary_affiliation(affil_set),
            "race": extract_race(affil_set),
            "all_affiliations": affil_list,
            "affiliation_count": len(affil_list)
        }
    
    # save mapping
    with open(AFFILIATION_MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(character_affil_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved character-to-affiliation mapping to {AFFILIATION_MAPPING_FILE}")
    print_sample_entries(character_affil_mapping, 5)
    
    return character_affil_mapping


def create_character_entries(char_affil_mapping):
    # step 2: creates character entries with affiliations and portals
    print("\n" + "="*60)
    print("STEP 2: Creating Character Entries with Affiliations and Portals")
    print("="*60)
    
    print(f"Loaded affiliation mapping for {len(char_affil_mapping)} characters")
    
    # load portal mapping to map sub-portals to 16 main portals
    portal_mapping = load_portal_mapping(PORTAL_MAPPING_FILE)
    if portal_mapping:
        print(f"Loaded portal mapping with {len(portal_mapping)} entries")
    else:
        print("Warning: Portal mapping file not found, using raw portal extraction")
    
    # load all pages from batch files
    all_pages, batch_files = load_batch_files(RAW_DATA_PATH, BATCH_FILE_PATTERN)
    print(f"Found {len(batch_files)} batch files")
    print(f"Total pages collected: {len(all_pages)}")
    
    # create character entries
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
        # extract portals and map them to main portals using the comprehensive mapping
        portals = extract_portals(wikitext, portal_mapping)
        
        # look up affiliations
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
        
        # if no portal found in wikitext, try to assign from affiliations
        if not portals and portal_mapping:
            assigned_portal = None
            
            # first try primary affiliation
            if affiliation:
                matched_key = match_portal_to_key(affiliation, portal_mapping)
                if matched_key:
                    main_portals = portal_mapping[matched_key]
                    if main_portals:
                        if isinstance(main_portals, list):
                            assigned_portal = main_portals[-1]
                        else:
                            assigned_portal = main_portals
            
            # if no match, try all affiliations
            if not assigned_portal and all_affiliations:
                for affil in all_affiliations:
                    matched_key = match_portal_to_key(affil, portal_mapping)
                    if matched_key:
                        main_portals = portal_mapping[matched_key]
                        if main_portals:
                            if isinstance(main_portals, list):
                                assigned_portal = main_portals[-1]
                            else:
                                assigned_portal = main_portals
                            break  # use first match
            
            if assigned_portal:
                portals = [assigned_portal]
        
        # track portal statistics
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
    
    # save characters
    with open(CHARACTERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    
    print(f"\nStored {len(entries)} characters in {CHARACTERS_FILE}")
    print(f"Characters with affiliations: {stats['with_affil']}")
    print(f"Characters without affiliations: {stats['without_affil']}")
    if len(entries) > 0:
        print(f"Characters with portals: {stats['with_portal']} ({100*stats['with_portal']/len(entries):.1f}%)")
        print(f"Characters without portals: {stats['without_portal']} ({100*stats['without_portal']/len(entries):.1f}%)")
    print(f"Non-character pages skipped: {stats['skipped']}")
    
    # show portal distribution
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
    # step 3: extracts edges from wikitext links
    print("\n" + "="*60)
    print("STEP 3: Extracting Edges from Wikitext")
    print("="*60)
    
    print(f"Loaded {len(characters)} characters")
    
    # create character title mapping
    character_titles, character_title_map = create_character_title_map(characters)
    print(f"Created lookup set with {len(character_titles)} variations")
    
    # load all pages
    all_pages, _ = load_batch_files(RAW_DATA_PATH, BATCH_FILE_PATTERN)
    print(f"Processing {len(all_pages)} pages to extract edges...")
    
    # extract edges
    edges = []
    characters_with_links = 0
    
    for source_title, page_data in all_pages.items():
        if source_title not in character_titles or not is_character(source_title):
            continue
        
        wikitext = page_data.get("wikitext", "")
        if not wikitext:
            continue
        
        # extract wiki links
        wiki_links = re.findall(r'\[\[([^\]|#:]+)(?:\|[^\]]+)?\]\]', wikitext)
        
        found_links = False
        for link in wiki_links:
            link = link.strip()
            
            # skip special namespaces and portal links
            if ":" in link or "portal" in link.lower():
                continue
            
            # find character title
            target_title = find_character_title(link, character_title_map)
            if not target_title or not is_character(target_title):
                continue
            
            # skip self-loops
            if target_title == source_title:
                continue
            
            edges.append({"source": source_title, "target": target_title})
            found_links = True
        
        if found_links:
            characters_with_links += 1
    
    # save edges
    with open(EDGES_FILE, 'w', encoding='utf-8') as f:
        json.dump(edges, f, indent=2, ensure_ascii=False)
    
    print(f"\nExtracted {len(edges)} edges")
    print(f"Characters with outgoing links: {characters_with_links}")
    print(f"Saved edges to {EDGES_FILE}")
    print_sample_edges(edges, 10)
    
    return edges


def build_network(edges, characters):
    # step 4: builds networkx graph with all attributes
    print("\n" + "="*60)
    print("STEP 4: Building NetworkX Graph")
    print("="*60)
    
    print(f"Loaded {len(edges)} edges from file")
    print(f"Loaded {len(characters)} characters")
    
    # create character lookup
    char_dict = {}
    for char in characters:
        char_dict[char["title"]] = char
    
    # build directed graph
    G = nx.DiGraph()
    
    # add nodes
    for char in characters:
        title = char["title"]
        if not is_character(title):
            continue
        
        portals = char.get("portals", [])
        if portals:
            portals_str = ', '.join(portals)
        else:
            portals_str = ''
        
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
    
    # add edges
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        if source in G and target in G:
            if G.has_edge(source, target):
                G[source][target]["weight"] += 1
            else:
                G.add_edge(source, target, weight=1)
    
    print(f"Added {G.number_of_edges()} edges to graph")
    
    # print statistics
    print_network_statistics(G, char_dict)
    
    return G


def save_graph(G):
    # step 5: saves graph in multiple formats
    print("\n" + "="*60)
    print("STEP 5: Saving Graph in Multiple Formats")
    print("="*60)
    
    # create export copy (convert lists to strings)
    G_export = prepare_graph_for_export(G)
    
    # save gexf
    save_gexf(G_export, GEXF_FILE)
    
    # save pickle (preserves all data types)
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


# helper functions for printing and graph export

def print_sample_entries(mapping, n=5):
    # prints sample entries from affiliation mapping
    print("\nSample entries:")
    items_list = list(mapping.items())[:n]
    for i, (char, data) in enumerate(items_list):
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
    # prints sample character entries
    print("\nSample entries:")
    for entry in entries[:n]:
        portals = entry.get('portals', [])
        if portals:
            portals_str = ', '.join(portals)
        else:
            portals_str = 'None'
        race = entry.get('race')
        if race:
            race_str = f", race: {race}"
        else:
            race_str = ""
        print(f"  {entry['name']}: {entry['affiliation']}{race_str} "
              f"({entry['affiliation_count']} affiliations, portals: {portals_str})")


def print_sample_edges(edges, n=10):
    # prints sample edges
    print("\nSample edges:")
    for edge in edges[:n]:
        print(f"  {edge['source']} -> {edge['target']}")


def print_network_statistics(G, char_dict):
    # prints network statistics
    print("\n" + "="*60)
    print("NETWORK STATISTICS")
    print("="*60)
    
    print(f"\nBasic Properties:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.6f}")
    
    if G.number_of_nodes() > 0:
        # connected components
        wcc = list(nx.weakly_connected_components(G))
        print(f"\nConnected Components:")
        print(f"  Weakly connected components: {len(wcc)}")
        if wcc:
            largest_wcc = max(wcc, key=len)
            pct = 100*len(largest_wcc)/G.number_of_nodes()
            print(f"  Largest component size: {len(largest_wcc)} nodes ({pct:.2f}%)")
        
        scc = list(nx.strongly_connected_components(G))
        print(f"  Strongly connected components: {len(scc)}")
        if scc:
            largest_scc = max(scc, key=len)
            print(f"  Largest strongly connected component: {len(largest_scc)} nodes")
        
        # degree statistics
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        total_degrees = [d for n, d in G.degree()]
        
        print(f"\nDegree Statistics:")
        print(f"  Average in-degree: {np.mean(in_degrees):.2f}")
        print(f"  Average out-degree: {np.mean(out_degrees):.2f}")
        print(f"  Average total degree: {np.mean(total_degrees):.2f}")
        if in_degrees:
            print(f"  Max in-degree: {max(in_degrees)}")
        else:
            print(f"  Max in-degree: 0")
        if out_degrees:
            print(f"  Max out-degree: {max(out_degrees)}")
        else:
            print(f"  Max out-degree: 0")
        
        # top characters by degree
        degree_centrality = nx.degree_centrality(G)
        top_10 = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print(f"\nTop 10 Characters by Degree Centrality:")
        for i, (node, centrality) in enumerate(top_10, 1):
            char_info = char_dict.get(node, {})
            name = char_info.get("name", node)
            affiliation = char_info.get("affiliation", "Unknown")
            print(f"  {i:2d}. {name} ({affiliation}): "
                  f"degree={G.degree(node)}, centrality={centrality:.4f}")
        
        # affiliation distribution
        affiliations = []
        for n in G.nodes():
            affil = G.nodes[n].get("affiliation")
            if affil:
                affiliations.append(affil)
        affil_counts = Counter(affiliations)
        
        print(f"\nTop 10 Affiliations (by node count):")
        for affil, count in affil_counts.most_common(10):
            print(f"  {affil}: {count} characters")


def prepare_graph_for_export(G):
    # prepares graph for export by converting lists to strings
    G_export = G.copy()
    
    for node in G_export.nodes():
        node_data = G_export.nodes[node]
        
        # convert all_affiliations list to string
        if 'all_affiliations' in node_data:
            if isinstance(node_data['all_affiliations'], list):
                affil_strs = []
                for x in node_data['all_affiliations']:
                    if x is not None:
                        affil_strs.append(str(x))
                node_data['all_affiliations'] = ', '.join(affil_strs)
            elif node_data['all_affiliations'] is None:
                node_data['all_affiliations'] = ''
        
        # remove portal_list (keep only portals string)
        if 'portal_list' in node_data:
            del node_data['portal_list']
        
        # convert None and non-serializable types
        for key, value in list(node_data.items()):
            if value is None:
                node_data[key] = ''
            elif isinstance(value, (list, tuple, dict)):
                node_data[key] = str(value)
    
    # convert edge attributes
    for u, v, edge_data in G_export.edges(data=True):
        for key, value in list(edge_data.items()):
            if value is None:
                edge_data[key] = ''
            elif isinstance(value, (list, tuple, dict)):
                edge_data[key] = str(value)
    
    return G_export


def save_gexf(G_export, path):
    # saves graph as gexf format
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


# main execution

def main():
    # main execution function
    # step 1: build affiliation mapping
    char_affil_mapping = build_affiliation_mapping()
    
    # step 2: create character entries
    characters = create_character_entries(char_affil_mapping)
    
    # step 3: extract edges
    edges = extract_edges(characters)
    
    # step 4: build network
    G = build_network(edges, characters)
    
    # step 5: save graph
    save_graph(G)


if __name__ == "__main__":
    main()
