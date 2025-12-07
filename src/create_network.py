#!/usr/bin/env python3
# Creates Warhammer 40K character network from Lexicanum data

import json
import re
import pickle
import networkx as nx

from config import (
    CHAR_CATEGORIES_FILE, AFFILIATION_MAPPING_FILE,
    CHARACTERS_FILE, EDGES_FILE,
    GEXF_FILE, PICKLE_FILE,
    GEXF_FILTERED_FILE, PICKLE_FILTERED_FILE,
    PORTAL_MAPPING_FILE, DATA_PATH
)
from helpers import (
    extract_portals, is_character, determine_primary_affiliation, extract_race,
    load_batch_files, create_character_title_map, find_character_title,
    load_portal_mapping, match_portal_to_key, assign_portal_from_affiliations,
    prepare_graph_for_export, parse_character_affiliations, build_affiliation_dict,
    add_semantic_cluster_and_betweenness, get_display_name
)
from config import RAW_DATA_PATH, BATCH_FILE_PATTERN


def build_affiliation_mapping():
    with open(CHAR_CATEGORIES_FILE, 'r', encoding='utf-8') as f:
        char_categories = json.load(f)
    
    character_affiliations = parse_character_affiliations(char_categories)
    character_affil_mapping = build_affiliation_dict(character_affiliations)
    
    with open(AFFILIATION_MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(character_affil_mapping, f, indent=2, ensure_ascii=False)
    
    return character_affil_mapping


def create_character_entries(char_affil_mapping):
    portal_mapping = load_portal_mapping(PORTAL_MAPPING_FILE)
    all_pages, _ = load_batch_files(RAW_DATA_PATH, BATCH_FILE_PATTERN)
    
    entries = []
    for title, page_data in sorted(all_pages.items()):
        if not is_character(title):
            continue
        
        display_name = get_display_name(title)
        file_name = title.replace("/", "_").replace("\\", "_")
        wikitext = page_data.get("wikitext", "")
        portals = extract_portals(wikitext, portal_mapping)
        
        # Try to get affiliation data - check both title and display name
        affil_data = None
        if title in char_affil_mapping:
            affil_data = char_affil_mapping[title]
        elif display_name in char_affil_mapping:
            affil_data = char_affil_mapping[display_name]
        
        if affil_data:
            if "primary_affiliation" in affil_data:
                affiliation = affil_data["primary_affiliation"]
            else:
                affiliation = None
            if "race" in affil_data:
                race = affil_data["race"]
            else:
                race = None
            if "all_affiliations" in affil_data:
                all_affiliations = affil_data["all_affiliations"]
            else:
                all_affiliations = []
            if "affiliation_count" in affil_data:
                affiliation_count = affil_data["affiliation_count"]
            else:
                affiliation_count = 0
        else:
            affiliation = None
            race = None
            all_affiliations = []
            affiliation_count = 0
        
        if not portals:
            portals = assign_portal_from_affiliations(affiliation, all_affiliations, portal_mapping)
        
        entries.append({
            "title": title,
            "name": display_name,
            "file_name": file_name,
            "pageid": page_data.get("pageid") if "pageid" in page_data else None,
            "affiliation": affiliation,
            "race": race,
            "all_affiliations": all_affiliations,
            "affiliation_count": affiliation_count,
            "portals": portals
        })
    
    with open(CHARACTERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    
    return entries


def extract_edges(characters):
    character_titles, character_title_map = create_character_title_map(characters)
    all_pages, _ = load_batch_files(RAW_DATA_PATH, BATCH_FILE_PATTERN)
    
    edges = []
    for source_title, page_data in all_pages.items():
        if source_title not in character_titles or not is_character(source_title):
            continue
        
        # Get wikitext - need this to find links
        if "wikitext" in page_data:
            wikitext = page_data["wikitext"]
        else:
            wikitext = ""
        if not wikitext:
            continue
        
        # Extract wiki links - this regex was a pain to get right
        wiki_links = re.findall(r'\[\[([^\]|#:]+)(?:\|[^\]]+)?\]\]', wikitext)
        for link in wiki_links:
            link = link.strip()
            if ":" in link or "portal" in link.lower():
                continue
    
            target_title = find_character_title(link, character_title_map)
            if target_title and is_character(target_title) and target_title != source_title:
                edges.append({"source": source_title, "target": target_title})
    
    with open(EDGES_FILE, 'w', encoding='utf-8') as f:
        json.dump(edges, f, indent=2, ensure_ascii=False)
    
    return edges


def build_network(edges, characters):
    G = nx.DiGraph()
    for char in characters:
        title = char["title"]
        if not is_character(title):
            continue
        
        # Get portal info
        if "portals" in char:
            portals = char["portals"]
        else:
            portals = []
        
        if portals:
            portals_str = ', '.join(portals)
        else:
            portals_str = ''
        
        # Get other attributes with defaults
        name = char["name"] if "name" in char else ""
        file_name = char["file_name"] if "file_name" in char else ""
        pageid = char["pageid"] if "pageid" in char else None
        affiliation = char["affiliation"] if "affiliation" in char else None
        race = char["race"] if "race" in char else None
        all_affiliations = char["all_affiliations"] if "all_affiliations" in char else []
        affiliation_count = char["affiliation_count"] if "affiliation_count" in char else 0
        
        G.add_node(
            title,
            name=name,
            file_name=file_name,
            pageid=pageid,
            affiliation=affiliation,
            race=race,
            all_affiliations=all_affiliations,
            affiliation_count=affiliation_count,
            portals=portals_str,
            portal_list=portals
        )
    
    # Add edges, counting multiple links as weight
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        if source in G and target in G:
            if G.has_edge(source, target):
                # Increment weight if edge already exists
                if "weight" in G[source][target]:
                    current_weight = G[source][target]["weight"]
                else:
                    current_weight = 1
                G[source][target]["weight"] = current_weight + 1
            else:
                G.add_edge(source, target, weight=1)
    
    return G


def save_graph(G, output_file=PICKLE_FILE, gexf_file=GEXF_FILE):
    G_export = prepare_graph_for_export(G)
    
    if gexf_file.exists():
        gexf_file.unlink()
    nx.write_gexf(G_export, gexf_file)
    
    with open(output_file, 'wb') as f:
        pickle.dump(G, f)
    print(f"Network saved: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


def filter_network(G):
    components = list(nx.weakly_connected_components(G))
    if not components:
        raise ValueError("No connected components found")
    
    largest_component = max(components, key=len)
    G_filtered = G.subgraph(largest_component).copy()
    
    # Remove nodes with degree < 2
    # TODO: could optimize but well see
    while True:
        nodes_to_remove = []
        for n in G_filtered.nodes():
            total_degree = G_filtered.in_degree(n) + G_filtered.out_degree(n)
            if total_degree < 2:
                nodes_to_remove.append(n)
        if not nodes_to_remove:
            break
        G_filtered.remove_nodes_from(nodes_to_remove)
    
    return G_filtered


def save_filtered_network(G_filtered):
    print("Computing Louvain communities for filtered network")
    G_undirected = G_filtered.to_undirected()
    partition = nx.community.louvain_communities(G_undirected)
    
    node_to_community = {}
    comm_id = 0
    for community in partition:
        for node in community:
            node_to_community[node] = comm_id
        comm_id = comm_id + 1
    
    for node in G_filtered.nodes():
        if node in node_to_community:
            G_filtered.nodes[node]['community_id'] = node_to_community[node]
        else:
            G_filtered.nodes[node]['community_id'] = -1
    
    print(f"Found {len(partition)} communities")
    
    clusters_file = DATA_PATH / "optimal_semantic_clusters.json"
    if clusters_file.exists():
        add_semantic_cluster_and_betweenness(G_filtered, EDGES_FILE, clusters_file, DATA_PATH)
    
    save_graph(G_filtered, PICKLE_FILTERED_FILE, GEXF_FILTERED_FILE)
    
    semantic_gexf_file = DATA_PATH / "semantic_network.gexf"
    save_graph(G_filtered, PICKLE_FILTERED_FILE, semantic_gexf_file)
    
    print(f"Filtered network: {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")


def main(create_filtered=True):
    char_affil_mapping = build_affiliation_mapping()
    print(f"Mapped {len(char_affil_mapping)} characters")
    print()
    
    characters = create_character_entries(char_affil_mapping)
    print(f"Created {len(characters)} character entries")
    print()
    
    edges = extract_edges(characters)
    print(f"Network has {len(edges)} edges")
    print()
    
    G = build_network(edges, characters)
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print()
    
    print("Saving network")
    save_graph(G, PICKLE_FILE, GEXF_FILE)
    print()
    
    # Filter to largest component - makes analysis cleaner
    if create_filtered:
        G_filtered = filter_network(G)
        save_filtered_network(G_filtered)
        print()
    
    print("Done")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create Warhammer 40K character network')
    parser.add_argument('--no-filtered', action='store_true',
                       help='Skip creating filtered network')
    args = parser.parse_args()
    
    main(create_filtered=not args.no_filtered)
