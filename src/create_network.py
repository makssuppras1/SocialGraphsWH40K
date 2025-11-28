#!/usr/bin/env python3
# creates warhammer 40k character network from lexicanum data

import json
import re
import pickle
import networkx as nx
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
    # builds character-to-affiliation mapping from category data
    with open(CHAR_CATEGORIES_FILE, 'r', encoding='utf-8') as f:
        char_categories = json.load(f)
    
    character_affiliations = {}
    for category, characters in char_categories.items():
        match = re.match(r"Characters\s*\(([^)]+)\)", category)
        if match:
            affil_name = match.group(1).strip()
            for char in characters:
                if char not in character_affiliations:
                    character_affiliations[char] = set()
                character_affiliations[char].add(affil_name)
    
    character_affil_mapping = {}
    for char, affil_set in character_affiliations.items():
        character_affil_mapping[char] = {
            "primary_affiliation": determine_primary_affiliation(affil_set),
            "race": extract_race(affil_set),
            "all_affiliations": sorted(list(affil_set)),
            "affiliation_count": len(affil_set)
        }
    
    with open(AFFILIATION_MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(character_affil_mapping, f, indent=2, ensure_ascii=False)
    
    return character_affil_mapping


def create_character_entries(char_affil_mapping):
    # creates character entries with affiliations and portals
    portal_mapping = load_portal_mapping(PORTAL_MAPPING_FILE)
    all_pages, _ = load_batch_files(RAW_DATA_PATH, BATCH_FILE_PATTERN)
    
    entries = []
    for title, page_data in sorted(all_pages.items()):
        if not is_character(title):
            continue
        
        display_name = re.sub(r"\s*\([^)]*\)$", "", title).strip()
        file_name = title.replace("/", "_").replace("\\", "_")
        wikitext = page_data.get("wikitext", "")
        portals = extract_portals(wikitext, portal_mapping)
        
        affil_data = char_affil_mapping.get(title) or char_affil_mapping.get(display_name)
        if affil_data:
            affiliation = affil_data.get("primary_affiliation")
            race = affil_data.get("race")
            all_affiliations = affil_data.get("all_affiliations", [])
            affiliation_count = affil_data.get("affiliation_count", 0)
        else:
            affiliation = race = None
            all_affiliations = []
            affiliation_count = 0
        
        # assign portal from affiliations if not found in wikitext
        if not portals and portal_mapping:
            if affiliation:
                matched_key = match_portal_to_key(affiliation, portal_mapping)
                if matched_key:
                    main_portals = portal_mapping[matched_key]
                    if main_portals:
                        portals = [main_portals[-1] if isinstance(main_portals, list) else main_portals]
            
            if not portals and all_affiliations:
                for affil in all_affiliations:
                    matched_key = match_portal_to_key(affil, portal_mapping)
                    if matched_key:
                        main_portals = portal_mapping[matched_key]
                        if main_portals:
                            portals = [main_portals[-1] if isinstance(main_portals, list) else main_portals]
                            break
        
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
    
    with open(CHARACTERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    
    return entries


def extract_edges(characters):
    # extracts edges from wikitext links
    character_titles, character_title_map = create_character_title_map(characters)
    all_pages, _ = load_batch_files(RAW_DATA_PATH, BATCH_FILE_PATTERN)
    
    edges = []
    for source_title, page_data in all_pages.items():
        if source_title not in character_titles or not is_character(source_title):
            continue
        
        wikitext = page_data.get("wikitext", "")
        if not wikitext:
            continue
        
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
    # builds networkx graph with all attributes
    G = nx.DiGraph()
    
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
    
    for edge in edges:
        source, target = edge["source"], edge["target"]
        if source in G and target in G:
            if G.has_edge(source, target):
                G[source][target]["weight"] += 1
            else:
                G.add_edge(source, target, weight=1)
    
    return G


def prepare_graph_for_export(G):
    # prepares graph for export by converting lists to strings
    G_export = G.copy()
    
    for node in G_export.nodes():
        node_data = G_export.nodes[node]
        if 'all_affiliations' in node_data and isinstance(node_data['all_affiliations'], list):
            node_data['all_affiliations'] = ', '.join(str(x) for x in node_data['all_affiliations'] if x is not None)
        elif 'all_affiliations' in node_data and node_data['all_affiliations'] is None:
            node_data['all_affiliations'] = ''
        
        if 'portal_list' in node_data:
            del node_data['portal_list']
        
        for key, value in list(node_data.items()):
            if value is None:
                node_data[key] = ''
            elif isinstance(value, (list, tuple, dict)):
                node_data[key] = str(value)
    
    for u, v, edge_data in G_export.edges(data=True):
        for key, value in list(edge_data.items()):
            if value is None:
                edge_data[key] = ''
            elif isinstance(value, (list, tuple, dict)):
                edge_data[key] = str(value)
    
    return G_export


def save_graph(G):
    # saves graph in multiple formats
    G_export = prepare_graph_for_export(G)
    
    try:
        if GEXF_FILE.exists():
            GEXF_FILE.unlink()
        nx.write_gexf(G_export, GEXF_FILE)
    except Exception as e:
        print(f"Warning: Could not save GEXF format: {e}")
    
    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump(G, f)
    
    print(f"Network created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


def main():
    char_affil_mapping = build_affiliation_mapping()
    characters = create_character_entries(char_affil_mapping)
    edges = extract_edges(characters)
    G = build_network(edges, characters)
    save_graph(G)


if __name__ == "__main__":
    main()
