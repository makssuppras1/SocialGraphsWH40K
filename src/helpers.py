import re
import json
import networkx as nx
import pandas as pd
from pathlib import Path


MAIN_PORTALS = {
    "Adeptus Custodes", "Mechanicum", "Astra Militarum", "Chaos",
    "Drukhari", "Asuryani", "Genestealer Cults", "Imperium",
    "Inquisition", "Leagues of Votann", "Adepta Sororitas", "Space Marines",
    "Necrons", "Orks", "T'au", "Tyranids"
}

GENERAL_RACES = {
    "Human", "Ork", "Eldar", "Tau", "Necron", "Tyranid", "Chaos Human",
    "Abhumans and Mutants", "Blank", "Daemon"
}

CHARACTER_EXCLUDE_PATTERNS = [
    r'^list of', r'^known members of', r'^known captains of',
    r'^members of', r'^characters of', r'formation', r'^category:',
    r'\(list\)$', r'\(list\)'
]

def normalize_portal_name(portal_name):
    if not portal_name:
        return portal_name

    patterns = [
        (r'SpaceMarines$', 'SpaceMarine'),
        (r'DarkAngels$', 'DarkAngel'),
        (r'ThousandSons$', 'ThousandSon'),
        (r'EmperorsChildren$', 'EmperorsChild'),
        (r'ChaosKnights$', 'ChaosKnight'),
        (r'BlackTemplars$', 'BlackTemplar'),
        (r'FleshTearers$', 'FleshTearer'),
        (r'IronHands$', 'IronHand'),
        (r'IronWarriors$', 'IronWarrior'),
        (r'BloodAngels$', 'BloodAngel'),
        (r'(\w+)Marines$', r'\1Marine'),
        (r'(\w+)Angels$', r'\1Angel'),
        (r'(\w+)Sons$', r'\1Son'),
        (r'(\w+)Knights$', r'\1Knight'),
    ]
    
    for pattern, replacement in patterns:
        if re.search(pattern, portal_name):
            return re.sub(pattern, replacement, portal_name)
    return portal_name

def camel_case_to_spaced(name):
    # Convert CamelCase to "Camel Case" by adding spaces before capital letters
    # (?<!^) means "not at the start", (?=[A-Z]) means "followed by uppercase"
    result = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)
    return result.strip()

def _generate_name_variations(name):
    # Generate different ways the name might appear (with/without spaces, underscores, etc.)
    spaced = camel_case_to_spaced(name)
    variations = [
        name,
        spaced,
        name.replace("_", " "),
        name.replace(" ", "_"),
        name.replace("_", ""),
        name.replace(" ", ""),
        spaced.replace(" ", "_"),
        spaced.replace(" ", "")
    ]
    return variations

def load_portal_mapping(portal_mapping_file):
    if not portal_mapping_file.exists():
        return {}
    with open(portal_mapping_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_main_portal(portal_name):
    if not portal_name:
        return False
    
    portal_lower = portal_name.lower()
    spaced_lower = camel_case_to_spaced(portal_name).lower()
    
    for main_portal in MAIN_PORTALS:
        if main_portal.lower() == portal_lower or main_portal.lower() == spaced_lower:
            return True
    return False


def match_portal_to_key(portal_name, portal_mapping):
    # Try to find a matching key in portal_mapping for the given portal name
    if not portal_name or not portal_mapping:
        return None
    
    # First, try exact matches with name variations
    variations = _generate_name_variations(portal_name)
    for variation in variations:
        if variation in portal_mapping:
            return variation
    
    # Case-insensitive fallback
    portal_lower = portal_name.lower()
    spaced_lower = camel_case_to_spaced(portal_name).lower()
    for key in portal_mapping.keys():
        key_lower = key.lower()
        if key_lower == portal_lower or key_lower == spaced_lower:
            return key
    
    # Last resort: normalize plural forms (SpaceMarines -> SpaceMarine)
    normalized = normalize_portal_name(portal_name)
    if normalized and normalized != portal_name:
        normalized_variations = _generate_name_variations(normalized)
        for variation in normalized_variations:
            if variation in portal_mapping:
                return variation
    
    return None


def map_portal_to_main_portals(portal_name, portal_mapping):
    if not portal_name:
        return None
    
    if is_main_portal(portal_name):
        spaced = camel_case_to_spaced(portal_name)
        for main_portal in MAIN_PORTALS:
            if main_portal.lower() in (portal_name.lower(), spaced.lower()):
                return main_portal
        return portal_name
    
    if not portal_mapping:
        return None
    
    matched_key = match_portal_to_key(portal_name, portal_mapping)
    if not matched_key:
        return None
    
    main_portals = portal_mapping[matched_key]
    if not main_portals:
        return None
    
    if isinstance(main_portals, list):
        return main_portals[-1]
    else:
        return main_portals


def extract_portals(wikitext, portal_mapping=None):
    # Extract portal names from wiki text (e.g., {{SpaceMarinePortal}})
    if not wikitext:
        return []
    
    # Find all portal templates like {{SpaceMarinePortal}}
    portal_pattern = r'\{\{([A-Za-z0-9_]+Portal)\}\}'
    portals = re.findall(portal_pattern, wikitext)
    
    # Clean up portal names (remove "Portal" suffix and normalize)
    cleaned_portals = []
    for portal in portals:
        cleaned = portal.replace('Portal', '').strip()
        if cleaned:
            normalized = normalize_portal_name(cleaned)
            if normalized:
                cleaned_portals.append(normalized)
    
    # If no mapping provided, return the last unique portal
    if not portal_mapping:
        unique_portals = list(dict.fromkeys(cleaned_portals))
        if unique_portals:
            return [unique_portals[-1]]
        else:
            return []
    
    # Map portals to main portals using the mapping
    main_portals = []
    for portal in cleaned_portals:
        main_portal = map_portal_to_main_portals(portal, portal_mapping)
        if main_portal:
            main_portals.append(main_portal)
    
    # Return the last unique main portal
    unique_portals = list(dict.fromkeys(main_portals))
    if unique_portals:
        return [unique_portals[-1]]
    else:
        return []


def is_character(title):
    title_lower = title.lower()
    for pattern in CHARACTER_EXCLUDE_PATTERNS:
        if re.search(pattern, title_lower):
            return False
    return True


def extract_race(affiliations):
    if not affiliations:
        return None
    for aff in affiliations:
        if aff in GENERAL_RACES:
            return aff
    return None


def determine_primary_affiliation(affiliations):
    if not affiliations:
        return None
    specific = []
    for aff in affiliations:
        if aff not in GENERAL_RACES:
            specific.append(aff)
    if specific:
        return specific[0]
    else:
        return None


def load_batch_files(raw_data_path, batch_pattern):
    batch_files = sorted(raw_data_path.glob(batch_pattern))
    all_pages = {}
    
    for batch_file in batch_files:
        with open(batch_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'pages' in data:
                all_pages.update(data['pages'])
    
    return all_pages, batch_files


def create_character_title_map(characters):
    character_titles = set()
    character_title_map = {}
    
    # Build mapping of all name variations to canonical title
    for char in characters:
        title = char["title"]
        name = char["name"]
        file_name = char["file_name"]
        
        # Add all variations to the set
        character_titles.add(title)
        character_titles.add(name)
        character_titles.add(file_name)
        
        # Map everything back to the canonical title
        character_title_map[title] = title
        character_title_map[name] = title
        character_title_map[file_name] = title
        character_title_map[title.replace("_", " ")] = title
        character_title_map[name.replace(" ", "_")] = title
    
    return character_titles, character_title_map


def find_character_title(link, character_title_map):
    # Try exact match first
    if link in character_title_map:
        return character_title_map[link]
    
    # Try with underscores instead of spaces
    variation1 = link.replace(" ", "_")
    if variation1 in character_title_map:
        return character_title_map[variation1]
    
    # Try with spaces instead of underscores
    variation2 = link.replace("_", " ")
    if variation2 in character_title_map:
        return character_title_map[variation2]
    
    # No match found
    return None


def assign_portal_from_affiliations(affiliation, all_affiliations, portal_mapping):
    # Try to assign a portal based on character affiliations
    if not portal_mapping:
        return []
    
    # First try the primary affiliation
    if affiliation:
        matched_key = match_portal_to_key(affiliation, portal_mapping)
        if matched_key:
            main_portals = portal_mapping[matched_key]
            if main_portals:
                # If it's a list, take the last one (most specific)
                if isinstance(main_portals, list):
                    portal = main_portals[-1]
                else:
                    portal = main_portals
                return [portal]
    
    # If that didn't work, try all affiliations
    if all_affiliations:
        for affil in all_affiliations:
            matched_key = match_portal_to_key(affil, portal_mapping)
            if matched_key:
                main_portals = portal_mapping[matched_key]
                if main_portals:
                    if isinstance(main_portals, list):
                        portal = main_portals[-1]
                    else:
                        portal = main_portals
                    return [portal]
    
    return []


def prepare_graph_for_export(G):
    # Prepare graph for export by converting complex types to strings
    # GEXF format doesn't handle lists/dicts well, so we convert them
    G_export = G.copy()
    
    # Process node attributes
    for node in G_export.nodes():
        node_data = G_export.nodes[node]
        
        # Convert all_affiliations list to comma-separated string
        if 'all_affiliations' in node_data:
            if isinstance(node_data['all_affiliations'], list):
                affiliation_parts = []
                for x in node_data['all_affiliations']:
                    if x is not None:
                        affiliation_parts.append(str(x))
                node_data['all_affiliations'] = ', '.join(affiliation_parts)
            elif node_data['all_affiliations'] is None:
                node_data['all_affiliations'] = ''
        
        # Remove portal_list (we keep portals as string)
        if 'portal_list' in node_data:
            del node_data['portal_list']
        
        # Convert None and complex types to strings
        for key, value in list(node_data.items()):
            if value is None:
                node_data[key] = ''
            elif isinstance(value, (list, tuple, dict)):
                node_data[key] = str(value)
    
    # Process edge attributes
    for u, v, edge_data in G_export.edges(data=True):
        for key, value in list(edge_data.items()):
            if value is None:
                edge_data[key] = ''
            elif isinstance(value, (list, tuple, dict)):
                edge_data[key] = str(value)
    
    return G_export


def add_semantic_cluster_and_betweenness(G, edges_file, clusters_file, data_path):
    with open(clusters_file, 'r', encoding='utf-8') as f:
        clusters_data = json.load(f)
    
    node_to_cluster = {}
    for node_title, cluster_info in clusters_data.get('assignments', {}).items():
        node_to_cluster[node_title] = cluster_info.get('semantic_cluster_id', -1)
    
    valid_nodes = set(G.nodes())
    with open(edges_file, 'r', encoding='utf-8') as f:
        edges_data = json.load(f)
    
    G_directed = nx.DiGraph()
    for edge in edges_data:
        source, target = edge['source'], edge['target']
        if source in valid_nodes and target in valid_nodes:
            G_directed.add_edge(source, target)
    
    betweenness = nx.betweenness_centrality(G_directed)
    
    for node in G.nodes():
        G.nodes[node]['semantic_cluster_id'] = node_to_cluster.get(node, -1)
        G.nodes[node]['betweenness'] = betweenness.get(node, 0.0)


def parse_character_affiliations(char_categories):
    character_affiliations = {}
    for category, characters in char_categories.items():
        match = re.match(r"Characters\s*\(([^)]+)\)", category)
        if match:
            affil_name = match.group(1).strip()
            for char in characters:
                if char not in character_affiliations:
                    character_affiliations[char] = set()
                character_affiliations[char].add(affil_name)
    return character_affiliations


def build_affiliation_dict(character_affiliations):
    character_affil_mapping = {}
    for char, affil_set in character_affiliations.items():
        character_affil_mapping[char] = {
            "primary_affiliation": determine_primary_affiliation(affil_set),
            "race": extract_race(affil_set),
            "all_affiliations": sorted(list(affil_set)),
            "affiliation_count": len(affil_set)
        }
    return character_affil_mapping

def get_display_name(title):
    # Remove parenthetical info from title (e.g., "Character (Primarch)" -> "Character")
    display_name = re.sub(r"\s*\([^)]*\)$", "", title).strip()
    return display_name

def filter_missing_values(df, columns):
    # Remove rows where any of the specified columns have missing values
    # Returns a new dataframe with only valid rows
    valid_indices = []
    for idx in df.index:
        row = df.loc[idx]
        all_valid = True
        for col in columns:
            if pd.isna(row[col]):
                all_valid = False
                break
        if all_valid:
            valid_indices.append(idx)
    return df.loc[valid_indices]

def calculate_percentile(values, percentile):
    # Calculate percentile manually (e.g., 0.90 for 90th percentile)
    sorted_values = values.copy()
    sorted_values.sort()
    index = int(len(sorted_values) * percentile)
    if index >= len(sorted_values):
        index = len(sorted_values) - 1
    return sorted_values[index]

def simple_sort_descending(items, key_index=None):
    # Simple sort descending - not efficient but easy to understand
    # If key_index is None, items should be comparable directly
    # If key_index is provided, items are tuples and we sort by that index
    sorted_items = items.copy()
    for i in range(len(sorted_items)):
        for j in range(i + 1, len(sorted_items)):
            if key_index is not None:
                val_i = sorted_items[i][key_index]
                val_j = sorted_items[j][key_index]
            else:
                val_i = sorted_items[i]
                val_j = sorted_items[j]
            
            if val_i < val_j:
                temp = sorted_items[i]
                sorted_items[i] = sorted_items[j]
                sorted_items[j] = temp
    return sorted_items
