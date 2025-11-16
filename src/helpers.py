"""Helper functions for network creation."""
import re
import json
from pathlib import Path


# Constants
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


# Portal name normalization
def normalize_portal_name(portal_name):
    """Normalize portal names (e.g., SpaceMarines -> SpaceMarine)."""
    if not portal_name:
        return portal_name
    
    # Common plural patterns
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
    """Convert camelCase to spaced format (e.g., 'AstraMilitarum' -> 'Astra Militarum')."""
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', name).strip()


def _generate_name_variations(name):
    """Generate common name variations for matching."""
    spaced = camel_case_to_spaced(name)
    return [
        name, spaced,
        name.replace("_", " "), name.replace(" ", "_"),
        name.replace("_", ""), name.replace(" ", ""),
        spaced.replace(" ", "_"), spaced.replace(" ", "")
    ]


# Portal mapping
def load_portal_mapping(portal_mapping_file):
    """Load portal mapping from JSON file."""
    if not portal_mapping_file.exists():
        return {}
    with open(portal_mapping_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_main_portal(portal_name):
    """Check if portal name is one of the 16 main portals."""
    if not portal_name:
        return False
    
    portal_lower = portal_name.lower()
    spaced_lower = camel_case_to_spaced(portal_name).lower()
    
    return any(mp.lower() == portal_lower or mp.lower() == spaced_lower 
               for mp in MAIN_PORTALS)


def match_portal_to_key(portal_name, portal_mapping):
    """Match portal name to a key in portal mapping using various name variations."""
    if not portal_name or not portal_mapping:
        return None
    
    # Try exact match and variations
    for variation in _generate_name_variations(portal_name):
        if variation in portal_mapping:
            return variation
    
    # Try case-insensitive match
    portal_lower = portal_name.lower()
    spaced_lower = camel_case_to_spaced(portal_name).lower()
    for key in portal_mapping.keys():
        if key.lower() in (portal_lower, spaced_lower):
            return key
    
    # Try normalized variations
    normalized = normalize_portal_name(portal_name)
    if normalized and normalized != portal_name:
        for variation in _generate_name_variations(normalized):
            if variation in portal_mapping:
                return variation
    
    return None


def map_portal_to_main_portals(portal_name, portal_mapping):
    """Map portal name to one of the 16 main portals."""
    if not portal_name:
        return None
    
    # Check if already a main portal
    if is_main_portal(portal_name):
        spaced = camel_case_to_spaced(portal_name)
        for main_portal in MAIN_PORTALS:
            if main_portal.lower() in (portal_name.lower(), spaced.lower()):
                return main_portal
        return portal_name
    
    # Use mapping file
    if not portal_mapping:
        return None
    
    matched_key = match_portal_to_key(portal_name, portal_mapping)
    if not matched_key:
        return None
    
    main_portals = portal_mapping[matched_key]
    if not main_portals:
        return None
    
    # Return last portal if multiple (conflict resolution)
    return main_portals[-1] if isinstance(main_portals, list) else main_portals


def extract_portals(wikitext, portal_mapping=None):
    """Extract portal names from wikitext and map to main portals.
    
    Extracts {{NamePortal}} patterns and maps to 16 main portals.
    Returns list with single portal (last one if multiple found).
    """
    if not wikitext:
        return []
    
    # Extract portal templates
    portal_pattern = r'\{\{([A-Za-z0-9_]+Portal)\}\}'
    portals = re.findall(portal_pattern, wikitext)
    
    # Clean and normalize
    cleaned_portals = []
    for portal in portals:
        cleaned = portal.replace('Portal', '').strip()
        if cleaned:
            normalized = normalize_portal_name(cleaned)
            if normalized:
                cleaned_portals.append(normalized)
    
    # If no mapping, return last unique portal
    if not portal_mapping:
        unique_portals = list(dict.fromkeys(cleaned_portals))  # Preserves order
        return [unique_portals[-1]] if unique_portals else []
    
    # Map to main portals
    main_portals = []
    for portal in cleaned_portals:
        main_portal = map_portal_to_main_portals(portal, portal_mapping)
        if main_portal:
            main_portals.append(main_portal)
    
    # Return last unique portal
    unique_portals = list(dict.fromkeys(main_portals))
    return [unique_portals[-1]] if unique_portals else []


# Character filtering
def is_character(title):
    """Check if title represents a character (not a list, formation, etc.)."""
    title_lower = title.lower()
    return not any(re.search(pattern, title_lower) 
                   for pattern in CHARACTER_EXCLUDE_PATTERNS)


def extract_race(affiliations):
    """Extract race/species from affiliations. Returns first race found or None."""
    if not affiliations:
        return None
    return next((aff for aff in affiliations if aff in GENERAL_RACES), None)


def determine_primary_affiliation(affiliations):
    """Determine primary affiliation, excluding races.
    
    Races are stored separately. Returns first specific affiliation or None.
    """
    if not affiliations:
        return None
    specific = [aff for aff in affiliations if aff not in GENERAL_RACES]
    return specific[0] if specific else None


# File loading
def load_batch_files(raw_data_path, batch_pattern):
    """Load all pages from batch JSON files."""
    batch_files = sorted(raw_data_path.glob(batch_pattern))
    all_pages = {}
    
    for batch_file in batch_files:
        with open(batch_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'pages' in data:
                all_pages.update(data['pages'])
    
    return all_pages, batch_files


# Character title mapping
def create_character_title_map(characters):
    """Create mapping from character title variations to canonical titles."""
    character_titles = set()
    character_title_map = {}
    
    for char in characters:
        title = char["title"]
        name = char["name"]
        file_name = char["file_name"]
        
        character_titles.update([title, name, file_name])
        
        # Map variations to canonical title
        character_title_map.update({
            title: title,
            name: title,
            file_name: title,
            title.replace("_", " "): title,
            name.replace(" ", "_"): title
        })
    
    return character_titles, character_title_map


def find_character_title(link, character_title_map):
    """Find canonical character title for a given link."""
    if link in character_title_map:
        return character_title_map[link]
    
    # Try space/underscore variations
    for variation in [link.replace(" ", "_"), link.replace("_", " ")]:
        if variation in character_title_map:
            return character_title_map[variation]
    
    return None
