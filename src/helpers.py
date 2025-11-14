"""Helper functions for network creation."""
import re
import json
from pathlib import Path


def normalize_portal_name(portal_name):
    """Normalize portal names to handle variations like SpaceMarine/SpaceMarines."""
    if not portal_name:
        return portal_name
    
    # Handle plural/singular variations
    plural_patterns = [
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
        (r'RavenGuard$', 'RavenGuard'),
        # General patterns
        (r'(\w+)Marines$', r'\1Marine'),
        (r'(\w+)Angels$', r'\1Angel'),
        (r'(\w+)Sons$', r'\1Son'),
        (r'(\w+)Knights$', r'\1Knight'),
    ]
    
    normalized = portal_name
    for pattern, replacement in plural_patterns:
        if re.search(pattern, normalized):
            normalized = re.sub(pattern, replacement, normalized)
            break
    
    return normalized


def load_portal_mapping(portal_mapping_file):
    """Load portal mapping from JSON file.
    
    Returns a dictionary mapping portal names to their main portal(s).
    """
    if not portal_mapping_file.exists():
        return {}
    
    with open(portal_mapping_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def camel_case_to_spaced(name):
    """Convert camelCase to spaced format (e.g., 'AstraMilitarum' -> 'Astra Militarum')."""
    # Insert space before capital letters (but not at the start)
    spaced = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)
    return spaced.strip()


def match_portal_to_key(portal_name, portal_mapping):
    """Match a portal name to a key in the portal mapping.
    
    Tries various name variations to find a match.
    Returns the matched key or None.
    """
    if not portal_name or not portal_mapping:
        return None
    
    # Try exact match
    if portal_name in portal_mapping:
        return portal_name
    
    # Try camelCase to spaced conversion (e.g., "AstraMilitarum" -> "Astra Militarum")
    spaced = camel_case_to_spaced(portal_name)
    if spaced in portal_mapping:
        return spaced
    
    # Try with spaces/underscores variations
    variations = [
        portal_name.replace("_", " "),
        portal_name.replace(" ", "_"),
        portal_name.replace("_", ""),
        portal_name.replace(" ", ""),
        spaced.replace(" ", "_"),
        spaced.replace(" ", ""),
    ]
    
    for variation in variations:
        if variation in portal_mapping:
            return variation
    
    # Try case-insensitive match
    portal_lower = portal_name.lower()
    spaced_lower = spaced.lower()
    for key in portal_mapping.keys():
        if key.lower() == portal_lower or key.lower() == spaced_lower:
            return key
    
    # Try normalized variations
    normalized = normalize_portal_name(portal_name)
    if normalized and normalized != portal_name:
        normalized_spaced = camel_case_to_spaced(normalized)
        if normalized in portal_mapping:
            return normalized
        if normalized_spaced in portal_mapping:
            return normalized_spaced
        for variation in [
            normalized.replace("_", " "), 
            normalized.replace(" ", "_"),
            normalized_spaced.replace(" ", "_"),
            normalized_spaced.replace(" ", "")
        ]:
            if variation in portal_mapping:
                return variation
    
    return None


# The 16 main faction portals
MAIN_PORTALS = {
    "Adeptus Custodes", "Mechanicum", "Astra Militarum", "Chaos",
    "Drukhari", "Asuryani", "Genestealer Cults", "Imperium",
    "Inquisition", "Leagues of Votann", "Adepta Sororitas", "Space Marines",
    "Necrons", "Orks", "T'au", "Tyranids"
}


def is_main_portal(portal_name):
    """Check if a portal name is already one of the 16 main portals."""
    if not portal_name:
        return False
    
    # Try exact match
    if portal_name in MAIN_PORTALS:
        return True
    
    # Try camelCase to spaced conversion
    spaced = camel_case_to_spaced(portal_name)
    if spaced in MAIN_PORTALS:
        return True
    
    # Try case-insensitive match
    portal_lower = portal_name.lower()
    spaced_lower = spaced.lower()
    for main_portal in MAIN_PORTALS:
        if main_portal.lower() == portal_lower or main_portal.lower() == spaced_lower:
            return True
    
    return False


def map_portal_to_main_portals(portal_name, portal_mapping):
    """Map a portal name to one of the 16 main portals.
    
    If the portal is already a main portal, returns it directly.
    Otherwise, uses the portal mapping file to find the main portal(s).
    If multiple portals are found, returns the last one (as requested).
    Returns None if no mapping is found.
    """
    if not portal_name:
        return None
    
    # If it's already a main portal, return it directly (don't remap it)
    if is_main_portal(portal_name):
        # Return the canonical main portal name
        spaced = camel_case_to_spaced(portal_name)
        for main_portal in MAIN_PORTALS:
            if (main_portal.lower() == portal_name.lower() or 
                main_portal.lower() == spaced.lower()):
                return main_portal
        return portal_name  # Fallback
    
    # Not a main portal, use mapping file
    if not portal_mapping:
        return None
    
    # Find matching key in the mapping
    matched_key = match_portal_to_key(portal_name, portal_mapping)
    if not matched_key:
        return None
    
    # Get the main portals for this key
    main_portals = portal_mapping[matched_key]
    
    if not main_portals:
        return None
    
    # If multiple portals, return the last one (as requested for conflict resolution)
    return main_portals[-1] if isinstance(main_portals, list) else main_portals


def extract_portals(wikitext, portal_mapping=None):
    """Extract portal names from wikitext and map them to main portals.
    
    Only extracts faction portal templates that end with "Portal" (capital P).
    Maps extracted portals to one of the 16 main portals using the mapping file.
    Examples: {{AstraMilitarumPortal}}, {{ImperiumPortal}}, {{InquisitionPortal}}
    
    Args:
        wikitext: The wikitext to extract portals from
        portal_mapping: Optional portal mapping dictionary. If None, returns raw portals.
    
    Returns:
        List of main portal names (mapped to 16 main portals if mapping provided)
    """
    if not wikitext:
        return []
    
    # Strict pattern: Only match {{NamePortal}} where Name is alphanumeric/underscore
    portal_pattern = r'\{\{([A-Za-z0-9_]+Portal)\}\}'
    portals = re.findall(portal_pattern, wikitext)
    
    cleaned_portals = []
    for portal in portals:
        # Remove "Portal" suffix to get the faction name
        cleaned = portal.replace('Portal', '').strip()
        if cleaned:
            normalized = normalize_portal_name(cleaned)
            if normalized:
                cleaned_portals.append(normalized)
    
    # If no mapping provided, return only the last portal if multiple found
    if not portal_mapping:
        if cleaned_portals:
            # Remove duplicates while preserving order, then return last one
            seen = set()
            unique_portals = []
            for portal in cleaned_portals:
                if portal not in seen:
                    seen.add(portal)
                    unique_portals.append(portal)
            return [unique_portals[-1]] if unique_portals else []
        return []
    
    # Map each portal to main portals
    main_portals = []
    for portal in cleaned_portals:
        main_portal = map_portal_to_main_portals(portal, portal_mapping)
        if main_portal:
            main_portals.append(main_portal)
    
    # If multiple portals found, return only the last one (as requested for ambiguity resolution)
    if main_portals:
        # Remove duplicates while preserving order, then return last one
        seen = set()
        unique_portals = []
        for portal in main_portals:
            if portal not in seen:
                seen.add(portal)
                unique_portals.append(portal)
        return [unique_portals[-1]] if unique_portals else []
    
    return []


def is_character(title):
    """Check if title represents a character (not a list, formation, etc.)."""
    title_lower = title.lower()
    exclude_patterns = [
        r'^list of',
        r'^known members of',
        r'^known captains of',
        r'^members of',
        r'^characters of',
        r'formation',
        r'^category:',
        r'\(list\)$',  # Ends with "(list)" or "(List)"
        r'\(list\)',   # Contains "(list)" anywhere
    ]
    for pattern in exclude_patterns:
        if re.search(pattern, title_lower):
            return False
    return True


# General races/species - these should be stored as race attribute, not affiliation
GENERAL_RACES = {
    "Human", "Ork", "Eldar", "Tau", "Necron", "Tyranid", "Chaos Human",
    "Abhumans and Mutants", "Blank", "Daemon"
}


def extract_race(affiliations):
    """Extract race/species from affiliations.
    
    Returns the first race found in the affiliations, or None if no race found.
    """
    if not affiliations:
        return None
    
    for aff in affiliations:
        if aff in GENERAL_RACES:
            return aff
    
    return None


def determine_primary_affiliation(affiliations):
    """Determine primary affiliation based on specificity.
    
    Races are excluded from primary affiliation - they should be stored separately.
    Returns None if only races are present.
    """
    if not affiliations:
        return None
    
    # Filter out races - they should be stored as a separate race attribute
    specific = [aff for aff in affiliations if aff not in GENERAL_RACES]
    
    # Return the first specific affiliation, or None if only races present
    return specific[0] if specific else None


def load_batch_files(raw_data_path, batch_pattern):
    """Load all pages from batch JSON files."""
    from pathlib import Path
    import json
    
    batch_files = sorted(raw_data_path.glob(batch_pattern))
    all_pages = {}
    
    for batch_file in batch_files:
        with open(batch_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'pages' in data:
                all_pages.update(data['pages'])
    
    return all_pages, batch_files


def create_character_title_map(characters):
    """Create a mapping from character title variations to canonical titles."""
    character_titles = set()
    character_title_map = {}
    
    for char in characters:
        title = char["title"]
        name = char["name"]
        file_name = char["file_name"]
        
        character_titles.add(title)
        character_titles.add(name)
        character_titles.add(file_name)
        
        # Map variations to canonical title
        character_title_map[title] = title
        character_title_map[name] = title
        character_title_map[file_name] = title
        character_title_map[title.replace("_", " ")] = title
        character_title_map[name.replace(" ", "_")] = title
    
    return character_titles, character_title_map


def find_character_title(link, character_title_map):
    """Find the canonical character title for a given link."""
    # Try exact match first
    if link in character_title_map:
        return character_title_map[link]
    
    # Try with underscores/spaces swapped
    for variation in [link.replace(" ", "_"), link.replace("_", " ")]:
        if variation in character_title_map:
            return character_title_map[variation]
    
    return None

