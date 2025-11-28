# helper functions for network creation stuff
import re
import json
from pathlib import Path


# constants
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
    # normalizes portal names like SpaceMarines -> SpaceMarine
    if not portal_name:
        return portal_name
    
    # common plural patterns
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
    # converts camelCase to spaced like 'AstraMilitarum' -> 'Astra Militarum'
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', name).strip()


def _generate_name_variations(name):
    # makes different variations of a name for matching
    spaced = camel_case_to_spaced(name)
    variations = [
        name, spaced,
        name.replace("_", " "), name.replace(" ", "_"),
        name.replace("_", ""), name.replace(" ", ""),
        spaced.replace(" ", "_"), spaced.replace(" ", "")
    ]
    return variations


def load_portal_mapping(portal_mapping_file):
    # loads portal mapping from json file
    if not portal_mapping_file.exists():
        return {}
    with open(portal_mapping_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_main_portal(portal_name):
    # checks if portal is one of the 16 main ones
    if not portal_name:
        return False
    
    portal_lower = portal_name.lower()
    spaced_lower = camel_case_to_spaced(portal_name).lower()
    
    for mp in MAIN_PORTALS:
        if mp.lower() == portal_lower or mp.lower() == spaced_lower:
            return True
    return False


def match_portal_to_key(portal_name, portal_mapping):
    # tries to match portal name to a key in mapping using variations
    if not portal_name or not portal_mapping:
        return None
    
    # try exact match and variations
    variations = _generate_name_variations(portal_name)
    for variation in variations:
        if variation in portal_mapping:
            return variation
    
    # try case insensitive
    portal_lower = portal_name.lower()
    spaced_lower = camel_case_to_spaced(portal_name).lower()
    for key in portal_mapping.keys():
        if key.lower() in (portal_lower, spaced_lower):
            return key
    
    # try normalized
    normalized = normalize_portal_name(portal_name)
    if normalized and normalized != portal_name:
        normalized_variations = _generate_name_variations(normalized)
        for variation in normalized_variations:
            if variation in portal_mapping:
                return variation
    
    return None


def map_portal_to_main_portals(portal_name, portal_mapping):
    # maps portal name to one of the 16 main portals
    if not portal_name:
        return None
    
    # check if already main portal
    if is_main_portal(portal_name):
        spaced = camel_case_to_spaced(portal_name)
        for main_portal in MAIN_PORTALS:
            if main_portal.lower() in (portal_name.lower(), spaced.lower()):
                return main_portal
        return portal_name
    
    # use mapping file
    if not portal_mapping:
        return None
    
    matched_key = match_portal_to_key(portal_name, portal_mapping)
    if not matched_key:
        return None
    
    main_portals = portal_mapping[matched_key]
    if not main_portals:
        return None
    
    # return last portal if multiple (conflict resolution)
    if isinstance(main_portals, list):
        return main_portals[-1]
    else:
        return main_portals


def extract_portals(wikitext, portal_mapping=None):
    # extracts portal names from wikitext and maps to main portals
    # returns list with single portal (last one if multiple found)
    if not wikitext:
        return []
    
    # extract portal templates
    portal_pattern = r'\{\{([A-Za-z0-9_]+Portal)\}\}'
    portals = re.findall(portal_pattern, wikitext)
    
    # clean and normalize
    cleaned_portals = []
    for portal in portals:
        cleaned = portal.replace('Portal', '').strip()
        if cleaned:
            normalized = normalize_portal_name(cleaned)
            if normalized:
                cleaned_portals.append(normalized)
    
    # if no mapping, return last unique portal
    if not portal_mapping:
        unique_portals = list(dict.fromkeys(cleaned_portals))  # preserves order
        if unique_portals:
            return [unique_portals[-1]]
        else:
            return []
    
    # map to main portals
    main_portals = []
    for portal in cleaned_portals:
        main_portal = map_portal_to_main_portals(portal, portal_mapping)
        if main_portal:
            main_portals.append(main_portal)
    
    # return last unique portal
    unique_portals = list(dict.fromkeys(main_portals))
    if unique_portals:
        return [unique_portals[-1]]
    else:
        return []


def is_character(title):
    # checks if title is a character (not a list, formation, etc)
    title_lower = title.lower()
    for pattern in CHARACTER_EXCLUDE_PATTERNS:
        if re.search(pattern, title_lower):
            return False
    return True


def extract_race(affiliations):
    # gets race/species from affiliations, returns first race found or None
    if not affiliations:
        return None
    for aff in affiliations:
        if aff in GENERAL_RACES:
            return aff
    return None


def determine_primary_affiliation(affiliations):
    # gets primary affiliation, excluding races
    # races stored separately. returns first specific affiliation or None
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
    # loads all pages from batch json files
    batch_files = sorted(raw_data_path.glob(batch_pattern))
    all_pages = {}
    
    for batch_file in batch_files:
        with open(batch_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'pages' in data:
                all_pages.update(data['pages'])
    
    return all_pages, batch_files


def create_character_title_map(characters):
    # creates mapping from character title variations to canonical titles
    character_titles = set()
    character_title_map = {}
    
    for char in characters:
        title = char["title"]
        name = char["name"]
        file_name = char["file_name"]
        
        character_titles.update([title, name, file_name])
        
        # map variations to canonical title
        character_title_map[title] = title
        character_title_map[name] = title
        character_title_map[file_name] = title
        character_title_map[title.replace("_", " ")] = title
        character_title_map[name.replace(" ", "_")] = title
    
    return character_titles, character_title_map


def find_character_title(link, character_title_map):
    # finds canonical character title for a given link
    if link in character_title_map:
        return character_title_map[link]
    
    # try space/underscore variations
    variation1 = link.replace(" ", "_")
    if variation1 in character_title_map:
        return character_title_map[variation1]
    
    variation2 = link.replace("_", " ")
    if variation2 in character_title_map:
        return character_title_map[variation2]
    
    return None
