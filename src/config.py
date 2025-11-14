"""Configuration and path settings for network creation."""
from pathlib import Path

# Get the script's directory and set up paths relative to script location
SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR.parent  # Go up one level from notebooks/ to Final Assignment/
RAW_DATA_PATH = BASE_DIR / "raw_data"
DATA_PATH = BASE_DIR / "data"
DATA_PATH.mkdir(exist_ok=True)

# File paths
CHAR_CATEGORIES_FILE = RAW_DATA_PATH / "lexicanum_characters_by_category_generated.json"
AFFILIATION_MAPPING_FILE = DATA_PATH / "character_affiliation_mapping.json"
CHARACTERS_FILE = DATA_PATH / "lexicanum_characters.json"
EDGES_FILE = DATA_PATH / "lexicanum_edges.json"
GEXF_FILE = DATA_PATH / "lexicanum_network.gexf"
PICKLE_FILE = DATA_PATH / "lexicanum_network.pkl"
GEXF_FILTERED_FILE = DATA_PATH / "lexicanum_network_filtered.gexf"
PICKLE_FILTERED_FILE = DATA_PATH / "lexicanum_network_filtered.pkl"
PORTAL_MAPPING_FILE = DATA_PATH / "faction_portal_affiliations_comprehensive.json"

# Batch file pattern
BATCH_FILE_PATTERN = "lexicanum_page_texts_batch_*.json"


