# configuration and path settings for network creation
import pickle
import pandas as pd
from pathlib import Path

# get script directory and set up paths relative to script location
SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR.parent
RAW_DATA_PATH = BASE_DIR / "raw_data"
DATA_PATH = BASE_DIR / "data"
DATA_PATH.mkdir(exist_ok=True)
IMAGES_PATH = BASE_DIR / "images"
IMAGES_PATH.mkdir(exist_ok=True)

# file paths
CHAR_CATEGORIES_FILE = RAW_DATA_PATH / "lexicanum_characters_by_category_generated.json"
AFFILIATION_MAPPING_FILE = DATA_PATH / "character_affiliation_mapping.json"
CHARACTERS_FILE = DATA_PATH / "lexicanum_characters.json"
EDGES_FILE = DATA_PATH / "lexicanum_edges.json"
GEXF_FILE = DATA_PATH / "lexicanum_network.gexf"
PICKLE_FILE = DATA_PATH / "lexicanum_network.pkl"
GEXF_FILTERED_FILE = DATA_PATH / "lexicanum_network_filtered.gexf"
PICKLE_FILTERED_FILE = DATA_PATH / "lexicanum_network_filtered.pkl"
PORTAL_MAPPING_FILE = DATA_PATH / "faction_portal_affiliations_comprehensive.json"

# batch file pattern
BATCH_FILE_PATTERN = "lexicanum_page_texts_batch_*.json"

def load_network(use_filtered=True):
    # Load network from pickle file
    network_file = PICKLE_FILTERED_FILE if use_filtered else PICKLE_FILE
    with open(network_file, 'rb') as f:
        return pickle.load(f)

def load_results():
    # Load analysis results from CSV
    csv_path = BASE_DIR / "data" / "mythology_vs_narrative_analysis.csv"
    return pd.read_csv(csv_path)
