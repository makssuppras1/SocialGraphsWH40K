import json
import requests
import time

API_URL = "https://wh40k.lexicanum.com/mediawiki/api.php"
HEADERS = {"User-Agent": "lexicanum-crawler/1.0 (contact:example@example.com)"}
SLEEP = 0.3

def fetch_character_data(character_name):
    params = {
        "action": "query",
        "titles": character_name,
        "prop": "extracts|links",
        "format": "json",
        "explaintext": True,
        "pllimit": "max"
    }
    response = requests.get(API_URL, params=params, headers=HEADERS)
    response.raise_for_status()
    return response.json()

def crawl_characters(character_list):
    character_data = {}
    for character in character_list:
        print(f"Fetching data for {character}...")
        data = fetch_character_data(character)
        character_data[character] = data
        time.sleep(SLEEP)
    return character_data

if __name__ == "__main__":
    characters_to_crawl = ["Emperor_of_Mankind", "Space_Marine", "Chaos_Daemon"]
    character_data = crawl_characters(characters_to_crawl)
    
    with open("data/raw/lexicanum_characters_by_category.json", "w", encoding="utf-8") as f:
        json.dump(character_data, f, ensure_ascii=False, indent=2)
    print("Character data has been saved.")