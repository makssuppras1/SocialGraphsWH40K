"""
extract_page_texts.py

Fetch full page wikitext (latest revision) for every character page listed in the
crawler output JSON and write results to a single JSON file.

Output format:
{
  "source": "path/to/input.json",
  "generated": 1234567890.0,
  "pages": {
    "Title A": {"pageid": 123, "wikitext": "..."},
    "Title B": {"pageid": null, "wikitext": null, "missing": true},
    ...
  }
}

Usage (PowerShell):
  python .\extract_page_texts.py --input ..\data\raw\lexicanum_characters_by_category.json --output ..\data\processed\lexicanum_page_texts.json --limit 200
"""

from __future__ import annotations
import argparse
import json
import time
import random
from pathlib import Path
import sys

try:
    import requests
    from requests.exceptions import RequestException
except Exception:
    print('requests not found; please install requests (pip install requests)')
    raise

API_URL = "https://wh40k.lexicanum.com/mediawiki/api.php"
HEADERS = {"User-Agent": "lexicanum-page-text-extractor/1.0 (contact:example@example.com)"}
DEFAULT_SLEEP = 0.3
DEFAULT_MAX_RETRIES = 3
# backoff defaults
DEFAULT_BACKOFF_BASE = 1.0
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_BACKOFF_MAX = 60.0
DEFAULT_BACKOFF_JITTER = 0.1


def api_get(session, params, max_retries=None, backoff_base=DEFAULT_BACKOFF_BASE,
            backoff_factor=DEFAULT_BACKOFF_FACTOR, backoff_max=DEFAULT_BACKOFF_MAX,
            jitter=DEFAULT_BACKOFF_JITTER):
    if max_retries is None:
        max_retries = DEFAULT_MAX_RETRIES
    tries = 0
    while True:
        try:
            r = session.get(API_URL, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except RequestException as e:
            tries += 1
            if tries >= max_retries:
                raise
            delay = min(backoff_max, backoff_base * (backoff_factor ** (tries - 1)))
            jitter_amt = random.uniform(-jitter * delay, jitter * delay)
            sleep_time = max(0.1, delay + jitter_amt)
            print(f"Request error ({e}) — retry {tries}/{max_retries} in {sleep_time:.1f}s...")
            time.sleep(sleep_time)


def load_titles_from_crawler(input_path: Path) -> list:
    with input_path.open('r', encoding='utf-8') as fh:
        payload = json.load(fh)
    pages_by_category = payload.get('pages_by_category') or payload.get('pages_by_cat') or {}
    titles = set()
    for cat, pages in pages_by_category.items():
        for p in pages:
            t = p.get('title') if isinstance(p, dict) else p
            if t:
                titles.add(t)
    return sorted(titles)


def get_page_wikitext(session, title: str, max_retries: int, backoff_opts: dict) -> dict:
    """Return dict with pageid and wikitext (or missing flag).

    Uses 'query' + prop=revisions&rvprop=content&rvslots=main to retrieve latest wikitext.
    """
    params = {
        'action': 'query',
        'titles': title,
        'prop': 'revisions',
        'rvslots': 'main',
        'rvprop': 'content',
        'format': 'json'
    }
    data = api_get(session, params, max_retries=max_retries, **backoff_opts)
    pages = data.get('query', {}).get('pages', {})
    for page in pages.values():
        if page.get('missing') is not None:
            return {'pageid': None, 'wikitext': None, 'missing': True}
        pageid = page.get('pageid')
        revs = page.get('revisions') or []
        if revs:
            rev = revs[0]
            # MediaWiki stores content under slots->main->'*' in newer versions
            wikitext = None
            slots = rev.get('slots')
            if isinstance(slots, dict):
                wikitext = slots.get('main', {}).get('*')
            if wikitext is None:
                wikitext = rev.get('*') or rev.get('content')
            return {'pageid': pageid, 'wikitext': wikitext}
        # no revisions found
        return {'pageid': pageid, 'wikitext': None}
    return {'pageid': None, 'wikitext': None}


def main():
    parser = argparse.ArgumentParser(description='Fetch wikitext for pages listed in crawler JSON')
    parser.add_argument('--input', '-i', default='lexicanum_characters_by_category.json', help='crawler JSON file')
    parser.add_argument('--output', '-o', default='lexicanum_page_texts.json', help='output JSON file')
    parser.add_argument('--limit', type=int, default=None, help='limit number of pages (for testing)')
    parser.add_argument('--start', type=int, default=0, help='start offset for batching (zero-based)')
    parser.add_argument('--max-retries', type=int, default=DEFAULT_MAX_RETRIES, help='max API attempts per request')
    parser.add_argument('--backoff-base', type=float, default=DEFAULT_BACKOFF_BASE)
    parser.add_argument('--backoff-factor', type=float, default=DEFAULT_BACKOFF_FACTOR)
    parser.add_argument('--backoff-max', type=float, default=DEFAULT_BACKOFF_MAX)
    parser.add_argument('--backoff-jitter', type=float, default=DEFAULT_BACKOFF_JITTER)
    parser.add_argument('--sleep', type=float, default=DEFAULT_SLEEP, help='polite pause between requests')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(2)

    titles = load_titles_from_crawler(input_path)
    # apply batching slice
    if args.start:
        titles = titles[args.start:]
    if args.limit:
        titles = titles[:args.limit]

    print(f"Will fetch {len(titles)} pages from {input_path}")
    session = requests.Session()

    out = {}
    backoff_opts = {
        'backoff_base': args.backoff_base,
        'backoff_factor': args.backoff_factor,
        'backoff_max': args.backoff_max,
        'jitter': args.backoff_jitter
    }

    for i, title in enumerate(titles, start=1):
        try:
            if i % 50 == 0 or i == 1:
                print(f"Progress: {i}/{len(titles)} — last: {title}")
            res = get_page_wikitext(session, title, max_retries=args.max_retries, backoff_opts=backoff_opts)
            out[title] = res
        except Exception as e:
            print(f"Failed to fetch '{title}': {e}")
            out[title] = {'pageid': None, 'wikitext': None, 'error': str(e)}
        time.sleep(args.sleep)

    payload = {
        'source': str(input_path),
        'generated': time.time(),
        'pages': out
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(f"Wrote page texts to: {out_path} (pages: {len(out)})")


if __name__ == '__main__':
    main()
