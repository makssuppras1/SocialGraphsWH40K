"""Extract categories for characters from lexicanum page text batches.

Usage:
    python categorize_characters.py --input-dir ../data/processed \
        --out-by-char ../data/processed/lexicanum_character_categories.json \
        --out-by-cat ../data/processed/lexicanum_characters_by_category_generated.json

Outputs two JSON files:
 - by-char: mapping character name -> list of category strings
 - by-cat: mapping category -> list of character names

The script looks for files matching: lexicanum_page_texts_batch_*.json
and extracts categories found in the page wikitext using the pattern
[[Category:...]].
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


CATEGORY_RE = re.compile(r"\[\[Category:([^\]|]+)")


def extract_categories_from_wikitext(wikitext: str) -> List[str]:
    if not wikitext:
        return []
    cats = CATEGORY_RE.findall(wikitext)
    # strip whitespace
    return [c.strip() for c in cats if c and c.strip()]


def process_batch_file(path: Path, by_char: Dict[str, List[str]], by_cat: Dict[str, List[str]]):
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    pages = data.get("pages") or {}
    for name, info in pages.items():
        wikitext = info.get("wikitext") if isinstance(info, dict) else None
        cats = extract_categories_from_wikitext(wikitext if wikitext else "")
        if not cats:
            continue
        # ensure deterministic ordering and uniqueness
        unique = sorted(dict.fromkeys(cats))
        by_char.setdefault(name, []).extend([c for c in unique if c not in by_char.get(name, [])])
        for c in unique:
            if name not in by_cat.get(c, []):
                by_cat.setdefault(c, []).append(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory with lexicanum_page_texts_batch_*.json files")
    parser.add_argument("--out-by-char", required=True, help="Output JSON file: character -> [categories]")
    parser.add_argument("--out-by-cat", required=True, help="Output JSON file: category -> [characters]")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    pattern = str(input_dir / "lexicanum_page_texts_batch_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No batch files found with pattern: {pattern}")

    by_char: Dict[str, List[str]] = {}
    by_cat: Dict[str, List[str]] = {}

    for f in files:
        print(f"Processing {f} ...")
        process_batch_file(Path(f), by_char, by_cat)

    # sort lists for determinism
    for k in by_char:
        by_char[k] = sorted(set(by_char[k]))
    for k in by_cat:
        by_cat[k] = sorted(set(by_cat[k]))

    out_by_char = Path(args.out_by_char)
    out_by_cat = Path(args.out_by_cat)
    out_by_char.parent.mkdir(parents=True, exist_ok=True)
    out_by_cat.parent.mkdir(parents=True, exist_ok=True)

    with out_by_char.open("w", encoding="utf-8") as fh:
        json.dump(by_char, fh, indent=2, ensure_ascii=False)

    with out_by_cat.open("w", encoding="utf-8") as fh:
        json.dump(by_cat, fh, indent=2, ensure_ascii=False)

    print(f"Wrote {len(by_char)} characters with categories to {out_by_char}")
    print(f"Wrote {len(by_cat)} categories to {out_by_cat}")


if __name__ == "__main__":
    main()
