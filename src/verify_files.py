#!/usr/bin/env python3
# verifies that filtered network files exist and shows their locations

from pathlib import Path
from config import GEXF_FILTERED_FILE, PICKLE_FILTERED_FILE, DATA_PATH

print("="*60)
print("FILTERED NETWORK FILES VERIFICATION")
print("="*60)

print(f"\nData directory: {DATA_PATH.resolve()}")
print(f"\nFiltered GEXF file:")
print(f"  Path: {GEXF_FILTERED_FILE.resolve()}")
print(f"  Exists: {GEXF_FILTERED_FILE.exists()}")
if GEXF_FILTERED_FILE.exists():
    size = GEXF_FILTERED_FILE.stat().st_size
    print(f"  Size: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")

print(f"\nFiltered Pickle file:")
print(f"  Path: {PICKLE_FILTERED_FILE.resolve()}")
print(f"  Exists: {PICKLE_FILTERED_FILE.exists()}")
if PICKLE_FILTERED_FILE.exists():
    size = PICKLE_FILTERED_FILE.stat().st_size
    print(f"  Size: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")

print("\n" + "="*60)
print("All files in data directory:")
print("="*60)
for file in sorted(DATA_PATH.glob("*")):
    if file.is_file():
        size = file.stat().st_size
        print(f"  {file.name:50} {size:>12,} bytes")
