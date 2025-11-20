#!/usr/bin/env python3
import csv
from pathlib import Path

INPUT = Path(__file__).parent / 'rpps_long_clean.csv'
BACKUP = INPUT.with_suffix('.csv.bak')

if not INPUT.exists():
    print(f'Input file not found: {INPUT}')
    raise SystemExit(1)

# Backup
INPUT.replace(BACKUP)
print(f'Backup created: {BACKUP}')

kept = 0
removed = 0
TEMP = INPUT.with_suffix('.clean.csv')

with BACKUP.open('r', newline='', encoding='utf-8') as fin, TEMP.open('w', newline='', encoding='utf-8') as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    for row in reader:
        # Skip empty rows
        if not any(cell.strip() for cell in row if isinstance(cell, str)):
            continue
        # If any cell contains 'total' (case-insensitive), treat as a summary/total row and skip
        if any('total' in str(cell).lower() for cell in row):
            removed += 1
            continue
        writer.writerow(row)
        kept += 1

# Replace original with cleaned file
TEMP.replace(INPUT)
print(f'Cleaned file written: {INPUT}')
print(f'Rows kept: {kept}, rows removed: {removed}')
