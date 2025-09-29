import pandas as pd
import json
import os
import csv
from collections import defaultdict

# --- Configuration ---
DATA_PATH = "kb_pipeline_v2/03_cleaned/"
CSV_FILE = "interpretations.cleaned.csv"
OUTPUT_CSV = "kb_pipeline_v2/schema_suggestions.csv"

SEMANTIC_KEYWORDS = [
    'atmakaraka', 'darakaraka', 'karaka', 'arudha', 'lagna', 'dasha', 'ashtakavarga',
    'shadbala', 'vimsopaka', 'vargottama', 'nakshatra', 'yoga', 'dosha', 'aspect',
    'conjunction', 'dignity', 'avastha', 'lord', 'planet', 'house', 'sign'
]

def find_key_paths(data, path_counts, current_path=''):
    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{current_path}.{key}" if current_path else key
            path_counts[new_path] += 1
            find_key_paths(value, path_counts, new_path)
    elif isinstance(data, list):
        new_path = f"{current_path}[]"
        for item in data:
            find_key_paths(item, path_counts, new_path)

def classify_path(path: str) -> str:
    lower_path = path.lower()
    for keyword in SEMANTIC_KEYWORDS:
        if keyword in lower_path:
            return keyword
    return 'unclustered'

def main():
    print("🚀 Starting Schema Suggestion Process...")
    df = pd.read_csv(os.path.join(DATA_PATH, CSV_FILE), dtype=str).fillna("")
    path_counts = defaultdict(int)
    for json_string in df['astrological_trigger_json'].dropna():
        try:
            data = json.loads(json.loads(json_string))
            find_key_paths(data, path_counts)
        except:
            try:
                data = json.loads(json_string)
                find_key_paths(data, path_counts)
            except:
                continue

    clusters = defaultdict(list)
    for path, count in path_counts.items():
        concept = classify_path(path)
        clusters[concept].append({'path': path, 'count': count})

    suggestion_rows = []
    for concept, variants in clusters.items():
        if not variants: continue
        variants_sorted = sorted(variants, key=lambda x: -x['count'])
        suggested_canonical = variants_sorted[0]['path']
        for variant in variants:
            suggestion_rows.append({
                "primary_concept": concept,
                "variant_path": variant['path'],
                "occurrence_count": variant['count'],
                "suggested_canonical_path": suggested_canonical
            })
            
    out_df = pd.DataFrame(suggestion_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Schema suggestions saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()