import pandas as pd
import json
import os
import csv
from collections import defaultdict

# --- Configuration ---
DATA_PATH = "kb_pipeline_v2/03_cleaned/"
CSV_FILE = "interpretations.cleaned.csv"
OUTPUT_CSV = "schema_report.csv"

# --- NEW: Refine this list after reviewing the "unclustered" output ---
SEMANTIC_KEYWORDS = [
    'planet', 'house', 'sign', 'lord', 'dignity', 'nakshatra', 'aspect',
    'conjunction', 'karaka', 'arudha', 'varga', 'dasha', 'yoga', 'dosha',
    'strength', 'avastha', 'ashtakavarga', 'period', 'relationship', 'placement'
]

def find_key_paths(data, path_counts, current_path=''):
    """Recursively finds all unique key paths and increments their counts."""
    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{current_path}.{key}" if current_path else key
            path_counts[new_path] += 1
            find_key_paths(value, path_counts, new_path)
    elif isinstance(data, list):
        new_path = f"{current_path}[]"
        for item in data:
            find_key_paths(item, path_counts, new_path)
    # Note: Context sampling could be added here for a more advanced version.

def main():
    print("🚀 Starting JSON Schema Clustering Process...")
    input_path = os.path.join(DATA_PATH, CSV_FILE)

    if not os.path.exists(input_path):
        print(f"❌ Error: File not found at {input_path}")
        return

    df = pd.read_csv(input_path, dtype=str).fillna("")
    print(f"Loaded {len(df)} rows to analyze.")

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

    # --- Clustering Logic ---
    clusters = defaultdict(list)
    for path in sorted(path_counts.keys()):
        found = False
        for keyword in SEMANTIC_KEYWORDS:
            if keyword in path.lower():
                clusters[keyword].append(path)
                found = True
                break
        if not found:
            clusters['unclustered'].append(path)
    
    # --- NEW: Write to CSV ---
    output_path = os.path.join(os.path.dirname(DATA_PATH), OUTPUT_CSV)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['cluster', 'key_path', 'occurrence_count'])
        for cluster_name, paths in clusters.items():
            for path in paths:
                writer.writerow([cluster_name, path, path_counts[path]])

    print("\n" + "="*40)
    print("--- ✅ Diagnosis Complete ---")
    print(f"Report saved to: {output_path}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()