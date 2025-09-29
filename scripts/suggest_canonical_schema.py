import pandas as pd
import os
import re
from collections import defaultdict

# --- Configuration ---
INPUT_CSV = "kb_pipeline_v2/03_cleaned/schema_report.csv"
OUTPUT_CSV = "kb_pipeline_v2/03_cleaned//canonical_schema_suggestions.csv"

# Canonical concepts (order matters: more specific first)
SEMANTIC_KEYWORDS = [
    'atmakaraka', 'darakaraka', 'karaka', 'arudha', 'lagna', 'dasha',
    'ashtakavarga', 'shadbala', 'vimsopaka', 'vargottama',
    'nakshatra', 'yoga', 'dosha', 'aspect', 'conjunction',
    'dignity', 'avastha', 'lord', 'planet', 'house', 'sign'
]

# Synonym dictionary for collapsing variants → canonical form
SYNONYM_MAP = {
    # Planet
    r'planet\d+$': 'planet',
    r'planet_\d+$': 'planet',
    r'planets$': 'planet',
    r'planetary_.*': 'planet',
    r'planet1.*': 'planet',
    r'planet2.*': 'planet',

    # House
    r'house\d+$': 'house',
    r'house_\d+$': 'house',
    r'house_number': 'house',
    r'house_index': 'house',
    r'houses$': 'house',
    r'ruled_house': 'house',
    r'house_ruled': 'house',
    r'house_ruler': 'house',

    # Lordship
    r'sign_lordship': 'sign_lord',
    r'sign_ruler': 'sign_lord',
    r'house_lordship': 'house_lord',
    r'house_ruler': 'house_lord',
    r'nakshatra_ruler': 'nakshatra_lord',
    r'sub_nakshatra_lord': 'nakshatra_lord',

    # Aspects
    r'aspects$': 'aspect',
    r'aspected_house': 'aspect.house',
    r'aspected_planet': 'aspect.planet',

    # Dignity
    r'planet_dignity': 'dignity',
    r'specific_dignity': 'dignity',
    r'dignity_strength.*': 'dignity',

    # Conjunction
    r'conjunctions$': 'conjunction',
    r'conjunction_planet': 'conjunction',

    # Generic noise
    r'\.any$': '',
    r'\.consider$': '',
    r'\.description$': '',
    r'\.status$': ''
}

def normalize_path(path: str) -> str:
    """Normalize raw key path into a cleaner canonical variant."""
    # 1. Drop noisy prefixes
    path = re.sub(r'^(components\[\]\.|calculation_context\.|interpretation_context\.)', '', path)

    # 2. Collapse multiple brackets (e.g. []. → .)
    path = path.replace('[]', '')

    # 3. Apply synonym rules
    for pattern, replacement in SYNONYM_MAP.items():
        path = re.sub(pattern, replacement, path)

    # 4. Collapse duplicate dots
    path = re.sub(r'\.+', '.', path).strip('.')

    return path


def classify_path(path: str) -> str:
    """Classify a normalized path into a high-level semantic concept."""
    lower_path = path.lower()
    for keyword in SEMANTIC_KEYWORDS:
        if keyword in lower_path:
            return keyword
    return 'unclustered'


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: Input file not found at {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} unique key paths.")

    clusters = defaultdict(list)

    for _, row in df.iterrows():
        raw_path = row["key_path"]
        count = int(row["occurrence_count"])

        norm_path = normalize_path(raw_path)
        concept = classify_path(norm_path)

        clusters[concept].append({'raw': raw_path, 'normalized': norm_path, 'count': count})

    # --- Build Suggestions ---
    suggestion_rows = []
    for concept, variants in clusters.items():
        if not variants:
            continue

        # Aggregate counts per normalized key
        agg = defaultdict(int)
        for v in variants:
            agg[v['normalized']] += v['count']

        sorted_variants = sorted(agg.items(), key=lambda x: -x[1])
        suggested = sorted_variants[0][0]
        total_occ = sum(agg.values())

        suggestion_rows.append({
            "primary_concept": concept,
            "suggested_canonical_path": suggested,
            "total_occurrences": total_occ,
            "num_variants": len(agg),
            "all_variants_and_counts": "; ".join([f"{k} ({v})" for k, v in sorted_variants])
        })

    # Save output
    out_df = pd.DataFrame(suggestion_rows).sort_values(by="total_occurrences", ascending=False)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "="*40)
    print("--- ✅ Canonical Schema Suggestion Report ---")
    print(f"Generated {len(out_df)} canonical concept groups.")
    print(f"Saved to: {OUTPUT_CSV}")
    print("="*40 + "\n")
    print("Top 10 Concept Groups:\n")
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
