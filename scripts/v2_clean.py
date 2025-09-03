# v2_clean.py

import pandas as pd
import os
import json
from jsonschema import validate, ValidationError

from v2_kb_utils import (
    V2_INTERPRETATIONS_SCHEMA,
    CONSOLIDATED_DATA_PATH,
    CLEANED_DATA_PATH,
    save_df
)

# --- FORMAL JSON SCHEMA FOR THE ASTROLOGICAL TRIGGER ---
# This is our structural guarantee for the database.
TRIGGER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "calculation_context": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "name": {"type": "string"}
            },
            "required": ["type"]
        },
        "components": {
            "type": "array",
            "items": {"type": "object"}
        }
    },
    "required": ["calculation_context", "components"]
}


def clean_text(text: str) -> str:
    """A simple text cleaner for final polishing."""
    if not isinstance(text, str):
        return ""
    # Fix common unicode errors and strip whitespace
    text = text.replace("â€™", "’").replace("â€œ", "“").replace("â€", "”")
    return " ".join(text.strip().split())


def main():
    print("🚀 Starting V2 Final Cleaning and Validation Process...")
    input_file = os.path.join(CONSOLIDATED_DATA_PATH, "interpretations.consolidated.csv")
    if not os.path.exists(input_file):
        print("Consolidated file not found. Please run consolidate script first. Exiting.")
        return

    df = pd.read_csv(input_file, dtype=str).fillna("")
    print(f"📄 Processing {len(df)} consolidated interpretations.")

    cleaned_rows = []
    error_rows = []

    for index, row in df.iterrows():
        # 1. Clean text fields for EVERY row first to ensure consistent output
        processed_row = row.to_dict()
        processed_row['interpretation_text'] = clean_text(row['interpretation_text'])
        processed_row['interpretation_summary_ai'] = clean_text(row['interpretation_summary_ai'])
        processed_row['interpretation_summary_raw'] = clean_text(row['interpretation_summary_raw'])

        # 2. Validate the Trigger JSON
        try:
            trigger_obj = json.loads(processed_row['astrological_trigger_json'])
            validate(instance=trigger_obj, schema=TRIGGER_JSON_SCHEMA)
            
            # If validation succeeds, finalize status and add to cleaned list
            processed_row['status'] = 'CLEANED'
            cleaned_rows.append(processed_row)
            
        except (json.JSONDecodeError, ValidationError) as e:
            # If validation fails, add notes and add to error list
            print(f"  - ❌ Schema Validation Failed for Fact ID {processed_row['fact_id']}: {e}")
            processed_row['notes'] = f"CLEANING_ERROR: JSON schema validation failed. {e}"
            processed_row['status'] = 'VALIDATION_FAILED'
            error_rows.append(processed_row)

    if cleaned_rows:
        cleaned_df = pd.DataFrame(cleaned_rows)
        cleaned_df = cleaned_df[V2_INTERPRETATIONS_SCHEMA]
        save_df(cleaned_df, CLEANED_DATA_PATH, "interpretations.cleaned.csv")
    
    if error_rows:
        errors_df = pd.DataFrame(error_rows)
        errors_df = errors_df[V2_INTERPRETATIONS_SCHEMA]
        save_df(errors_df, CLEANED_DATA_PATH, "interpretations.errors.csv")

    print("\n✅ V2 Clean process complete.")
    print(f"   -> {len(cleaned_rows)} rows passed validation.")
    print(f"   -> {len(error_rows)} rows failed validation (see errors.csv).")

if __name__ == "__main__":
    main()