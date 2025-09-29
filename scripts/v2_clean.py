import pandas as pd
import os
import json
import string
from jsonschema import validate, ValidationError
from dotenv import load_dotenv
import google.generativeai as genai # <-- ADD THIS IMPORT
from typing import Optional, Dict, List

# --- Import V2 utilities ---
from v2_kb_utils import (
    TAXONOMY,
    CANONICAL_SCHEMAS,
    CONSOLIDATED_DATA_PATH,
    CLEANED_DATA_PATH,
    save_df
    # MODEL has been removed from here
)

# --- CONFIGURATION & INITIALIZATION ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = genai.GenerativeModel('gemini-1.5-pro-latest') # <-- ADD THIS DEFINITION

# --- NEW: AI Correction Helper Functions ---

def get_json_correction_prompt(incorrect_json: dict, error: ValidationError) -> str:
    """Creates a prompt to ask the AI to fix a JSON object based on a validation error."""
    return f"""
    You are a JSON syntax and schema correction expert. Your sole task is to fix a JSON object that failed validation.

    **FAILED JSON OBJECT:**
    ---
    {json.dumps(incorrect_json, indent=2)}
    ---

    **VALIDATION ERROR:**
    ---
    Error: "{error.message}"
    Path to error in JSON: {list(error.path)}
    ---

    **RELEVANT SCHEMA RULE IT FAILED:**
    ---
    {json.dumps(error.schema, indent=2)}
    ---

    **INSTRUCTIONS:**
    1. Analyze the error and the schema rule.
    2. Correct the FAILED JSON OBJECT so that it passes the schema validation.
    3. Your entire output MUST be ONLY the corrected, valid JSON object. Do not include any other text or explanations.

    **CORRECTED JSON RESPONSE:**
    """

def correct_json_with_ai(incorrect_json: dict, error: ValidationError) -> Optional[dict]:
    """Attempts to correct a JSON object using an AI call."""
    prompt = get_json_correction_prompt(incorrect_json, error)
    try:
        response = MODEL.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.0)
        )
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_text)
    except Exception as e:
        print(f"     - ⚠️ AI correction call itself failed: {e}")
        return None

# --- Core Helper Functions ---

def generate_master_schema(canonical_schemas: dict) -> dict:
    """Dynamically generates a comprehensive JSON schema from the CANONICAL_SCHEMAS."""
    schema_definitions = []
    type_mapping = {
        "string": {"type": ["string", "null"]},
        "integer": {"type": ["integer", "null"]},
        "float": {"type": ["number", "null"]},
        "list[string]": {"type": "array", "items": {"type": "string"}},
        "list[integer]": {"type": "array", "items": {"type": "integer"}}
    }
    for schema_name, schema_info in canonical_schemas.items():
        properties = {"type": {"const": schema_name}}
        required = ["type"]
        if 'attributes' in schema_info:
            for attr, attr_type in schema_info['attributes'].items():
                if attr != 'type':
                    properties[attr] = type_mapping.get(attr_type, {"type": "string"})
                    required.append(attr)
        definition = {
            "type": "object",
            "properties": properties,
            "required": sorted(list(set(required))),
            "additionalProperties": False
        }
        schema_definitions.append(definition)
    return {"oneOf": schema_definitions}

def build_valid_tags_set(taxonomy: dict) -> set:
    """Creates a flat set of all valid themes, groups, and sub-themes for fast lookups."""
    tags = set()
    for theme, groups in taxonomy.items():
        tags.add(theme)
        for group, sub_themes in groups.items():
            tags.add(group)
            for sub_theme in sub_themes:
                tags.add(sub_theme)
    return tags

def validate_thematic_tags(row: pd.Series, valid_tags: set) -> list:
    """Validates the theme, group, and sub-theme tags in a row against the taxonomy."""
    invalid_tags = []
    for col in ['theme', 'theme_group', 'sub_theme']:
        tags = [tag.strip() for tag in str(row.get(col, '')).split(',') if tag.strip()]
        for tag in tags:
            if tag not in valid_tags:
                invalid_tags.append(tag)
    return invalid_tags

def clean_text(text: str) -> str:
    """A simple text cleaner for final polishing."""
    if not isinstance(text, str):
        return ""
    text = text.replace("’", "'").replace("“", "\"").replace("”", "\"")
    text = " ".join(text.strip().split())
    text = text.strip(string.punctuation)
    return text

# --- MAIN ORCHESTRATOR ---

def main():
    print("🚀 Starting V2 Final Cleaning and Validation Process...")
    
    print("  - Dynamically generating JSON schema from CANONICAL_SCHEMAS...")
    master_json_schema = generate_master_schema(CANONICAL_SCHEMAS)
    
    print("  - Building valid thematic tags set from TAXONOMY...")
    valid_thematic_tags = build_valid_tags_set(TAXONOMY)
    
    input_file = os.path.join(CONSOLIDATED_DATA_PATH, "interpretations.production.csv")
    if not os.path.exists(input_file):
        print(f"❌ Production file not found at {input_file}. Exiting.")
        return

    df = pd.read_csv(input_file, dtype=str).fillna("")
    print(f"📄 Processing {len(df)} consolidated interpretations for final validation.")

    all_processed_rows = []

    for index, row in df.iterrows():
        processed_row = row.to_dict()
        error_messages = []
        validation_passed = True

        try:
            trigger_obj = json.loads(processed_row['astrological_trigger_json'])
            validate(instance=trigger_obj, schema=master_json_schema)
        except json.JSONDecodeError:
            error_messages.append("JSON_ERROR: Trigger is not valid JSON.")
            validation_passed = False
        except ValidationError as e:
            print(f"  -> 🔬 Schema error for fact_id {row.get('fact_id', 'N/A')}. Attempting AI correction...")
            corrected_obj = correct_json_with_ai(trigger_obj, e)
            
            if corrected_obj:
                try:
                    validate(instance=corrected_obj, schema=master_json_schema)
                    print("     - ✅ AI correction successful and validated.")
                    processed_row['astrological_trigger_json'] = json.dumps(corrected_obj)
                    processed_row['notes'] = "AI_CORRECTED: JSON passed validation after AI fix."
                except ValidationError as e2:
                    print("     - ❌ AI correction failed re-validation. Staging original.")
                    error_messages.append(f"SCHEMA_ERROR (AI_FIX_FAILED): {e2.message} (path: {list(e2.path)})")
                    validation_passed = False
            else:
                print("     - ❌ AI correction call failed. Staging original.")
                error_messages.append(f"SCHEMA_ERROR: {e.message} (path: {list(e.path)})")
                validation_passed = False

        invalid_tags = validate_thematic_tags(row, valid_thematic_tags)
        if invalid_tags:
            error_messages.append(f"THEME_ERROR: Invalid tags found: {', '.join(invalid_tags)}")
            validation_passed = False

        if not validation_passed:
            processed_row['notes'] = " | ".join(error_messages)
            processed_row['status'] = 'Staged'
        else:
            processed_row['status'] = 'Passed'
            for col in ['interpretation_text', 'interpretation_summary_ai']:
                if col in processed_row:
                    processed_row[col] = clean_text(processed_row[col])
            if "AI_CORRECTED" not in processed_row.get('notes', ''):
                processed_row['notes'] = '' # Clear old notes only if it wasn't an AI correction
        
        all_processed_rows.append(processed_row)

    if all_processed_rows:
        final_df = pd.DataFrame(all_processed_rows)
        
        final_output_schema = [
            "fact_id", "status", "theme", "theme_group", "sub_theme", "raw_text",
            "astrological_trigger_json", "interpretation_text", "interpretation_summary_ai",
            "confidence_score", "source_name", "source_reference", "last_updated", "notes"
        ]
        
        for col in final_output_schema:
            if col not in final_df.columns:
                final_df[col] = ''
        
        final_df = final_df[final_output_schema]
        
        output_filename = "interpretations.final.csv"
        save_df(final_df, CLEANED_DATA_PATH, output_filename)

        passed_count = (final_df['status'] == 'Passed').sum()
        staged_count = (final_df['status'] == 'Staged').sum()

        print("\n✅ V2 Clean process complete.")
        print(f"  -> Saved {len(final_df)} total rows to {output_filename}.")
        print(f"  -> {passed_count} rows have 'Passed' status.")
        print(f"  -> {staged_count} rows have been 'Staged' for review.")
    else:
        print("ℹ️ No rows were processed.")


if __name__ == "__main__":
    main()