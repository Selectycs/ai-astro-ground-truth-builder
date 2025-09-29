# v4_clean_and_validate.py (with Auto File Detection)

import pandas as pd
import os
import json
import string
from jsonschema import validate, ValidationError
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Optional, Dict
from tqdm import tqdm
from datetime import datetime

# --- Import V2 Utilities ---
from v2_kb_utils import TAXONOMY, CANONICAL_SCHEMAS

# --- CONFIGURATION & INITIALIZATION ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = genai.GenerativeModel('gemini-1.5-pro-latest')

# Updated to look for V7 output
INPUT_DATA_PATH = "kb_pipeline_v7/processed" 
CLEANED_DATA_PATH = "kb_pipeline_v7/cleaned"
os.makedirs(CLEANED_DATA_PATH, exist_ok=True)

# --- HELPER FUNCTIONS ---
def save_df(df: pd.DataFrame, path: str, filename: str):
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    df.to_csv(full_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Successfully saved {len(df)} rows to '{full_path}'.")

# --- AI HELPER PROMPTS & FUNCTIONS ---
def get_regeneration_prompt(row: pd.Series) -> str:
    manual_topic_hint = row['topic']
    source_text = row['raw_text']
    all_schemas_str = json.dumps(CANONICAL_SCHEMAS, indent=2)
    taxonomy_str = json.dumps(TAXONOMY, indent=2)
    
    return f"""
You are an expert astrological data processor. A previous AI process has made a mistake. Your task is to re-process the given text using a manual **Correction Hint** to generate a new, correct set of data.

**YOUR LOGIC:**
1.  **Determine Schema:** First, analyze the **Correction Hint** to determine the single most appropriate schema type from the **Master Schema List**.
2.  **Synthesize Data:** Create a valid JSON object for that schema. You MUST synthesize information from BOTH sources:
    - Use the **Correction Hint** to identify the primary astrological entities (e.g., Mahadasha Lord, Antardasha Lord, planet names, house numbers).
    - Use the **Source Text** to fill in the descriptive attributes of the JSON (e.g., `condition`, `significations`, `karaka_for`, descriptions).
3.  **Create Summary & Themes:** Based on your combined understanding of both texts, also generate a new summary and theme keywords.

Your output MUST be a single JSON object with four keys: "completed_json", "summary", "inferred_context", and "classification_keywords".

---
**Master Schema List (for your reference):**
{all_schemas_str}

---
**Correction Hint (Primary Source for Entities):**
{manual_topic_hint}

---
**Source Text (Primary Source for Descriptions):**
{source_text}

---
**REFERENCE - Taxonomy**:
{taxonomy_str}

---
**JSON Response (with four keys):**
"""

def get_json_correction_prompt(incorrect_json: dict, error: ValidationError) -> str:
    return f"""
You are a JSON syntax and schema correction expert. Your sole task is to fix a JSON object that failed validation. Analyze the error and the schema rule, then correct the FAILED JSON OBJECT so it passes validation. Your entire output MUST be ONLY the corrected, valid JSON object.

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
**CORRECTED JSON RESPONSE:**
"""

def call_ai_model(prompt: str) -> Optional[Dict]:
    try:
        response = MODEL.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_text)
    except Exception as e:
        print(f"  - ⚠️ AI call failed: {e}")
        return None

# --- CORE HELPER FUNCTIONS ---
def generate_master_schema(canonical_schemas: dict) -> dict:
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
    tags = set()
    for theme, groups in taxonomy.items():
        tags.add(theme)
        for group, sub_themes in groups.items():
            tags.add(group)
            for sub_theme in sub_themes:
                tags.add(sub_theme)
    return tags

def validate_thematic_tags(row: pd.Series, valid_tags: set) -> list:
    invalid_tags = []
    for col in ['theme', 'theme_group', 'sub_theme']:
        tags = [tag.strip() for tag in str(row.get(col, '')).split(',') if tag.strip()]
        for tag in tags:
            if tag not in valid_tags:
                invalid_tags.append(tag)
    return invalid_tags

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.replace("’", "'").replace("“", "\"").replace("”", "\"")
    text = " ".join(text.strip().split())
    text = text.strip(string.punctuation)
    return text

def pre_validate_json_fix(json_obj: dict) -> dict:
    """
    Automatically fixes common AI errors before validation,
    such as replacing null with [] for list attributes.
    """
    if not isinstance(json_obj, dict):
        return json_obj

    schema_type = json_obj.get('type')
    if not schema_type or schema_type not in CANONICAL_SCHEMAS:
        return json_obj

    # Get the attributes for the specific schema type
    attributes = CANONICAL_SCHEMAS[schema_type].get('attributes', {})

    for key, expected_type in attributes.items():
        if "list" in expected_type and key in json_obj and json_obj[key] is None:
            # If the attribute should be a list but is null, replace it with []
            json_obj[key] = []
            
    return json_obj

# --- MAIN ORCHESTRATOR ---
def main():
    print("🚀 Starting V4 Final Cleaning and Validation Process...")
    
    # --- THIS SECTION IS CHANGED FOR AUTOMATIC FILE FINDING ---
    print(f"🔎 Searching for the latest processed file in '{INPUT_DATA_PATH}'...")
    if not os.path.isdir(INPUT_DATA_PATH):
        print(f"❌ Input directory not found at '{INPUT_DATA_PATH}'. Please run the classifier script first. Exiting.")
        return

    try:
        # Find all .csv files in the directory
        files = [f for f in os.listdir(INPUT_DATA_PATH) if f.endswith('.csv')]
        if not files:
            raise FileNotFoundError # Trigger the except block if no files are found

        # Get the full path for each file and find the one with the latest modification time
        latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(INPUT_DATA_PATH, f)))
        input_file_path = os.path.join(INPUT_DATA_PATH, latest_file)
        
    except (ValueError, FileNotFoundError):
        print(f"❌ No processed CSV files found in '{INPUT_DATA_PATH}'. Please check the directory. Exiting.")
        return
    # --- END OF CHANGES ---
    
    master_json_schema = generate_master_schema(CANONICAL_SCHEMAS)
    valid_thematic_tags = build_valid_tags_set(TAXONOMY)
    
    print(f"📄 Processing latest file: '{latest_file}'")
    df = pd.read_csv(input_file_path, dtype=str).fillna("")
    print(f"  -> Found {len(df)} total rows to process.")

    all_processed_rows = []
    
    TAXONOMY_MAP = {}
    for theme, groups in TAXONOMY.items():
        for group, sub_themes in groups.items():
            for sub_theme in sub_themes: TAXONOMY_MAP[sub_theme] = {'theme': theme, 'group': group}
            TAXONOMY_MAP[group] = {'theme': theme, 'group': group}
        TAXONOMY_MAP[theme] = {'theme': theme, 'group': None}

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Cleaning & Validating"):
        processed_row = row.to_dict()
        error_messages = []
        validation_passed = True

        if pd.notna(row['topic']) and row['topic'].strip() != '':
            print(f"\n  -> Manual topic '{row['topic']}' found for row {index}. Regenerating fact...")
            regen_prompt = get_regeneration_prompt(row)
            ai_regen_obj = call_ai_model(regen_prompt)
            
            if ai_regen_obj:
                completed_json = ai_regen_obj.get('completed_json', {})
                processed_row['astrological_trigger_json'] = json.dumps(completed_json)
                processed_row['schema_type'] = completed_json.get('type', row['topic'])
                summary = ai_regen_obj.get('summary', '')
                inferred = ai_regen_obj.get('inferred_context', '')
                processed_row['ai_summary'] = f"{summary} (Additional Context: {inferred})" if inferred else summary
                keywords = ai_regen_obj.get('classification_keywords', [])
                themes, groups, sub_themes = set(), set(), set()
                for kw in keywords:
                    mapped = TAXONOMY_MAP.get(kw)
                    if mapped:
                        themes.add(mapped['theme'])
                        if mapped.get('group'):
                            groups.add(mapped['group'])
                            if kw != mapped['group']: sub_themes.add(kw)
                processed_row['theme'] = ",".join(sorted(list(themes)))
                processed_row['theme_group'] = ",".join(sorted(list(groups)))
                processed_row['sub_theme'] = ",".join(sorted(list(sub_themes)))
                processed_row['notes'] = "MANUAL_REPROCESSED"
                print("    - ✅ Regeneration successful.")
            else:
                print("    - ❌ Regeneration failed. Staging original row.")
                error_messages.append("REGEN_ERROR: AI call failed.")
                validation_passed = False

        try:
            trigger_obj = json.loads(processed_row['astrological_trigger_json'])
            # --- ADD THIS ONE LINE ---
            trigger_obj = pre_validate_json_fix(trigger_obj)
            validate(instance=trigger_obj, schema=master_json_schema)
            # Update the processed_row in case the object was fixed
            processed_row['astrological_trigger_json'] = json.dumps(trigger_obj)
        except json.JSONDecodeError:
            error_messages.append("JSON_ERROR: Not valid JSON.")
            validation_passed = False
        except ValidationError as e:
            print(f"\n  -> Pre-validation failed for row {index}. Attempting AI correction...")
            correction_prompt = get_json_correction_prompt(trigger_obj, e)
            corrected_obj = call_ai_model(correction_prompt)
            if corrected_obj:
                try:
                    validate(instance=corrected_obj, schema=master_json_schema)
                    print("    - ✅ AI correction successful.")
                    processed_row['astrological_trigger_json'] = json.dumps(corrected_obj)
                    processed_row['notes'] = f"{processed_row.get('notes', '')} | AI_CORRECTED".strip(' |')
                except ValidationError as e2:
                    print("    - ❌ AI correction failed re-validation.")
                    error_messages.append(f"SCHEMA_ERROR (AI_FIX_FAILED): {e2.message}")
                    validation_passed = False
            else:
                error_messages.append(f"SCHEMA_ERROR: {e.message}")
                validation_passed = False

        invalid_tags = validate_thematic_tags(pd.Series(processed_row), valid_thematic_tags)
        if invalid_tags:
            error_messages.append(f"THEME_ERROR: Invalid tags: {', '.join(invalid_tags)}")
            validation_passed = False

        if not validation_passed:
            processed_row['notes'] = " | ".join(error_messages)
            processed_row['status'] = 'Staged'
        else:
            processed_row['status'] = 'Passed'
            for col in ['raw_text', 'ai_summary']:
                if col in processed_row:
                    processed_row[col] = clean_text(processed_row[col])
            if "AI_CORRECTED" not in processed_row.get('notes', '') and "MANUAL_REPROCESSED" not in processed_row.get('notes', ''):
                processed_row['notes'] = ''
        
        all_processed_rows.append(processed_row)

    if all_processed_rows:
        final_df = pd.DataFrame(all_processed_rows)
        final_df['last_updated'] = datetime.now().isoformat()

        # --- NEW DEDUPLICATION STEP ADDED HERE ---
        print(f"\n🔎 Performing final deduplication based on 'Raw Text' and 'JSON'...")
        # First, we need to get the final column names before renaming
        dedupe_subset = ['raw_text', 'astrological_trigger_json']
        original_len = len(final_df)
        final_df.drop_duplicates(subset=dedupe_subset, keep='first', inplace=True)
        new_len = len(final_df)
        print(f"  -> Removed {original_len - new_len} duplicate rows.")
        # --- END OF DEDUPLICATION STEP ---
        
        #final_df['fact_id'] = ''
        final_column_order = [
            'fact_id', 'topic', 'raw_text', 'ai_summary', 'astrological_trigger_json',
            'theme', 'theme_group', 'sub_theme', 'source_name', 'source_page',
            'status', 'last_updated', 'notes'
        ]
        
        for col in final_column_order:
            if col not in final_df.columns:
                final_df[col] = ''
        final_df = final_df[final_column_order]

        final_df.rename(columns={
            'fact_id': 'Fact ID', 'topic': 'Topic', 'raw_text': 'Raw Text',
            'ai_summary': 'AI Summary', 'astrological_trigger_json': 'JSON',
            'theme': 'Theme', 'theme_group': 'Group', 'sub_theme': 'Sub-Theme',
            'source_name': 'Source', 'source_page': 'Page', 'status': 'Status',
            'last_updated': 'Last Updated', 'notes': 'Notes'
        }, inplace=True)
        
        #output_filename = "interpretations.final.csv"
        # Create a unique output name based on the input file
        output_filename = f"cleaned_{latest_file}"
        save_df(final_df, CLEANED_DATA_PATH, output_filename)

        passed_count = (final_df['Status'] == 'Passed').sum()
        staged_count = (final_df['Status'] == 'Staged').sum()

        print("\n✅ V4 Clean process complete.")
        print(f"  -> Saved {len(final_df)} total rows.")
        print(f"  -> {passed_count} rows have 'Passed' status.")
        print(f"  -> {staged_count} rows have been 'Staged' for review.")
    else:
        print("ℹ️ No rows were processed.")

if __name__ == "__main__":
    main()