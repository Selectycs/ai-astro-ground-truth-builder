import pandas as pd
import google.generativeai as genai
import os
import json
import re
import ftfy
from dotenv import load_dotenv
import asyncio
from typing import Dict, List, Any, Optional
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

# --- Import V2 utilities ---
from v2_kb_utils import (
    V2_CANONICAL_SCHEMA, RAW_DATA_PATH, VALIDATED_DATA_PATH, save_df,
    SCHEMA_KEYWORDS, TAXONOMY, CANONICAL_SCHEMAS, CANONICAL_ENTITIES,
    CANONICAL_SCHEMA_EXAMPLES_TEXT
)

# --- CONFIGURATION & INITIALIZATION ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = genai.GenerativeModel('gemini-1.5-flash-latest')
AI_CONCURRENT_REQUESTS = 25

# --- TAXONOMY REVERSE LOOKUP (for fast mapping) ---
TAXONOMY_MAP = {}
for theme, groups in TAXONOMY.items():
    for group, sub_themes in groups.items():
        for sub_theme in sub_themes:
            TAXONOMY_MAP[sub_theme] = {'theme': theme, 'group': group}
        TAXONOMY_MAP[group] = {'theme': theme, 'group': group}
    TAXONOMY_MAP[theme] = {'theme': theme, 'group': None}


# --- HELPER FUNCTIONS ---
def reevaluate_schema_type(row):
    """
    Re-evaluates the schema_type using the 'topic' as the primary context,
    and the 'header' as a fallback.
    """
    context_source = row['topic'] or row['header']
    if pd.isna(context_source) or str(context_source).strip() == '':
        return row['schema_type']
    
    context_text = row['raw_text'] + ' ' + str(context_source)
    matched_schemas = set()
    
    for schema, keywords in SCHEMA_KEYWORDS.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', context_text, re.IGNORECASE):
                matched_schemas.add(schema)
    
    final_schemas = sorted(list(matched_schemas))
    return json.dumps(final_schemas if final_schemas else ["unstructured"])

def get_decomposition_prompt(row: Dict) -> str:
    """Creates a prompt for the 'Decomposer' AI, which splits text into atomic facts."""
    return f"""
You are a text analysis engine. Your sole job is to take a potentially complex sentence and break it down into an array of simple, atomic facts. The `topic` or `header` provides the main subject.

**CRITICAL RULES:**
1.  Your entire output MUST be a single, valid JSON object with one key: `"atomic_facts"`.
2.  The value of `"atomic_facts"` MUST be an array of strings.
3.  Each string in the array must be a complete, self-contained factual sentence.
4.  Use the `topic` or `header` as the subject of the sentences you create.

**Example 1: Splitting into multiple facts**
- Topic: "12th House"
- Raw Text: "Core Significators (Karaka): Saturn (for loss), Ketu (for liberation), Venus (for bed pleasures)."
- Correct Output:
{{
  "atomic_facts": [
    "For the 12th house, Saturn is the karaka for loss.",
    "For the 12th house, Ketu is the karaka for liberation.",
    "For the 12th house, Venus is the karaka for bed pleasures."
  ]
}}

---
**Your Task:**

* **Topic**: "{row['topic'] or row['header']}"
* **Raw Text**: "{row['raw_text']}"

**JSON Response:**
"""

def get_extraction_prompt(atomic_sentence: str, task: Dict) -> str:
    """Creates a prompt for the 'Extractor' AI, asking for the JSON trigger and a list of specific keywords."""
    target_schema_type = task['single_schema_type']
    original_row_data = task['original_row_data']
    schema_definition = CANONICAL_SCHEMAS.get(target_schema_type, CANONICAL_SCHEMAS['unstructured'])
    
    attributes_text = json.dumps(schema_definition['attributes'])
    taxonomy_str = json.dumps(TAXONOMY, indent=2)
    canonical_entities_str = json.dumps(CANONICAL_ENTITIES, indent=2)

    return f"""
You are a data extraction and classification engine. Your job is to create a structured JSON object, identify all relevant theme keywords for a sentence, and create a summary.

**MASTER RULES:**
1.  Your entire response MUST be a single, valid JSON object with the exact keys shown in the `FINAL JSON OUTPUT STRUCTURE`.
2.  The `final_trigger` object MUST perfectly match the structure of the `Target Schema Definition`.
3.  All entity names MUST be normalized using the `CANONICAL_ENTITIES` reference.
4.  From the `TAXONOMY`, you MUST select **all relevant keywords** (prioritizing `sub_theme` or `theme_group` levels) that describe the `atomic_sentence`.
5.  Place these keywords into the `classification_keywords` list. If no specific keywords apply, return an empty list `[]`.
6.  **`summary`**: Write a direct and concise summary (max 20 words) of ONLY the `atomic_sentence`. DO NOT add any new information or inference.
7.  **`inferred_context`**: After summarizing, think about the broader astrological implications. What might this fact imply about the house's affairs or the planet's nature? Write this as a brief, separate string. If no strong implications exist, return an empty string "".

**INPUT DATA:**
* **atomic_sentence**: "{atomic_sentence}"
* **original_topic**: "{original_row_data['topic'] or original_row_data['header']}"
* **target_schema**: "{target_schema_type}"

**REFERENCE DEFINITIONS:**
* **Target Schema Definition**: {attributes_text}
* **CANONICAL_ENTITIES**: {canonical_entities_str}
* **TAXONOMY**: {taxonomy_str}

**FINAL JSON OUTPUT STRUCTURE:**
{{
    "final_trigger": {{}},
    "classification_keywords": ["...", "..."],
    "summary": "...",
    "inferred_context": "..."
}}

**JSON Response:**
"""

def get_theme_prompt_for_missing(row: pd.Series) -> str:
    """Creates a targeted prompt to ask the AI for theme keywords for a single fact."""
    trigger_obj = {}
    try:
        trigger_obj = json.loads(row['astrological_trigger_json'])
    except (json.JSONDecodeError, TypeError):
        trigger_obj = {"error": "Could not parse trigger JSON"}

    taxonomy_str = json.dumps(TAXONOMY, indent=2)

    return f"""
You are an expert astrological classifier. Your sole job is to analyze an astrological fact and select all relevant keywords from the provided TAXONOMY.

**ASTROLOGICAL FACT TO ANALYZE:**
* **Interpretation Text**: "{row['interpretation_text']}"
* **Astrological Trigger**: {json.dumps(trigger_obj, indent=2)}

**REFERENCE TAXONOMY:**
{taxonomy_str}

**CRITICAL RULES:**
1.  Your entire output MUST be a single, valid JSON object with one key: `"classification_keywords"`.
2.  From the `TAXONOMY`, you MUST select all relevant keywords that describe the fact.
3.  If no specific keywords apply, return an empty list `[]`.

**JSON Response:**
"""

def fix_row_themes_with_ai(row: pd.Series) -> Optional[Dict]:
    """Takes a single row with missing themes, calls the AI, and processes the keywords."""
    prompt = get_theme_prompt_for_missing(row)
    try:
        response = MODEL.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.0)
        )
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        ai_json = json.loads(json_text)
        
        keywords = ai_json.get('classification_keywords', [])
        if not keywords:
            return None

        all_themes, all_groups, all_sub_themes = set(), set(), set()
        for keyword in keywords:
            mapped = TAXONOMY_MAP.get(keyword)
            if mapped:
                all_themes.add(mapped['theme'])
                group = mapped.get('group')
                if group:
                    all_groups.add(group)
                    if keyword != group:
                        all_sub_themes.add(keyword)
        
        return {
            'theme': ",".join(sorted(list(all_themes))),
            'theme_group': ",".join(sorted(list(all_groups))),
            'sub_theme': ",".join(sorted(list(all_sub_themes)))
        }
    except Exception as e:
        print(f"    - ⚠️ AI call failed for row. Error: {e}")
        return None

# --- Post-AI Validation & Normalization ---
def _recursive_clean_strings(obj):
    """Recursively traverses a JSON object and applies encoding fixes to all strings."""
    if isinstance(obj, dict):
        return {k: _recursive_clean_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_recursive_clean_strings(elem) for elem in obj]
    if isinstance(obj, str):
        try:
            text = obj.encode('latin-1').decode('utf-8')
            return ftfy.fix_text(text)
        except (UnicodeEncodeError, UnicodeDecodeError):
            return ftfy.fix_text(obj)
    return obj

def _normalize_entity(value: str, entity_type: str) -> Optional[str]:
    """Finds the canonical name for a given alias."""
    if not isinstance(value, str) or not value:
        return None
    entity_map = CANONICAL_ENTITIES.get(entity_type, {})
    for canonical, aliases in entity_map.items():
        for alias in aliases:
            if re.search(r'\b' + re.escape(alias) + r'\b', value, re.IGNORECASE):
                return canonical
    return value.split(' ')[0].split('(')[0].strip()

def validate_and_normalize_ai_output(ai_json: Dict) -> Optional[Dict]:
    """Cleans and normalizes the JSON output from the AI."""
    if not ai_json or not isinstance(ai_json, dict):
        return None

    cleaned_json = _recursive_clean_strings(ai_json)
    trigger_data = cleaned_json.get("final_trigger")

    if trigger_data and isinstance(trigger_data, dict):
        entity_key_map = {
            'planet_name': 'planet', 'source_planet': 'planet', 'mahadasha_lord': 'planet',
            'antardasha_lord': 'planet', 'sign': 'sign', 'group_name': 'house_group',
            'nakshatra': 'nakshatra', 'varga': 'varga', 'varga_name': 'varga',
            'point_name': 'special_point', 'chart_name': 'special_chart'
        }
        for key, entity_type in entity_key_map.items():
            if key in trigger_data and trigger_data[key] is not None:
                trigger_data[key] = _normalize_entity(trigger_data[key], entity_type)
    
    return cleaned_json


# --- ASYNC AI WORKER ---
async def extract_row_async(task: Dict, semaphore: asyncio.Semaphore) -> Dict:
    """Sends a single atomic fact for extraction and classification."""
    async with semaphore:
        prompt = get_extraction_prompt(task['atomic_sentence'], task)
        try:
            response = await MODEL.generate_content_async(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0, max_output_tokens=2048))
            json_text = response.text.strip().replace("```json", "").replace("```", "")
            ai_json = json.loads(json_text)
            
            validated_ai_json = validate_and_normalize_ai_output(ai_json)
            return {'original_task': task, 'ai_feedback': validated_ai_json}
        except Exception as e:
            print(f"      - ❌ Error extracting from sentence: '{task['atomic_sentence'][:50]}...': {e}")
            return {'original_task': task, 'ai_feedback': None}
            
# --- MAIN ORCHESTRATOR ---
async def run_reconciliation_async(df: pd.DataFrame):
    """Orchestrates the new two-step 'Decompose then Extract' process."""
    print("  - Re-evaluating schema types...")
    df['schema_type'] = df.apply(reevaluate_schema_type, axis=1)

    # --- STEP 1: DECOMPOSITION ---
    print("\n🚀 Step 1: Sending rows to Decomposer AI to split compound facts...")
    
    async def decompose_row_async(row: Dict, semaphore: asyncio.Semaphore) -> Dict:
        async with semaphore:
            prompt = get_decomposition_prompt(row)
            try:
                response = await MODEL.generate_content_async(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
                json_text = response.text.strip().replace("```json", "").replace("```", "")
                decomposed_json = json.loads(json_text)
                return {'original_row': row, 'atomic_facts': decomposed_json.get('atomic_facts', [])}
            except Exception:
                return {'original_row': row, 'atomic_facts': [row['raw_text']]}

    semaphore = asyncio.Semaphore(AI_CONCURRENT_REQUESTS)
    tasks = [decompose_row_async(row, semaphore) for row in df.to_dict('records')]
    decomposition_results = await async_tqdm.gather(*tasks, desc="Decomposing Facts")

    # --- EXPANSION LOGIC ---
    print("\n  - Expanding rows based on schemas and decomposed atomic facts...")
    extraction_tasks = []
    for result in decomposition_results:
        original_row = result['original_row']
        atomic_facts = result['atomic_facts']
        
        try:
            schema_types = json.loads(original_row['schema_type'])
        except (json.JSONDecodeError, TypeError):
            schema_types = ["unstructured"]
        if not schema_types: schema_types = ["unstructured"]

        for schema_type in schema_types:
            for fact_text in atomic_facts:
                task_payload = {
                    'atomic_sentence': fact_text,
                    'original_row_data': original_row,
                    'single_schema_type': schema_type
                }
                extraction_tasks.append(task_payload)

    print(f"  - Generated {len(extraction_tasks)} total extraction tasks.")

    # --- STEP 2: EXTRACTION ---
    print("\n🚀 Step 2: Sending atomic facts to Extractor AI for JSON generation...")
    tasks = [extract_row_async(task, semaphore) for task in extraction_tasks]
    extraction_results = await async_tqdm.gather(*tasks, desc="Extracting JSON")

    # --- FINAL ASSEMBLY ---
    print("\n  - Assembling final results...")
    final_rows = []
    fact_counters = {} 

    for result in extraction_results:
        if result.get('ai_feedback'):
            original_task = result['original_task']
            original_row_data = original_task['original_row_data']
            ai_feedback = result['ai_feedback']
            
            new_row = original_row_data.copy()
            trigger_json = ai_feedback.get("final_trigger", {}) 
            
            base_fact_id = str(original_row_data.get('fact_id', ''))
            fact_counters[base_fact_id] = fact_counters.get(base_fact_id, 0) + 1
            current_count = fact_counters[base_fact_id]
            if current_count > 1 or len(extraction_results) > len(df):
                new_row['fact_id'] = f"{base_fact_id}_fact_{current_count}"

            requested_schema = original_task['single_schema_type']
            trigger_json['type'] = requested_schema
            new_row['astrological_trigger_json'] = json.dumps(trigger_json, ensure_ascii=False)
            
            new_row['interpretation_text'] = original_task.get('atomic_sentence', '') 

            # --- This is the requested code block for the enhanced summary ---
            summary_text = ai_feedback.get('summary', '')
            inferred_text = ai_feedback.get('inferred_context', '')

            final_summary = summary_text
            if inferred_text:
                final_summary += f" (Additional Context: {inferred_text})"
            new_row['interpretation_summary_ai'] = final_summary
            # --- End of summary block ---

            new_row['confidence_score'] = 0.95
            new_row['status'] = 'VALIDATED'
            new_row['notes'] = "Reconciled via Decompose-Extract v6"
            
            # --- "Bottom-Up" Theme Logic ---
            all_themes, all_groups, all_sub_themes = set(), set(), set()
            
            for keyword in ai_feedback.get('classification_keywords', []):
                mapped = TAXONOMY_MAP.get(keyword)
                if mapped:
                    all_themes.add(mapped['theme'])
                    group = mapped.get('group')
                    if group:
                        all_groups.add(group)
                        if keyword != group:
                            all_sub_themes.add(keyword)
            
            new_row['theme'] = ",".join(sorted(list(all_themes)))
            new_row['theme_group'] = ",".join(sorted(list(all_groups)))
            new_row['sub_theme'] = ",".join(sorted(list(all_sub_themes)))

            final_rows.append(new_row)

    return pd.DataFrame(final_rows)

def final_theme_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds any rows with empty, null, or whitespace themes after the main 
    process and uses a targeted AI call to fill them in.
    """
    print("\n🚀 Starting final cleanup pass to fill any remaining missing themes...")
    
    is_null = pd.isna(df['theme'])
    is_blank_string = df['theme'].astype(str).str.strip() == ''
    rows_to_process = df.loc[is_null | is_blank_string].copy()
    
    if rows_to_process.empty:
        print("   - ✅ No rows with empty themes found. Skipping.")
        return df
        
    print(f"   - 🔍 Found {len(rows_to_process)} rows with missing themes. Starting targeted AI classification...")
    
    for index, row in tqdm(rows_to_process.iterrows(), total=len(rows_to_process), desc="Fixing Themes"):
        theme_data = fix_row_themes_with_ai(row)
        
        current_notes = str(df.loc[index, 'notes'])
        if current_notes == 'nan': current_notes = '' 
        
        if theme_data and theme_data.get('theme'):
            df.loc[index, 'theme'] = theme_data['theme']
            df.loc[index, 'theme_group'] = theme_data['theme_group']
            df.loc[index, 'sub_theme'] = theme_data['sub_theme']
            
            note_to_add = "Themes added via AI cleanup"
            df.loc[index, 'notes'] = f"{current_notes} | {note_to_add}" if current_notes else note_to_add
        else:
            df.loc[index, 'theme'] = 'other'
            df.loc[index, 'theme_group'] = 'Other'
            df.loc[index, 'sub_theme'] = 'other'
            
            note_to_add = "Default 'other' theme assigned in cleanup"
            df.loc[index, 'notes'] = f"{current_notes} | {note_to_add}" if current_notes else note_to_add
            
    print("   - ✅ Theme cleanup complete.")
    return df

# --- Main Entry Point ---

def main():
    """Main function to run the reconciliation process."""
    print("🚀 Starting V2 Reconciliation Process from Scratch...")
    input_file = os.path.join(RAW_DATA_PATH, "raw_facts_Gemini_annotated.csv")
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file not found at {input_file}")
        return
        
    try:
        df = pd.read_csv(input_file, dtype=str).fillna("")
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return
    
    final_df = asyncio.run(run_reconciliation_async(df))

    if final_df is not None and not final_df.empty:
        
        final_df = final_theme_cleanup(final_df)
        
        print(f"\n🚀 Starting final de-duplication on {len(final_df)} rows...")
        deduped_df = final_df.drop_duplicates(
            subset=['interpretation_text', 'astrological_trigger_json'], keep='first'
        ).reset_index(drop=True)
        print(f"  - Removed {len(final_df) - len(deduped_df)} duplicates.")
        
        for col in V2_CANONICAL_SCHEMA:
            if col not in deduped_df.columns:
                deduped_df[col] = ''
        
        deduped_df = deduped_df[V2_CANONICAL_SCHEMA]
        
        save_df(deduped_df, VALIDATED_DATA_PATH, "interpretations.validated_canonical.csv")
    else:
        print("ℹ️ No rows were produced after reconciliation.")

if __name__ == "__main__":
    main()