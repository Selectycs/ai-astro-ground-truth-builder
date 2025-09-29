# v7_final_classifier.py (With "Fact Explosion" Logic)

import pandas as pd
import google.generativeai as genai
import os
import json
import re
import ftfy
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any
import pdfplumber
from tqdm import tqdm
from datetime import datetime
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
import hashlib
import copy

# --- Import V2 Utilities ---
from v2_kb_utils import (
    CANONICAL_SCHEMAS, SCHEMA_KEYWORDS, TAXONOMY,
    NAKSHATRAS, PLANETS_TO, SIGNS, VARGAS,
    SPECIAL_POINTS, SPECIAL_CHARTS
)

# --- CONFIGURATION ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = genai.GenerativeModel('gemini-1.5-pro-latest')

PROCESSED_DATA_PATH = "kb_pipeline_v8/processed"
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
AI_CONCURRENT_REQUESTS = 25

# --- HELPER FUNCTIONS ---
TAXONOMY_MAP = {}
for theme, groups in TAXONOMY.items():
    for group, sub_themes in groups.items():
        for sub_theme in sub_themes: TAXONOMY_MAP[sub_theme] = {'theme': theme, 'group': group}
        TAXONOMY_MAP[group] = {'theme': theme, 'group': group}
    TAXONOMY_MAP[theme] = {'theme': theme, 'group': None}

def save_df(df: pd.DataFrame, path: str, filename: str):
    full_path = os.path.join(path, filename)
    df.to_csv(full_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Successfully saved {len(df)} rows to '{full_path}'.")

def get_single_page_text(pdf: pdfplumber.PDF, page_num: int) -> Optional[str]:
    try:
        page = pdf.pages[page_num - 1]
        text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
        return ftfy.fix_text(text)
    except IndexError:
        print(f"Warning: Page number {page_num} is out of range for this PDF.")
        return None
    except Exception as e:
        print(f"Error reading page {page_num}: {e}")
        return None

def generate_fact_id(text_to_hash: str) -> str:
    return hashlib.sha256(text_to_hash.encode('utf-8')).hexdigest()

# --- STAGE 1: AI - Context-Rich Text Extraction ---
def get_extraction_prompt(page_text: str, schema_name: str, context_package: Dict) -> str:
    context_str = json.dumps(context_package, indent=2)
    return f"""
You are an expert Vedic Astrologer. Your task is to analyze this **single page** of text and extract all facts related to the **Context Package**.

**YOUR EXPERT BRIEFING:**
Use the following **Context Package** to guide your extraction.

**Context Package:**
---
{context_str}
---

**CRITICAL RULES:**
1.  Analyze only the provided text.
2.  Be Comprehensive: Find all text on this page that matches the content in your briefing.
3.  Group for Context: Group related, consecutive sentences that describe a single astrological fact into a single passage.
4.  Output Format: Your output MUST be a valid JSON object with a single key `"{schema_name}"`, containing a list of objects. Each object must have **only one key**: `raw_text`.
---
**Single Page of Text to Analyze**:
\"\"\"
{page_text}
\"\"\"
---
**JSON Response:**
"""

def get_json_fix_prompt(broken_text: str) -> str:
    return f"""The following text is not valid JSON. Please fix any syntax errors and return only the corrected, valid JSON object. Do not change the content.\n\n**BROKEN TEXT:**\n---\n{broken_text}\n---\n\n**CORRECTED JSON RESPONSE:**"""

# In v8_final_classifier.py

def stage1_extract_text(pdf_path: str, start_page: int, end_page: int) -> List[Dict[str, Any]]:
    # CHANGE 1: The function now returns a list of dictionaries, not just strings.
    all_passages = []
    ENTITY_ATTRIBUTE_MAP = {
        'planet_name': PLANETS_TO, 'source_planet': PLANETS_TO, 'aspected_by_planet': PLANETS_TO,
        'mahadasha_lord': PLANETS_TO, 'antardasha_lord': PLANETS_TO, 'pratyantardasha_lord': PLANETS_TO,
        'ruler': PLANETS_TO, 'sign': SIGNS, 'nakshatra': NAKSHATRAS, 'varga': VARGAS,
        'point_name': SPECIAL_POINTS, 'chart_name': SPECIAL_CHARTS
    }

    print("--- Stage 1: AI Extracting All Relevant Passages ---")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages_to_process = min(end_page, len(pdf.pages))
            tasks = [(pn, sn) for pn in range(start_page, total_pages_to_process + 1) for sn in CANONICAL_SCHEMAS.keys()]
            for page_num, schema_name in tqdm(tasks, desc="Processing Page-Schema Micro-Tasks"):
                if schema_name in ['prose', 'unstructured']: continue
                page_text = get_single_page_text(pdf, page_num)
                if not page_text or not page_text.strip(): continue
                context_package = {"primary_concept": schema_name, "direct_keywords": SCHEMA_KEYWORDS.get(schema_name, [])}
                related_entities = {}
                schema_attributes = CANONICAL_SCHEMAS[schema_name].get('attributes', {}).keys()
                for attr, entity_list in ENTITY_ATTRIBUTE_MAP.items():
                    if attr in schema_attributes:
                        entity_name = attr.replace('_name', '').replace('_lord', '')
                        related_entities[entity_name] = entity_list
                if related_entities: context_package["related_entities"] = related_entities
                prompt = get_extraction_prompt(page_text, schema_name, context_package)
                result = None
                try:
                    response = MODEL.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0, response_mime_type="application/json"))
                    result = json.loads(response.text)
                except Exception:
                    fix_prompt = get_json_fix_prompt(response.text)
                    try:
                        fix_response = MODEL.generate_content(fix_prompt, generation_config=genai.types.GenerationConfig(temperature=0.0, response_mime_type="application/json"))
                        result = json.loads(fix_response.text)
                    except Exception:
                        print(f"\n      - ⚠️ Unrecoverable JSON for '{schema_name}' on page {page_num}. Skipping.")
                        continue
                if result and schema_name in result and isinstance(result[schema_name], list):
                    for fact in result[schema_name]:
                        if fact.get('raw_text'):
                            # CHANGE 2: Append a dictionary with both text and page number.
                            all_passages.append({'raw_text': fact['raw_text'], 'page_num': page_num})
    except FileNotFoundError:
        print(f"❌ CRITICAL ERROR: The PDF file was not found at '{pdf_path}'. Aborting.")
        return []
    except Exception as e:
        print(f"❌ A critical error occurred while opening the PDF: {e}")
    print(f"\n   -> AI extracted {len(all_passages)} total passages (including duplicates).")
    return all_passages

# --- STAGE 2-4: Python - Dedupe, Verify, Scaffold, and Sort ---
# In v8_final_classifier.py

def process_and_scaffold_text(passages: List[Dict[str, Any]], source_name: str, start_page: int, end_page: int) -> pd.DataFrame:
    
    # CHANGE 1: Update deduplication logic to handle dictionaries.
    # We create a dict where the key is the text and the value is the passage object.
    # This automatically removes duplicate texts, keeping the first one encountered.
    unique_passages_dict = {p['raw_text']: p for p in passages}
    unique_passages = sorted(list(unique_passages_dict.values()), key=lambda x: x['raw_text'])

    print(f"\n--- Stages 2-4: Python Processing ---")
    print(f"   -> Deduplicated to {len(unique_passages)} unique passages.")
    
    scaffolded_rows = []
    print("   -> Verifying schemas with keywords and building scaffolds...")
    
    # CHANGE 2: Loop through the list of passage dictionaries.
    for passage_data in unique_passages:
        raw_text = passage_data['raw_text']
        page_num = passage_data['page_num']
        
        matched_schemas = set()
        for schema_name, keywords in SCHEMA_KEYWORDS.items():
            if schema_name in ['prose', 'unstructured']: continue
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', raw_text, re.IGNORECASE):
                    matched_schemas.add(schema_name)
                    break
        if not matched_schemas and 'prose' in CANONICAL_SCHEMAS:
            matched_schemas.add('prose')

        for schema_name in matched_schemas:
            # CHANGE 3: Use the specific page_num for each fact.
            source_reference = f"p{page_num}"
            
            id_input_string = f"{source_name}-{source_reference}-{schema_name}-{raw_text}"
            fact_id = generate_fact_id(id_input_string)
            scaffold = CANONICAL_SCHEMAS[schema_name]['attributes'].copy()
            for key in scaffold: scaffold[key] = None
            scaffold['type'] = schema_name
            scaffolded_rows.append({
                'fact_id': fact_id, 'raw_text': raw_text, 'source_page': source_reference,
                'schema_type': schema_name, 'json_scaffold': json.dumps(scaffold),
                'topic': '', 'source_name': source_name, 'last_updated': datetime.now().isoformat()
            })
    
    df = pd.DataFrame(scaffolded_rows).drop_duplicates(subset=['fact_id'])
    print(f"   -> Generated {len(df)} verified rows.")
    df.sort_values(by='raw_text', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("   -> Sorted all rows by raw_text.")
    return df

# --- STAGE 5: AI - Fills, Summarizes, and Classifies (Asynchronously) ---
# --- MODIFICATION 1: Update the filler prompt ---
def get_filler_prompt(raw_text: str, json_scaffold: str) -> str:
    taxonomy_str = json.dumps(TAXONOMY, indent=2)
    try:
        scaffold_obj = json.loads(json_scaffold)
        schema_definition_str = json.dumps(CANONICAL_SCHEMAS[scaffold_obj['type']]['attributes'], indent=2)
    except: schema_definition_str = "Could not parse schema."
    return f"""
You are a multi-task data processing assistant. Based on the **Source Text**, you must perform three distinct tasks:
1. Fill in the `null` values in the **JSON Template**.
2. Write a summary and add astrological inferences.
3. Select relevant classification keywords from the **Taxonomy**.

Your output MUST be a single JSON object with four keys: "completed_json", "summary", "inferred_context", and "classification_keywords".
**TASK 1: FILL JSON - CRITICAL RULES:**
- **ZERO INFERENCE:** Fill values ONLY from the **Source Text**.
- **STRICT DATA TYPING:** Adhere to the data types in the Schema Definition (e.g., `10` for "integer", `["fame", "power"]` for "list[string]").
- **MULTIPLE VALUES:** If the Source Text mentions multiple entities for a single attribute (e.g., "5th, 7th, and 9th houses" for an `aspected_house` attribute), you MUST return them as a JSON list (e.g., `[5, 7, 9]`).
**TASK 2: CREATE SUMMARY - RULES:**
- `summary`: Rephrase the **Source Text** into a clear, concise summary.
- `inferred_context`: Add brief astrological inferences not in the text. If none, use an empty string.
**TASK 3: CLASSIFY THEMES - RULES:**
- `classification_keywords`: From the **Taxonomy**, select all relevant keywords (prioritizing the most specific `sub_theme` or `theme_group` levels) that describe the **Source Text**. Return a list of strings.
---
**REFERENCE - Schema Definition**: {schema_definition_str}
**REFERENCE - Taxonomy**: {taxonomy_str}
---
**Source Text**: \"\"\"{raw_text}\"\"\"
---
**JSON Template**: {json_scaffold}
---
**JSON Response (with four keys):**
"""

# --- MODIFICATION 2: Add the "Fact Explosion" helper function ---
def explode_fact(original_row: Dict[str, Any], completed_json: Dict[str, Any], ai_summary: str, themes: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Checks for lists in the completed_json and "explodes" the row into multiple atomic facts.
    """
    explosion_key = None
    explosion_list = []

    # Find a key with a list value to explode
    for key, value in completed_json.items():
        if isinstance(value, list) and len(value) > 1:
            explosion_key = key
            explosion_list = value
            break # Handle one explosion per fact for simplicity

    # If no explosion is needed, return the original row as a single-item list
    if not explosion_key:
        return [{
            'fact_id': original_row['fact_id'],
            'raw_text': original_row['raw_text'],
            'source_page': original_row['source_page'],
            'source_name': original_row['source_name'],
            'last_updated': original_row['last_updated'],
            'schema_type': original_row['schema_type'],
            'topic': original_row['topic'],
            'theme': themes['theme'],
            'theme_group': themes['theme_group'],
            'sub_theme': themes['sub_theme'],
            'astrological_trigger_json': json.dumps(completed_json),
            'ai_summary': ai_summary
        }]

    # If explosion is needed, create a new row for each item in the list
    exploded_rows = []
    for item in explosion_list:
        new_json = copy.deepcopy(completed_json)
        new_json[explosion_key] = item
        
        # Create a new unique fact_id for the atomic fact
        id_input_string = f"{original_row['source_name']}-{original_row['source_page']}-{original_row['schema_type']}-{original_row['raw_text']}-{str(item)}"
        new_fact_id = generate_fact_id(id_input_string)

        exploded_rows.append({
            'fact_id': new_fact_id,
            'raw_text': original_row['raw_text'],
            'source_page': original_row['source_page'],
            'source_name': original_row['source_name'],
            'last_updated': datetime.now().isoformat(),
            'schema_type': original_row['schema_type'],
            'topic': original_row['topic'],
            'theme': themes['theme'],
            'theme_group': themes['theme_group'],
            'sub_theme': themes['sub_theme'],
            'astrological_trigger_json': json.dumps(new_json),
            'ai_summary': ai_summary
        })
    return exploded_rows

# --- MODIFICATION 3: Update async_process_row to return a list ---
async def async_process_row(task_payload: Dict, semaphore: asyncio.Semaphore) -> List[Dict]:
    row = task_payload['row']
    async with semaphore:
        prompt = get_filler_prompt(row['raw_text'], row['json_scaffold'])
        try:
            response = await MODEL.generate_content_async(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0, response_mime_type="application/json"))
            ai_response_obj = json.loads(response.text)
            
            completed_json = ai_response_obj.get('completed_json', json.loads(row['json_scaffold']))
            summary = ai_response_obj.get('summary', '')
            inferred_context = ai_response_obj.get('inferred_context', '')
            ai_summary = summary
            if inferred_context: ai_summary += f" (Additional Context: {inferred_context})"
            
            keywords = ai_response_obj.get('classification_keywords', [])
            all_themes, all_groups, all_sub_themes = set(), set(), set()
            for keyword in keywords:
                mapped = TAXONOMY_MAP.get(keyword)
                if mapped:
                    all_themes.add(mapped['theme'])
                    group = mapped.get('group')
                    if group:
                        all_groups.add(group)
                        if keyword != group: all_sub_themes.add(keyword)

            if not all_themes:
                all_themes.add('other'); all_groups.add('Other'); all_sub_themes.add('other')
            elif not all_groups:
                all_groups.add('Other'); all_sub_themes.add('other')
            elif not all_sub_themes:
                all_sub_themes.add('other')
            
            themes = {
                'theme': ",".join(sorted(list(all_themes))),
                'theme_group': ",".join(sorted(list(all_groups))),
                'sub_theme': ",".join(sorted(list(all_sub_themes)))
            }
            
            # Use the new explode_fact function
            return explode_fact(row, completed_json, ai_summary, themes)

        except Exception as e:
            # On error, return the original row data in a list to maintain type consistency
            return [{
                'fact_id': row['fact_id'], 'raw_text': row['raw_text'], 'source_page': row['source_page'],
                'source_name': row['source_name'], 'last_updated': row['last_updated'], 'schema_type': row['schema_type'],
                'topic': row['topic'], 'theme': 'error', 'theme_group': 'Error', 'sub_theme': 'error',
                'astrological_trigger_json': row['json_scaffold'], 'ai_summary': f'Error during generation: {e}'
            }]

# --- MODIFICATION 4: Update stage5 to handle a list of lists ---
async def stage5_async_process_facts(scaffold_df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Stage 5: AI Filling, Summarizing & Classifying (Asynchronously) ---")
    if scaffold_df.empty: return pd.DataFrame()
    
    semaphore = asyncio.Semaphore(AI_CONCURRENT_REQUESTS)
    tasks_to_run = []
    for _, row in scaffold_df.iterrows():
        task_payload = {'row': row.to_dict()}
        tasks_to_run.append(async_process_row(task_payload, semaphore))
    
    # gather() will now return a list of lists of dictionaries
    results_list_of_lists = await async_tqdm.gather(*tasks_to_run, desc=f"   - Processing {len(scaffold_df)} facts...")
    
    # Flatten the list of lists into a single list of dictionaries
    flattened_results = [item for sublist in results_list_of_lists for item in sublist]
    
    if not flattened_results:
        print("   -> No results were generated after processing.")
        return pd.DataFrame()
        
    final_df = pd.DataFrame(flattened_results)
    print(f"   -> 'Fact Explosion' resulted in {len(final_df)} atomic facts.")
    return final_df


# --- MAIN ORCHESTRATOR ---
async def main():
    PDF_FILE = "source_material/books/Houses.pdf"
    BOOK_TITLE = "GeminiAI-Add"
    START_PAGE = 1
    END_PAGE = 22 # A larger chunk for testing

    print(f"🚀 Starting V8 'Extract-Dedupe-Verify-Sort-Fill' Ingestion for '{BOOK_TITLE}'...")
    
    extracted_passages = stage1_extract_text(PDF_FILE, START_PAGE, END_PAGE)
    
    if extracted_passages:
        scaffold_df = process_and_scaffold_text(extracted_passages, source_name=BOOK_TITLE, start_page=START_PAGE, end_page=END_PAGE)
        
        if not scaffold_df.empty:
            final_df = await stage5_async_process_facts(scaffold_df)
            
            if not final_df.empty:
                final_columns = [
                    'fact_id', 'raw_text', 'source_page', 'source_name', 'last_updated',
                    'schema_type', 'topic', 'theme', 'theme_group', 'sub_theme',
                    'astrological_trigger_json', 'ai_summary'
                ]
                # Ensure all columns exist, fill with empty string if not
                for col in final_columns:
                    if col not in final_df.columns:
                        final_df[col] = ''
                final_df = final_df[final_columns]
                
                output_filename = f"processed_v8_{BOOK_TITLE}_p{START_PAGE}-{END_PAGE}.csv"
                save_df(final_df, PROCESSED_DATA_PATH, output_filename)
            else:
                 print("\nℹ️ AI processing did not produce any final rows.")
        else:
            print("\nℹ️ Python verification did not produce any rows to process.")
    else:
        print("\nℹ️ AI extraction did not find any passages.")

if __name__ == '__main__':
    asyncio.run(main())