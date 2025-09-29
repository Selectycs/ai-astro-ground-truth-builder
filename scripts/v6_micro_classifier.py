# v6_micro_batch_classifier.py (Corrected "Expert Briefing" Logic)

import pandas as pd
import google.generativeai as genai
import os
import json
import re
import ftfy
from dotenv import load_dotenv
from typing import Dict, List, Optional
import pdfplumber
from tqdm import tqdm
from datetime import datetime
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
import hashlib

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

PROCESSED_DATA_PATH = "kb_pipeline_v6/processed"
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
    except Exception as e:
        print(f"Error reading page {page_num}: {e}")
        return None

def generate_fact_id(text_to_hash: str) -> str:
    return hashlib.sha256(text_to_hash.encode('utf-8')).hexdigest()

# --- STAGE 1: Page-by-Page, Schema-by-Schema Extraction ---
def get_extraction_prompt(page_text: str, schema_name: str, context_package: Dict) -> str:
    """
    CORRECTED PROMPT
    Creates the context-rich prompt for a single page and single schema.
    """
    context_str = json.dumps(context_package, indent=2)
    return f"""
You are an expert Vedic Astrologer. Your task is to analyze this **single page** of text and extract all text related to the **Primary Astrological Concept**.

**YOUR EXPERT BRIEFING:**
Use the following **Context Package** to guide your extraction. It contains the primary concept, direct keywords, and related entities you must look for. Your goal is to be exhaustive.

**Context Package:**
---
{context_str}
---

**CRITICAL RULES:**
1.  **Analyze only the provided text.** Do not use outside knowledge.
2.  **Be Comprehensive:** Find all text on this page that matches the concepts in your briefing.
3.  **Group for Context:** Group related, consecutive sentences that describe a single concept into a single passage.
4.  **Output Format:** Your output MUST be a valid JSON object with a single key `"{schema_name}"`, containing a list of objects. Each object must have **only one key**: `raw_text`.

---
**Single Page of Text to Analyze**:
\"\"\"
{page_text}
\"\"\"
---
**JSON Response:**
"""

def get_json_fix_prompt(broken_text: str) -> str:
    return f"""
The following text is not valid JSON. Please fix any syntax errors and return only the corrected, valid JSON object. Do not change the content.
**BROKEN TEXT:**
---
{broken_text}
---
**CORRECTED JSON RESPONSE:**
"""

def stage1_extract_and_scaffold(pdf_path: str, start_page: int, end_page: int, source_name: str) -> pd.DataFrame:
    """
    CORRECTED FUNCTION
    Loops page-by-page, then schema-by-schema, building the context package
    for each AI call to extract and scaffold facts.
    """
    scaffolded_rows = []
    ENTITY_ATTRIBUTE_MAP = {
        'planet_name': PLANETS_TO, 'source_planet': PLANETS_TO, 'aspected_by_planet': PLANETS_TO,
        'mahadasha_lord': PLANETS_TO, 'antardasha_lord': PLANETS_TO, 'pratyantardasha_lord': PLANETS_TO,
        'ruler': PLANETS_TO, 'sign': SIGNS, 'nakshatra': NAKSHATRAS, 'varga': VARGAS,
        'point_name': SPECIAL_POINTS, 'chart_name': SPECIAL_CHARTS
    }

    print("--- Stage 1: Page-by-Page, Schema-by-Schema Extraction ---")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages_to_process = min(end_page, len(pdf.pages))
            
            tasks = [(pn, sn) for pn in range(start_page, total_pages_to_process + 1) for sn in CANONICAL_SCHEMAS.keys()]
            
            for page_num, schema_name in tqdm(tasks, desc="Processing Page-Schema Micro-Tasks"):
                page_text = get_single_page_text(pdf, page_num)
                if not page_text or not page_text.strip(): continue

                # --- THIS IS THE CORRECTED CONTEXT-BUILDING LOGIC ---
                schema_def = CANONICAL_SCHEMAS[schema_name]
                context_package = {
                    "primary_concept": schema_name,
                    "direct_keywords": SCHEMA_KEYWORDS.get(schema_name, [])
                }
                related_entities = {}
                schema_attributes = schema_def.get('attributes', {}).keys()
                for attr, entity_list in ENTITY_ATTRIBUTE_MAP.items():
                    if attr in schema_attributes:
                        entity_name = attr.replace('_name', '').replace('_lord', '')
                        related_entities[entity_name] = entity_list
                if related_entities:
                    context_package["related_entities"] = related_entities
                # --- END OF CORRECTION ---
                
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
                        print(f"\n    - ⚠️ Unrecoverable JSON for '{schema_name}' on page {page_num}. Skipping.")
                        continue
                
                if result and schema_name in result and isinstance(result[schema_name], list):
                    for fact in result[schema_name]:
                        raw_text = fact.get('raw_text', '')
                        if not raw_text: continue
                        
                        id_input_string = f"{source_name}-{page_num}-{schema_name}-{raw_text}"
                        fact_id = generate_fact_id(id_input_string)
                        scaffold = CANONICAL_SCHEMAS[schema_name]['attributes'].copy()
                        for key in scaffold: scaffold[key] = None
                        scaffold['type'] = schema_name
                        scaffolded_rows.append({
                            'fact_id': fact_id, 'raw_text': raw_text, 'source_page': page_num,
                            'schema_type': schema_name, 'json_scaffold': json.dumps(scaffold),
                            'topic': '', 'source_name': source_name, 'last_updated': datetime.now().isoformat()
                        })
    except Exception as e:
        print(f"❌ A critical error occurred while opening the PDF: {e}")

    df = pd.DataFrame(scaffolded_rows).drop_duplicates(subset=['fact_id'])
    print(f"\n  -> Generated {len(df)} total unique rows with scaffolds.")
    return df

# --- STAGE 2: AI - Fills, Summarizes, and Classifies (Asynchronously) ---
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

async def async_process_row(task_payload: Dict, semaphore: asyncio.Semaphore) -> Dict:
    index = task_payload['index']
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
            return {'index': index, 'astrological_trigger_json': json.dumps(completed_json), 'ai_summary': ai_summary,
                    'theme': ",".join(sorted(list(all_themes))), 'theme_group': ",".join(sorted(list(all_groups))),
                    'sub_theme': ",".join(sorted(list(all_sub_themes)))}
        except Exception:
            return {'index': index, 'astrological_trigger_json': row['json_scaffold'],
                    'ai_summary': 'Error during generation.', 'theme': '', 'theme_group': '', 'sub_theme': ''}

async def stage2_async_process_facts(scaffold_df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Stage 2: AI Filling, Summarizing & Classifying (Asynchronously) ---")
    if scaffold_df.empty: return scaffold_df
    semaphore = asyncio.Semaphore(AI_CONCURRENT_REQUESTS)
    tasks_to_run = []
    for index, row in scaffold_df.iterrows():
        task_payload = {'index': index, 'row': row.to_dict()}
        tasks_to_run.append(async_process_row(task_payload, semaphore))
    results = await async_tqdm.gather(*tasks_to_run, desc=f"  - Processing {len(scaffold_df)} facts...")
    results_df = pd.DataFrame(results).set_index('index')
    final_df = pd.concat([scaffold_df, results_df], axis=1)
    return final_df

# --- MAIN ORCHESTRATOR ---
async def main():
    PDF_FILE = "source_material/books/Gemini-2.pdf"
    BOOK_TITLE = "BPHS_GCS"
    START_PAGE = 1
    END_PAGE = 5

    print(f"🚀 Starting V6 'Micro-Batch' Ingestion for '{BOOK_TITLE}'...")
    
    scaffold_df = stage1_extract_and_scaffold(PDF_FILE, START_PAGE, END_PAGE, source_name=BOOK_TITLE)
    
    if not scaffold_df.empty:
        final_df = await stage2_async_process_facts(scaffold_df)
        
        final_columns = [
            'fact_id', 'raw_text', 'source_page', 'source_name', 'last_updated',
            'schema_type', 'topic', 'theme', 'theme_group', 'sub_theme',
            'astrological_trigger_json', 'ai_summary'
        ]
        for col in final_columns:
            if col not in final_df.columns:
                final_df[col] = ''
        final_df = final_df[final_columns]
        
        output_filename = f"processed_v6_{BOOK_TITLE}_p{START_PAGE}-{END_PAGE}.csv"
        save_df(final_df, PROCESSED_DATA_PATH, output_filename)
    else:
        print("\nℹ️ Stage 1 did not produce any rows to process.")

if __name__ == '__main__':
    asyncio.run(main())