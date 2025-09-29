# v4_trusted_classifier_async.py (with Chunking)

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
from v2_kb_utils import CANONICAL_SCHEMAS, TAXONOMY

# --- CONFIGURATION ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = genai.GenerativeModel('gemini-1.5-pro-latest')

PROCESSED_DATA_PATH = "kb_pipeline_v4_trusted/processed"
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
AI_CONCURRENT_REQUESTS = 25

# --- (All helper and stage functions from the previous version remain here without change) ---
def generate_fact_id(text_to_hash: str) -> str:
    return hashlib.sha256(text_to_hash.encode('utf-8')).hexdigest()

TAXONOMY_MAP = {}
for theme, groups in TAXONOMY.items():
    for group, sub_themes in groups.items():
        for sub_theme in sub_themes:
            TAXONOMY_MAP[sub_theme] = {'theme': theme, 'group': group}
        TAXONOMY_MAP[group] = {'theme': theme, 'group': group}
    TAXONOMY_MAP[theme] = {'theme': theme, 'group': None}

def save_df(df: pd.DataFrame, path: str, filename: str):
    full_path = os.path.join(path, filename)
    df.to_csv(full_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Successfully saved {len(df)} rows to '{full_path}'.")

def get_pdf_text_chunk(pdf_path: str, start_page: int, end_page: int) -> Optional[str]:
    full_text_chunk = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            actual_end_page = min(end_page, len(pdf.pages))
            for i in range(start_page - 1, actual_end_page):
                page_num = i + 1
                page = pdf.pages[i]
                text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                cleaned_text = ftfy.fix_text(text)
                full_text_chunk += f"\n\n--- PAGE {page_num} ---\n\n{cleaned_text}"
        return full_text_chunk
    except Exception as e:
        print(f"❌ Error reading PDF file: {e}")
        return None

def get_json_fix_prompt(broken_text: str) -> str:
    return f"""
The following text is not valid JSON. Please fix any syntax errors and return only the corrected, valid JSON object. Do not change the content.
**BROKEN TEXT:**
---
{broken_text}
---
**CORRECTED JSON RESPONSE:**
"""

def get_extraction_prompt(text_chunk: str, schema_name: str) -> str:
    return f"""
You are an astrological text analysis engine. Your ONLY task is to find all passages in the provided text chunk that are relevant to the schema: **{schema_name}**.
Your output MUST be a JSON object with a single key `"{schema_name}"`, containing a list of objects. Each object must have two keys: `raw_text` and `source_page`.
---
**Text Chunk**:
\"\"\"
{text_chunk}
\"\"\"
---
**JSON Response:**
"""

def stage1_extract_and_scaffold(text_chunk: str, source_name: str, chunk_start_page: int) -> pd.DataFrame:
    scaffolded_rows = []
    # This function no longer needs a big print header, the main loop will handle it.
    for schema_name in tqdm(CANONICAL_SCHEMAS.keys(), desc=f"  - Processing Schemas for p{chunk_start_page}...", leave=False):
        prompt = get_extraction_prompt(text_chunk, schema_name)
        result = None
        try:
            response = MODEL.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
            json_text = response.text.strip().replace("```json", "").replace("```", "")
            result = json.loads(json_text)
        except json.JSONDecodeError:
            print(f"\n    - ⚠️ Invalid JSON for schema '{schema_name}' in chunk starting at p{chunk_start_page}. Attempting AI self-correction...")
            fix_prompt = get_json_fix_prompt(json_text)
            try:
                fix_response = MODEL.generate_content(fix_prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
                fixed_json_text = fix_response.text.strip().replace("```json", "").replace("```", "")
                result = json.loads(fixed_json_text)
                print(f"      - ✅ Self-correction successful for '{schema_name}'.")
            except Exception as e:
                print(f"      - ❌ Self-correction failed. Creating placeholder for '{schema_name}'.")
                with open(f"FAILED_RESPONSE_{source_name}_p{chunk_start_page}_{schema_name}.txt", "w") as f:
                    f.write(json_text)
                id_input_string = f"{source_name}-ERROR-{chunk_start_page}-{schema_name}"
                fact_id = generate_fact_id(id_input_string)
                scaffold = CANONICAL_SCHEMAS[schema_name]['attributes'].copy()
                for key in scaffold: scaffold[key] = None
                scaffold['type'] = schema_name
                scaffolded_rows.append({
                    'fact_id': fact_id,
                    'raw_text': f"CRITICAL_ERROR: AI response for '{schema_name}' in page chunk starting at {chunk_start_page} was unreadable. See FAILED_RESPONSE_{source_name}_p{chunk_start_page}_{schema_name}.txt",
                    'source_page': chunk_start_page, 'schema_type': schema_name,
                    'json_scaffold': json.dumps(scaffold), 'topic': '',
                    'source_name': source_name, 'last_updated': datetime.now().isoformat()
                })
                continue
        try:
            if result and schema_name in result and isinstance(result[schema_name], list):
                for fact in result[schema_name]:
                    raw_text = fact.get('raw_text', '')
                    source_page = fact.get('source_page', 0)
                    id_input_string = f"{source_name}-{source_page}-{schema_name}-{raw_text}"
                    fact_id = generate_fact_id(id_input_string)
                    scaffold = CANONICAL_SCHEMAS[schema_name]['attributes'].copy()
                    for key in scaffold: scaffold[key] = None
                    scaffold['type'] = schema_name
                    scaffolded_rows.append({
                        'fact_id': fact_id, 'raw_text': raw_text, 'source_page': source_page,
                        'schema_type': schema_name, 'json_scaffold': json.dumps(scaffold),
                        'topic': '', 'source_name': source_name, 'last_updated': datetime.now().isoformat()
                    })
        except Exception as e:
             print(f"  - ⚠️ Error processing successfully parsed JSON for schema '{schema_name}': {e}")
    return pd.DataFrame(scaffolded_rows)

async def async_process_row(task_payload: Dict, semaphore: asyncio.Semaphore) -> Dict:
    index = task_payload['index']
    row = task_payload['row']
    async with semaphore:
        prompt = get_filler_prompt(row['raw_text'], row['json_scaffold'])
        try:
            response = await MODEL.generate_content_async(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
            json_text = response.text.strip().replace("```json", "").replace("```", "")
            ai_response_obj = json.loads(json_text)
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
        except Exception as e:
            print(f"  - ⚠️ Error during async Stage 2 for row {index}: {e}")
            return {'index': index, 'astrological_trigger_json': row['json_scaffold'],
                    'ai_summary': 'Error during generation.', 'theme': '', 'theme_group': '', 'sub_theme': ''}

async def stage2_async_process_facts(scaffold_df: pd.DataFrame) -> pd.DataFrame:
    if scaffold_df.empty: return scaffold_df
    semaphore = asyncio.Semaphore(AI_CONCURRENT_REQUESTS)
    tasks_to_run = []
    for index, row in scaffold_df.iterrows():
        task_payload = {'index': index, 'row': row.to_dict()}
        tasks_to_run.append(async_process_row(task_payload, semaphore))
    results = await async_tqdm.gather(*tasks_to_run, desc=f"  - Processing {len(scaffold_df)} facts...", leave=False)
    results_df = pd.DataFrame(results).set_index('index')
    final_df = pd.concat([scaffold_df, results_df], axis=1)
    return final_df

def get_filler_prompt(raw_text: str, json_scaffold: str) -> str:
    # This function remains unchanged
    taxonomy_str = json.dumps(TAXONOMY, indent=2)
    try:
        scaffold_obj = json.loads(json_scaffold)
        schema_definition_str = json.dumps(CANONICAL_SCHEMAS[scaffold_obj['type']]['attributes'], indent=2)
    except:
        schema_definition_str = "Could not parse schema."
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

# --- MAIN ORCHESTRATOR ---
async def main():
    """The main entry point, now with chunking logic."""
    PDF_FILE = "source_material/books/Gemini-2.pdf"
    BOOK_TITLE = "GeminiAI2"
    START_PAGE = 1
    END_PAGE = 97 # Process 100 pages total
    CHUNK_SIZE = 15 # Process in chunks of 15 pages

    print(f"🚀 Starting V4 Async Ingestion with Chunking for '{BOOK_TITLE}'...")
    
    all_chunks_df = pd.DataFrame()

    # The new chunking loop
    for i in tqdm(range(START_PAGE, END_PAGE + 1, CHUNK_SIZE), desc="Processing Chunks"):
        chunk_start = i
        chunk_end = min(i + CHUNK_SIZE - 1, END_PAGE)
        
        print(f"\n▶️ Processing pages {chunk_start}-{chunk_end}...")
        
        text_chunk = get_pdf_text_chunk(PDF_FILE, chunk_start, chunk_end)

        if text_chunk:
            scaffold_df = stage1_extract_and_scaffold(text_chunk, source_name=BOOK_TITLE, chunk_start_page=chunk_start)
            
            if not scaffold_df.empty:
                chunk_final_df = await stage2_async_process_facts(scaffold_df)
                all_chunks_df = pd.concat([all_chunks_df, chunk_final_df], ignore_index=True)
                
                # Save progress after each chunk
                output_filename = f"processed_themes_{BOOK_TITLE}_p{START_PAGE}-{END_PAGE}.csv"
                save_df(all_chunks_df, PROCESSED_DATA_PATH, output_filename)
            else:
                print(f"  -> No facts found in chunk {chunk_start}-{chunk_end}.")
        else:
            print(f"  -> Could not read text from chunk {chunk_start}-{chunk_end}.")
    
    print("\n\n🎉 All chunks processed.")
    if not all_chunks_df.empty:
        final_columns = [
            'fact_id', 'raw_text', 'source_page', 'source_name', 'last_updated',
            'schema_type', 'topic', 'theme', 'theme_group', 'sub_theme',
            'astrological_trigger_json', 'ai_summary'
        ]
        all_chunks_df = all_chunks_df[final_columns]
        
        output_filename = f"processed_themes_{BOOK_TITLE}_p{START_PAGE}-{END_PAGE}.csv"
        print(f"\nFinal consolidated file saved: '{os.path.join(PROCESSED_DATA_PATH, output_filename)}'")
    else:
        print("No data was generated from any chunk.")

if __name__ == '__main__':
    asyncio.run(main())