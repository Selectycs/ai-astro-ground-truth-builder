# v8_semantic_chunker.py (Final Version with Tuned Prompt and Model Selection)

import pandas as pd
import google.generativeai as genai
import os
import json
import ftfy
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any
import pdfplumber
import asyncio
from elasticsearch import Elasticsearch
import math
import hashlib

# --- Import V2 Utilities ---
from v2_kb_utils import CANONICAL_SCHEMAS, CANONICAL_ENTITIES, TAXONOMY

# --- CONFIGURATION ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- MODEL SELECTION ---
# Use 1.5 Pro for cost-effective, high-quality results.
MODEL = genai.GenerativeModel('gemini-2.5-pro')
# OPTIONAL: Uncomment the line below to use a more advanced model for potentially higher accuracy on complex texts.
# MODEL = genai.GenerativeModel('gemini-pro') # Or the specific name for 2.5 Pro when available

PROCESSED_DATA_PATH = "kb_pipeline_v8/tagged"
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# --- MASTER PROMPT TEMPLATE (IMPROVED AND MORE STRICT) ---
MASTER_PROMPT_TEMPLATE = """
You are an expert Vedic Astrologer and a master archivist. Your task is to process the provided astrological text and deconstruct it into a series of self-contained "knowledge chunks". For each chunk, you will generate a rich set of metadata tags that must be based on the reference dictionaries provided.
Your output MUST be a single, valid JSON object: a list where each item contains a "chunk_text" and its corresponding "metadata" object.
---
**REFERENCE DICTIONARIES (Your tools for tagging):**
1. **SCHEMA DICTIONARY:**
{schema_dict_json}
2. **ENTITY DICTIONARY:**
{entity_dict_json}
3. **TAXONOMY DICTIONARY:**
{taxonomy_dict_json}
---
**STEP-BY-STEP INSTRUCTIONS:**
1. **SEGMENT THE TEXT:** Read the entire **Source Text** below. Identify logical breakpoints to create an array of "knowledge chunks". Your final `chunk_text` output should only contain the English-language rule, translation, and interpretation. Exclude any original Sanskrit verses, page numbers, or chapter titles.
2. **GENERATE METADATA (For EACH chunk):** For every chunk you create, generate a "metadata" object with the following keys:
    * **`concept_type` (string):** Classify the *type* of information in the chunk. Your answer MUST be one of the primary keys from the **SCHEMA DICTIONARY** (e.g., "placement", "yoga", "aspect").
    * **`attributes` (object):** Your primary task is to extract all relevant details from the chunk and map them to the corresponding keys defined for the chunk's `concept_type` in the **SCHEMA DICTIONARY**. For example, for a `planet_profile`, you must extract friends and enemies into the `friends` and `enemies` keys. For a `placement`, you must extract the `planet_name` and `house`. If a key from the schema is not mentioned in the text, omit it from the object.
    * **`configurations` (list of objects):** Create a "configurations" list. Each object in this list must represent **one complete and distinct set of conditions** that form the rule or yoga. The keys inside each object must match the attribute keys from the **SCHEMA DICTIONARY**.
    * **`sub_themes` (list of strings):** Your **ONLY** task for thematic classification is to select the most specific `sub_themes` from the **TAXONOMY DICTIONARY** that describe the chunk.
    * **`source_ref` (string):** Use the provided **Source Context** to describe the origin.
---
**SOURCE CONTEXT:**
{source_context}
---
**SOURCE TEXT TO PROCESS:**
\"\"\"
{source_text}
\"\"\"
---
**FINAL JSON RESPONSE (a list of chunk objects):**
"""

# --- HELPER FUNCTIONS (Identical to previous version) ---
def get_chapter_text(pdf_path: str, start_page: int, end_page: int) -> Optional[str]:
    full_text = ""
    print(f"📖 Reading pages {start_page}-{end_page} from '{pdf_path}'...")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            actual_end_page = min(end_page, len(pdf.pages))
            for i in range(start_page - 1, actual_end_page):
                page = pdf.pages[i]
                text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                cleaned_text = ftfy.fix_text(text)
                full_text += f"\n\n--- PAGE {i + 1} ---\n\n{cleaned_text}"
        return full_text.strip()
    except Exception as e:
        print(f"❌ Error reading PDF file: {e}")
        return None

def prepare_context_data() -> Dict[str, Any]:
    taxonomy_map = {}
    for theme, groups in TAXONOMY.items():
        for group, sub_themes in groups.items():
            for sub_theme in sub_themes:
                taxonomy_map[sub_theme] = {'theme': theme, 'group': group}
            taxonomy_map[group] = {'theme': theme, 'group': group}
        taxonomy_map[theme] = {'theme': theme, 'group': None}
    
    return {
        "schema_dict_json": json.dumps(CANONICAL_SCHEMAS, indent=2),
        "entity_dict_json": json.dumps(CANONICAL_ENTITIES, indent=2),
        "taxonomy_dict_json": json.dumps(TAXONOMY, indent=2),
        "taxonomy_map": taxonomy_map
    }

async def call_ai_chunker(prompt: str) -> Optional[List[Dict]]:
    try:
        response = await MODEL.generate_content_async(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"\n- ⚠️ AI call or JSON parsing failed: {e}")
        return None

def index_data_to_es(df: pd.DataFrame, es_client: Elasticsearch, index_name: str):
    print(f"📦 Indexing {len(df)} documents to Elasticsearch index '{index_name}'...")
    try:
        for i, row in df.iterrows():
            doc = row.to_dict()
            doc_cleaned = {}
            for k, v in doc.items():
                if isinstance(v, list):
                    doc_cleaned[k] = v
                elif pd.notna(v):
                    doc_cleaned[k] = v
            chunk_text = doc.get('chunk_text', '')
            unique_id = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
            es_client.index(index=index_name, document=doc_cleaned, id=unique_id)
        print("✅ Elasticsearch indexing successful.")
    except Exception as e:
        print(f"❌ Elasticsearch indexing failed: {e}")

# --- SELF-HEALING BATCH PROCESSOR (Identical to previous version) ---
async def process_batch(pdf_file, book_title, batch_name, start_page, end_page, context_data, es_client, es_index_name, create_csv_backup):
    try:
        source_text = get_chapter_text(pdf_file, start_page, end_page)
        if not source_text:
            print(f"❌ No source text for '{batch_name}'. Skipping.")
            return

        final_prompt = MASTER_PROMPT_TEMPLATE.format(
            schema_dict_json=context_data["schema_dict_json"],
            entity_dict_json=context_data["entity_dict_json"],
            taxonomy_dict_json=context_data["taxonomy_dict_json"],
            source_context=f"{book_title}, {batch_name}",
            source_text=source_text
        )

        ai_response_json = await call_ai_chunker(final_prompt)
        if ai_response_json is None:
            batch_size = end_page - start_page + 1
            if batch_size > 1:
                print(f"↪️ Batch '{batch_name}' failed. Splitting and retrying...")
                mid_page = start_page + batch_size // 2
                await process_batch(pdf_file, book_title, f"{batch_name}-a", start_page, mid_page, context_data, es_client, es_index_name, create_csv_backup)
                await process_batch(pdf_file, book_title, f"{batch_name}-b", mid_page + 1, end_page, context_data, es_client, es_index_name, create_csv_backup)
                return
            else:
                print(f"❌ AI processing failed for single page '{batch_name}'. Cannot split further. Skipping.")
                return

        print("⚙️  Processing AI response and structuring data...")
        processed_rows = []
        for chunk_data in ai_response_json:
            metadata = chunk_data.get('metadata', {})
            attributes = metadata.get('attributes', {})
            configurations_list = metadata.get('configurations', [])
            sub_themes_list = metadata.get('sub_themes', [])
            row = {
                'chunk_text': chunk_data.get('chunk_text'),
                'meta_source_ref': metadata.get('source_ref'),
                'meta_concept_type': metadata.get('concept_type'),
                'attr_name': attributes.get('name')
            }
            if configurations_list:
                row['meta_configurations_json'] = json.dumps(configurations_list)
            else:
                row['meta_configurations_json'] = None
            themes, groups = set(), set()
            taxonomy_map = context_data["taxonomy_map"]
            if isinstance(sub_themes_list, list):
                for sub_theme in sub_themes_list:
                    if sub_theme in taxonomy_map:
                        themes.add(taxonomy_map[sub_theme]['theme'])
                        if taxonomy_map[sub_theme].get('group'):
                            groups.add(taxonomy_map[sub_theme]['group'])
            row['meta_themes'] = list(sorted(themes))
            row['meta_groups'] = list(sorted(groups))
            row['meta_sub_themes'] = sub_themes_list
            processed_rows.append(row)

        if not processed_rows:
            print(f"ℹ️ No rows were processed for '{batch_name}'.")
            return

        final_df = pd.DataFrame(processed_rows)
        print(f"✅ AI processing successful for '{batch_name}'. Generated {len(final_df)} knowledge chunks.")
        index_data_to_es(final_df, es_client, es_index_name)
        if create_csv_backup:
            output_filename = f"tagged_{book_title}_{batch_name}.csv".replace(' ', '_').replace('/', '_')
            full_path = os.path.join(PROCESSED_DATA_PATH, output_filename)
            final_df.to_csv(full_path, index=False, encoding='utf-8-sig')
            print(f"🗒️  Successfully saved CSV backup to '{full_path}'.")
    except Exception as e:
        print(f"❌ An unexpected error occurred while processing '{batch_name}': {e}")

# --- MAIN ORCHESTRATOR (Identical to previous version) ---
async def main():
    PDF_FILE = "source_material/books/BPHS1.pdf"
    BOOK_TITLE = "BPHS1"
    ES_INDEX_NAME = "keshoo_knowledge_base"
    CREATE_CSV_BACKUP = True
    PAGE_BATCH_SIZE = 5

    CHAPTERS_TO_PROCESS = [
        {'name': "Chapter 27 - EVALUATION OF STRENGTHS", 'start_page': 262, 'end_page': 287},
        {'name': "Chapter 28 - ISHTA AND KASHTA BALAS", 'start_page': 288, 'end_page': 291},
        {'name': "Chapter 29 - BHAVA PADAS", 'start_page': 292, 'end_page': 302},
        {'name': "Chapter 30 - UPA PADAS", 'start_page': 302, 'end_page': 309},
        {'name': "Chapter 31 - ARGALA OR PLANETARY INTERYENTION", 'start_page': 310, 'end_page': 315},
        {'name': "Chapter 32 - PLANETARY KARAKATWAS", 'start_page': 315, 'end_page': 325},
        {'name': "Chapter 33 - EFFECTS OF KARAKAMSA", 'start_page': 325, 'end_page': 339},
        {'name': "Chapter 34 - YOGA KARAKAS", 'start_page': 340, 'end_page': 356},
        {'name': "Chapter 35 - NABHASA YOGAS", 'start_page': 357, 'end_page': 365},
        {'name': "Chapter 36 - MANY OTHER YOGAS", 'start_page': 365, 'end_page': 383},
        {'name': "Chapter 37 - LANARYOGAS", 'start_page': 383, 'end_page': 385},
        {'name': "Chapter 38 - SOLARYOGAS", 'start_page': 385, 'end_page': 386},
        {'name': "Chapter 39 - RATAYOGAS", 'start_page': 386, 'end_page': 398},
        {'name': "Chapter 40 - YOGAS FOR ROYAL ASSOCIATION", 'start_page': 398, 'end_page': 401},
        {'name': "Chapter 41 - YOGAS FOR WEALTH", 'start_page': 401, 'end_page': 411},
        {'name': "Chapter 42 - COMBINATIONS FOR PENURY", 'start_page': 412, 'end_page': 415},
        {'name': "Chapter 43 - LONGEVITY", 'start_page': 415, 'end_page': 438},
        {'name': "Chapter 44 - MARAKA ( KILLER) PLANETS", 'start_page': 439, 'end_page': 447},
        {'name': "Chapter 45 - AVASTHAS OF PLANETS", 'start_page': 448, 'end_page': 482},
    ]

    print(f"🚀 Starting V8 Semantic Chunker Pipeline for '{BOOK_TITLE}'...")
    
    try:
        es_client = Elasticsearch(
            hosts=[os.getenv("ELASTIC_URL")],
            api_key=os.getenv("ELASTIC_API_KEY")
        )
        es_client.info()
        print("🔗 Successfully connected to Elasticsearch.")
    except Exception as e:
        print(f"❌ Could not connect to Elasticsearch. Error: {e}")
        return
        
    context_data = prepare_context_data()
    
    for chapter_info in CHAPTERS_TO_PROCESS:
        original_chapter_name = chapter_info['name']
        start_page = chapter_info['start_page']
        end_page = chapter_info['end_page']
        total_pages = end_page - start_page + 1
        num_batches = math.ceil(total_pages / PAGE_BATCH_SIZE)

        print(f"\n{'='*25} PROCESSING CHAPTER: {original_chapter_name} ({total_pages} pages in {num_batches} batches) {'='*25}")

        for i in range(num_batches):
            batch_start_page = start_page + (i * PAGE_BATCH_SIZE)
            batch_end_page = min(start_page + ((i + 1) * PAGE_BATCH_SIZE) - 1, end_page)
            
            batch_name = f"{original_chapter_name} (Part {i+1})" if num_batches > 1 else original_chapter_name
            
            print(f"\n--- Processing Initial Batch: {batch_name} (Pages {batch_start_page}-{batch_end_page}) ---")
            await process_batch(PDF_FILE, BOOK_TITLE, batch_name, batch_start_page, batch_end_page, context_data, es_client, ES_INDEX_NAME, CREATE_CSV_BACKUP)
            
    print(f"\n{'='*25} BATCH PROCESSING COMPLETE {'='*25}")

if __name__ == '__main__':
    asyncio.run(main())
