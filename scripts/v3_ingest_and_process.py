# v4_iterative_verify_fill.py

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

# --- Import V2 Utilities ---
# Using your established utils file as the single source of truth
from v2_kb_utils import CANONICAL_SCHEMAS, SCHEMA_KEYWORDS

# --- CONFIGURATION ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = genai.GenerativeModel('gemini-1.5-pro-latest')

PROCESSED_DATA_PATH = "kb_pipeline_v4/processed"
CACHE_DIR = "kb_pipeline_v4/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# --- HELPER FUNCTIONS ---
def save_df(df: pd.DataFrame, path: str, filename: str):
    """Saves a DataFrame to a specified path."""
    full_path = os.path.join(path, filename)
    df.to_csv(full_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Successfully saved {len(df)} rows to '{full_path}'.")

def get_pdf_text_chunk(pdf_path: str, start_page: int, end_page: int) -> Optional[str]:
    """Reads a chunk of pages from a PDF and returns it as a single string."""
    full_text_chunk = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            start_idx = start_page - 1
            end_idx = min(end_page, len(pdf.pages))
            for i in range(start_idx, end_idx):
                page_num = i + 1
                page = pdf.pages[i]
                text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                cleaned_text = ftfy.fix_text(text)
                full_text_chunk += f"\n\n--- PAGE {page_num} ---\n\n{cleaned_text}"
        return full_text_chunk
    except Exception as e:
        print(f"❌ Error reading PDF file: {e}")
        return None

# --- STAGE 1: AI - Suggest Passages and Schemas ---
def get_suggestion_prompt(text_chunk: str, primary_schema: str, all_schema_names: List[str]) -> str:
    """Prompt for Stage 1: Find text for a primary schema and suggest others."""
    return f"""
You are an astrological text analysis engine. Your task is to find all passages in the provided text chunk that are relevant to the **Primary Schema**.

**CRITICAL RULES:**
1.  Your primary focus is to find text related to: **{primary_schema}**.
2.  For each passage you find, you MUST also suggest a comma-separated list of any other schemas from the **Master Schema List** that could also apply to that same text.
3.  Group related, consecutive sentences into a single passage.
4.  Your output MUST be a JSON object with a single key `"{primary_schema}"`, which contains a list of objects.
5.  Each object in the list must have three keys: `raw_text`, `source_page`, and `ai_suggested_schemas`.

**Master Schema List**: {', '.join(all_schema_names)}

---
**Text Chunk**:
\"\"\"
{text_chunk}
\"\"\"
---

**JSON Response:**
"""

def stage1_extract_and_suggest(text_chunk: str) -> List[Dict]:
    """Iteratively calls the AI to find text passages and suggest schemas."""
    candidate_facts = []
    all_schema_names = list(CANONICAL_SCHEMAS.keys())
    
    print("--- Stage 1: AI Extracting Passages and Suggesting Schemas ---")
    for schema_name in tqdm(all_schema_names, desc="Stage 1 - Schemas"):
        prompt = get_suggestion_prompt(text_chunk, schema_name, all_schema_names)
        try:
            response = MODEL.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
            json_text = response.text.strip().replace("```json", "").replace("```", "")
            result = json.loads(json_text)
            
            if schema_name in result and isinstance(result[schema_name], list):
                for fact in result[schema_name]:
                    fact['primary_schema_found_for'] = schema_name # For debugging
                    candidate_facts.append(fact)

        except Exception as e:
            print(f"  - ⚠️ Error during Stage 1 for schema '{schema_name}': {e}")
            continue
            
    # Deduplicate based on raw_text, keeping the first instance
    df = pd.DataFrame(candidate_facts)
    df.drop_duplicates(subset=['raw_text'], keep='first', inplace=True)
    print(f"  -> Found {len(df)} unique candidate passages.")
    return df.to_dict('records')


# --- STAGE 2: Python - Verify Schemas & Build Scaffolds ---
def stage2_verify_and_scaffold(candidate_facts: List[Dict]) -> pd.DataFrame:
    """Uses Python keyword matching to verify schemas and create JSON scaffolds."""
    scaffolded_rows = []
    print("\n--- Stage 2: Python Verifying Schemas and Building Scaffolds ---")
    
    for fact in tqdm(candidate_facts, desc="Stage 2 - Verifying"):
        raw_text = fact.get('raw_text', '')
        # Keep the AI's suggestion for comparison, as requested
        ai_suggestion = fact.get('ai_suggested_schemas', '')
        
        verified_schemas = set()
        for schema_name, keywords in SCHEMA_KEYWORDS.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', raw_text, re.IGNORECASE):
                    verified_schemas.add(schema_name)
                    break 
        
        for schema_name in verified_schemas:
            scaffold = CANONICAL_SCHEMAS[schema_name]['attributes'].copy()
            # Set all values to None as placeholders
            for key in scaffold:
                scaffold[key] = None
            scaffold['type'] = schema_name # Pre-fill the type

            scaffolded_rows.append({
                'raw_text': raw_text,
                'source_page': fact.get('source_page'),
                'ai_suggested_schemas': ai_suggestion,
                'json_scaffold': json.dumps(scaffold)
            })
            
    print(f"  -> Generated {len(scaffolded_rows)} rows with verified JSON scaffolds.")
    return pd.DataFrame(scaffolded_rows)


# --- STAGE 3: AI - Fill Verified Scaffolds ---
def get_filler_prompt(raw_text: str, json_scaffold: str) -> str:
    """Prompt for Stage 3: Fill in the null values of a JSON template."""
    return f"""
You are a data entry assistant. Your only job is to fill in the `null` values in the provided **JSON Template** based on the **Source Text**.

**RULES:**
1.  Only fill in values that are explicitly mentioned or strongly implied in the text.
2.  It is acceptable and expected to leave values as `null` if the information is not present.
3.  Do not change the keys or the structure of the JSON. Return only the completed JSON.

---
**Source Text**: 
\"\"\"
{raw_text}
\"\"\"

---
**JSON Template**:
{json_scaffold}

---
**Completed JSON Response:**
"""

def stage3_fill_scaffolds(scaffold_df: pd.DataFrame) -> pd.DataFrame:
    """Iterates through the scaffolded DataFrame and uses AI to fill the JSON."""
    filled_jsons = []
    print("\n--- Stage 3: AI Filling JSON Values ---")
    
    for index, row in tqdm(scaffold_df.iterrows(), total=len(scaffold_df), desc="Stage 3 - Filling"):
        prompt = get_filler_prompt(row['raw_text'], row['json_scaffold'])
        try:
            response = MODEL.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
            json_text = response.text.strip().replace("```json", "").replace("```", "")
            filled_jsons.append(json_text)
        except Exception as e:
            print(f"  - ⚠️ Error during Stage 3 for row {index}: {e}")
            filled_jsons.append(row['json_scaffold']) # Append original scaffold on error

    scaffold_df['astrological_trigger_json'] = filled_jsons
    return scaffold_df

# --- MAIN ORCHESTRATOR ---
if __name__ == '__main__':
    PDF_FILE = "source_material/books/Gemini - Theme Assessment.pdf"
    BOOK_TITLE = "BPHS_GCS"
    START_PAGE = 1
    END_PAGE = 1 # Process 1 page for this run

    print(f"🚀 Starting V4 'Iterative-Verify-Fill' Ingestion for '{BOOK_TITLE}'...")
    
    # Get the raw text from the PDF
    text_chunk = get_pdf_text_chunk(PDF_FILE, START_PAGE, END_PAGE)

    if text_chunk:
        # STAGE 1
        candidate_facts = stage1_extract_and_suggest(text_chunk)
        
        if candidate_facts:
            # STAGE 2
            scaffold_df = stage2_verify_and_scaffold(candidate_facts)
            
            if not scaffold_df.empty:
                # STAGE 3
                final_df = stage3_fill_scaffolds(scaffold_df)
                
                # Final Cleanup
                final_df = final_df[['raw_text', 'source_page', 'ai_suggested_schemas', 'astrological_trigger_json']]
                
                # Save the final results
                output_filename = f"processed_{BOOK_TITLE}_p{START_PAGE}-{END_PAGE}.csv"
                save_df(final_df, PROCESSED_DATA_PATH, output_filename)
            else:
                print("\nℹ️ Stage 2 did not produce any rows to process.")
        else:
            print("\nℹ️ Stage 1 did not find any candidate passages.")
    else:
        print("\nℹ️ Could not read text from PDF. Aborting.")