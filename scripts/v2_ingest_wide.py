# -*- coding: utf-8 -*-
"""
v2_ingest_wide.py (Definitive Gemini Version - Automated Bake-Off & Caching)

This script automates the "Best of N" bake-off process to ensure maximum
completeness and uses a file-based cache to guarantee 100% deterministic
outputs on all subsequent re-runs. It also pre-cleans PDF text.
"""
import pandas as pd
import pdfplumber
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
import time
import hashlib
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime
from thefuzz import fuzz
from ftfy import fix_text

# --- Import KB Utilities ---
from v2_kb_utils import (
    TAXONOMY,
    V2_INTERPRETATIONS_SCHEMA,
    RAW_DATA_PATH,
    generate_id,
    save_df
)

# --- CONFIGURATION & INITIALIZATION ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")
genai.configure(api_key=GEMINI_API_KEY)

GENERATION_CONFIG = genai.types.GenerationConfig(
    temperature=0.0,
    max_output_tokens=8192
)
MODEL = genai.GenerativeModel('gemini-1.5-pro-latest')

# --- CACHING CONFIGURATION ---
CACHE_DIR = "kb_pipeline_v2/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Global Config & TAXONOMY FLATTENING ---
VALID_CLASSIFICATIONS: Set[Tuple[str, str]] = set()
ALL_THEMES: Set[str] = set()
ALL_SUBTHEMES: Set[str] = set()

def setup_taxonomy_helpers():
    """Creates flat sets for themes and classifications for efficient validation."""
    for theme, groups in TAXONOMY.items():
        ALL_THEMES.add(theme)
        for group, sub_themes in groups.items():
            for sub_theme in sub_themes:
                VALID_CLASSIFICATIONS.add((theme, sub_theme))
                ALL_SUBTHEMES.add(sub_theme)

setup_taxonomy_helpers()
flat_taxonomy_list = "\n".join(
    [f"- {theme} -> {sub_theme}" for theme, sub_theme in sorted(list(VALID_CLASSIFICATIONS))]
)

# --- NEW: Text Cleaning Function ---
def clean_pdf_text(text: str) -> str:
    """Fixes unicode errors and normalizes text from PDF extraction."""
    if not isinstance(text, str):
        return ""
    return fix_text(text)

# --- PROMPT BUILDER ---
def get_system_instructions() -> str:
    """Creates the permanent system instructions with the final hyper-strict rules."""
    return f"""
You are a hyper-precise Vedic Astrology knowledge extraction engine. Your sole job is to analyze text and convert it into structured JSON data.

**CRITICAL RULES:**
1.  **ABSOLUTE ATOMICITY:** This is the most important rule. Each JSON object you output MUST represent a single, fundamental astrological rule. Be ruthlessly granular.
    - If a sentence contains multiple clauses or ideas (e.g., 'A strong 12th lord, *or* the placement of Jupiter here...'), create a separate JSON object for each one.
    - **Crucially, this includes separating the definition of a rule from its results.** For example, if one sentence defines how a Yoga is formed and the next describes its effects (like granting wealth or status), you MUST create two separate JSON objects: one for the definition and one for the result.
2.  **CLASSIFY FROM LIST:** You MUST classify each fact. For the "theme" and "sub_themes" keys, you MUST select from the "VALID CLASSIFICATION PATHS" list provided below. The "sub_themes" value must be a JSON array of strings (e.g., ["raja_yogas", "dhana_yogas"]).
3.  **THE GOLDEN RULE OF JSON:** The `astrological_trigger_json` value is critical.
    - It MUST ALWAYS contain the top-level keys: `calculation_context` and `components`.
    - The value for `calculation_context` MUST be a JSON object like `{{"type": "static_placement"}}`. It MUST NOT be a simple string.
4.  **STRICT JSONL OUTPUT FORMAT:** You MUST NOT output anything other than one valid JSON object per line. Do not use markdown, code fences, or any conversational text.

**VALID CLASSIFICATION PATHS:**
{flat_taxonomy_list}

**OUTPUT FORMAT:**
`{{"theme": "...", "sub_themes": ["...", "..."], "interpretation_summary_raw": "...", "interpretation_text": "...", "interpretation_summary_ai": "...", "astrological_trigger_json": {{...}}}}`
"""

def get_user_request(page_text: str, book_title: str) -> str:
    return f"""
Analyze the following text from the book "{book_title}" and extract all astrological facts according to the rules I have provided.

**TEXT TO ANALYZE:**
{page_text}
"""

# --- API Wrapper with Caching ---
def extract_facts_from_text(page_text: str, book_title: str, use_cache: bool = True) -> Optional[str]:
    text_hash = hashlib.sha256(page_text.encode('utf-8')).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{text_hash}.json")

    if use_cache and os.path.exists(cache_path):
        print("    - CACHE HIT. Using stored response.")
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f).get("response")

    print("    - CACHE MISS. Calling Gemini API...")
    system_instructions = get_system_instructions()
    user_request = get_user_request(page_text, book_title)
    
    try:
        response = MODEL.generate_content(
            [system_instructions, user_request],
            generation_config=GENERATION_CONFIG
        )
        response_text = response.text
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({"response": response_text}, f, ensure_ascii=False)
        return response_text
    except Exception as e:
        print(f"  - ⚠️ Gemini API Error: {e}.")
        return None

# --- Classification Validator ---
def is_valid_classification(theme: str, sub_themes_list: List[str]) -> bool:
    if theme not in ALL_THEMES:
        print(f"    - ❌ AI hallucinated a new theme: '{theme}'")
        return False
    if not isinstance(sub_themes_list, list):
        print(f"    - ❌ AI returned sub_themes as a {type(sub_themes_list)}, not a list.")
        return False
    for sub_theme in sub_themes_list:
        if (theme, sub_theme) in VALID_CLASSIFICATIONS:
            continue
        if sub_theme in ALL_SUBTHEMES:
            print(f"    - ✨ Cross-theme classification: AI used '{sub_theme}' under '{theme}'.")
            continue
        print(f"    - ❌ AI hallucinated a new sub-theme: '{theme} -> {sub_theme}'")
        return False
    return True

# --- Core PDF Processor (returns results) ---
def process_pdf_run(pdf_path: str, book_title: str, source_type: str, start_page: Optional[int] = None, end_page: Optional[int] = None, use_cache: bool = True) -> List[Dict]:
    run_interpretations: List[Dict] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            start_index = (start_page - 1) if start_page else 0
            end_index = end_page if end_page else len(pdf.pages)
            pages_to_process = pdf.pages[start_index:end_index]
            
            for i, page in enumerate(pages_to_process):
                current_page_num = start_index + i + 1
                print(f"  -> Processing Page {current_page_num}...")
                page_text = page.extract_text()
                page_text = clean_pdf_text(page_text)

                if not page_text or len(page_text.strip()) < 50:
                    continue
                
                extracted_data = extract_facts_from_text(page_text, book_title, use_cache=use_cache)

                if extracted_data:
                    for line in extracted_data.strip().split('\n'):
                        if not line.strip().startswith('{'):
                            continue
                        try:
                            fact_json = json.loads(line)
                            theme = fact_json.get('theme', '')
                            sub_themes = fact_json.get('sub_themes', [])
                            if not is_valid_classification(theme, sub_themes):
                                continue
                            trigger_json_obj = fact_json.get('astrological_trigger_json', {})
                            trigger_json_str = json.dumps(trigger_json_obj)
                            text = fact_json.get('interpretation_text', '')
                            fact_id = generate_id(f"{trigger_json_str}{text}")
                            interpretation_group = generate_id(trigger_json_str)
                            
                            # --- FIX: Add placeholder fields for the new schema columns ---
                            run_interpretations.append({
                                "fact_id": fact_id,
                                "astrological_trigger_json": trigger_json_str,
                                "interpretation_summary_raw": fact_json.get('interpretation_summary_raw', ''),
                                "interpretation_summary_ai": fact_json.get('interpretation_summary_ai', ''),
                                "interpretation_text": text,
                                "theme": theme,
                                "sub_theme": ",".join(sub_themes),
                                "interpretation_group": interpretation_group,
                                "source_name": book_title,
                                "source_type": source_type,
                                "source_reference": f"Page {current_page_num}",
                                "status": "RAW",
                                "confidence_score": 0.75, # Base score, will be updated in reconcile
                                "primary_fact_id": "",
                                "fallback_tags": "[]",
                                "embedding_vector": "",
                                "schema_version": "", # Placeholder
                                "conflict_status": "", # Placeholder
                                "last_updated": datetime.now().isoformat(),
                                "notes": f"Extracted via v2_ingest_wide.py"
                            })
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"    - ⚠️ Error parsing JSON line on page {current_page_num}: {e}")
                time.sleep(1)
    except Exception as e:
        print(f"❌ An unexpected error occurred during a run: {e}")
    return run_interpretations

# --- CLI Orchestrator for "Best of N" Runs ---
if __name__ == '__main__':
    PDF_FILE = "source_material/books/Gemini - Theme Assessment.pdf"
    BOOK_TITLE = "Gemini"
    SOURCE_TYPE = "AI"
    START_PAGE = 1
    END_PAGE = 33
    
    NUM_RUNS = 3
    USE_CACHE_FOR_BAKEOFF = False

    print(f"🚀 Starting 'Best of {NUM_RUNS}' ingestion for '{BOOK_TITLE}'...")
    
    all_runs_interpretations: List[Dict] = []
    for i in range(NUM_RUNS):
        print("\n" + "="*20 + f" Starting Run {i+1}/{NUM_RUNS} " + "="*20)
        
        run_results = process_pdf_run(
            pdf_path=PDF_FILE, 
            book_title=BOOK_TITLE, 
            source_type=SOURCE_TYPE, 
            start_page=START_PAGE, 
            end_page=END_PAGE,
            use_cache=USE_CACHE_FOR_BAKEOFF
        )
        print(f"🏁 Run {i+1} finished, extracted {len(run_results)} facts.")
        all_runs_interpretations.extend(run_results)
        time.sleep(5)

    if all_runs_interpretations:
        print(f"\n✅ All runs complete. Total interpretations gathered: {len(all_runs_interpretations)}.")
        
        df = pd.DataFrame(all_runs_interpretations)
        df.drop_duplicates(subset=['interpretation_group', 'interpretation_text'], inplace=True, keep='first')
        
        print(f"✨ Unique interpretations after de-duplication: {len(df)}.")

        df = df[V2_INTERPRETATIONS_SCHEMA]
        output_filename = f"raw_facts_{BOOK_TITLE.replace(' ', '_').lower()}_combined.csv"
        save_df(df, RAW_DATA_PATH, output_filename)
    else:
        print("\nℹ️  Processing complete, but no new interpretations were extracted across all runs.")