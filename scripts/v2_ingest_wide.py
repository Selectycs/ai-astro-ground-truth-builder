import pandas as pd
import pdfplumber
import google.generativeai as genai
import os
import json
import re
import ftfy
from dotenv import load_dotenv
import time
import hashlib
from typing import List, Dict, Optional
from datetime import datetime

# --- Import KB Utilities ---
from v2_kb_utils import (
    V2_ANNOTATED_SCHEMA,
    RAW_DATA_PATH,
    generate_id,
    save_df,
    SCHEMA_KEYWORDS
)

# --- CONFIGURATION & INITIALIZATION ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = genai.GenerativeModel('gemini-1.5-pro-latest')
CACHE_DIR = "kb_pipeline_v2/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

SCHEMA_KEYWORD_TEXT = "\n".join([f"- {schema}: {', '.join(keywords[:5])}..." for schema, keywords in SCHEMA_KEYWORDS.items()])

def clean_and_normalize_text(text: str) -> str:
    if not isinstance(text, str): return ""
    try:
        text = text.encode('latin-1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    text = ftfy.fix_text(text)
    return text

# --- YOUR DEFINITIVE PROMPT FUNCTION ---
def get_system_instructions() -> str:
    """Creates the definitive system instructions for the annotation-focused ingest process."""
    return f"""
    You are a hyper-precise Vedic Astrology knowledge extraction and annotation engine.
    Your sole job is to analyze text, break it down into atomic facts, and output a structured JSON line for each fact with rich metadata. You must process facts in the exact order they appear.

    **CRITICAL RULES:**
    1.  **ATOMICITY:** Each JSON object MUST represent a single, fundamental astrological idea.
    2.  **SEQUENCE:** You MUST output JSON objects in the exact same sequence as the facts appear in the source text.
    3.  **CONTEXT FROM HEADERS:** The primary context for an interpretation is usually set by the most recent heading.
    4.  **KEYWORD JUSTIFICATION:** You can only assign a `schema_type` if it is justified by astrological keywords from the list below, found in either the sentence itself or its governing header.
    5.  **REPHRASE ONLY:** When creating the `interpretation_text`, you MUST preserve the exact astrological meaning of the source. DO NOT ADD any new astrological concepts or information that is not explicitly present in the source sentence.
    6.  **PROSE HANDLING:** If a sentence is clearly not an astrological interpretation (like 'Chapter 5' or 'Conclusion'), you MUST assign it the schema_type: ["prose"].

    **SCHEMA & KEYWORD DEFINITIONS:**
    {SCHEMA_KEYWORD_TEXT}

    **YOUR TASKS & OUTPUT DEFINITIONS:**
    For each atomic fact, generate a JSON object with the following keys:

    1.  `raw_text`: The exact, verbatim sentence or clause from the source text.
    2.  `interpretation_text`: The `raw_text` rephrased to be clearer and more user-friendly.
    3.  `header`: The text of the most recent preceding heading.
    4.  `schema_type`: A JSON array of strings, determined by the "KEYWORD JUSTIFICATION" rule. If no keywords match, return `["unstructured"]`.
    5.  `prior_type`: A JSON array of strings representing the `schema_type` of the immediately preceding fact. For the very first fact, this MUST be an empty array `[]`.
    6.  `is_conceptual_start`: Boolean (true/false). Set to `true` if this fact is the FIRST statement for a new astrological concept. Set to `false` for subsequent facts that elaborate on that same concept.

    **EXAMPLE OF PERFECT EXECUTION:**
    ---
    **Example Input Text:**
    "Lord of 1 in House 3
    ● Interpretation: Creates a courageous and ambitious individual. They excel in arts or writing."

    **Correct JSONL Output:**
    {{"raw_text": "Creates a courageous and ambitious individual.", "interpretation_text": "The native becomes a courageous and ambitious individual.", "header": "Lord of 1 in House 3", "schema_type": ["lordship"], "prior_type": [], "is_conceptual_start": true, "mark_for_deletion": false}}
    {{"raw_text": "They excel in arts or writing.", "interpretation_text": "They often find success in creative fields like the arts or writing.", "header": "Lord of 1 in House 3", "schema_type": ["lordship"], "prior_type": ["lordship"], "is_conceptual_start": false, "mark_for_deletion": false}}
    ---
    """

def get_user_request(page_text: str, book_title: str) -> str:
    return f"Analyze the following text from the book \"{book_title}\" and extract all astrological facts according to the rules I have provided.\n\n**TEXT TO ANALYZE:**\n{page_text}"

def has_keywords(text: str, keywords_dict: Dict) -> bool:
    if not isinstance(text, str): return False
    for schema_keywords in keywords_dict.values():
        for keyword in schema_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                return True
    return False

def extract_facts_from_text(page_text: str, book_title: str, use_cache: bool = True) -> Optional[str]:
    text_hash = hashlib.sha256(page_text.encode('utf-8')).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{text_hash}.json")
    if use_cache and os.path.exists(cache_path):
        print("          - CACHE HIT. Using stored response.")
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f).get("response")
    print("          - CACHE MISS. Calling Gemini API...")
    system_instructions = get_system_instructions()
    user_request = get_user_request(page_text, book_title)
    try:
        response = MODEL.generate_content(
            [system_instructions, user_request],
            generation_config=genai.types.GenerationConfig(temperature=0.0)
        )
        response_text = response.text
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({"response": response_text}, f, ensure_ascii=False)
        return response_text
    except Exception as e:
        print(f"          - ⚠️ Gemini API Error: {e}.")
        return None

# --- ROBUST PARAGRAPH SPLITTING FUNCTION ---
def split_into_paragraphs(text: str) -> List[str]:
    paragraphs = []
    current_paragraph = []
    for line in text.split('\n'):
        if line.strip() == '':
            if current_paragraph:
                paragraphs.append("\n".join(current_paragraph))
                current_paragraph = []
        else:
            current_paragraph.append(line)
    if current_paragraph:
        paragraphs.append("\n".join(current_paragraph))
    return paragraphs

# --- CORE PDF PROCESSOR ---
def process_pdf_run(pdf_path: str, book_title: str, source_type: str, start_page: int, end_page: int, use_cache: bool) -> List[Dict]:
    run_interpretations: List[Dict] = []
    paragraph_counter = 0

    with pdfplumber.open(pdf_path) as pdf:
        start_index = start_page - 1
        end_index = min(end_page, len(pdf.pages))
        pages_to_process = pdf.pages[start_index:end_index]

        for i, page in enumerate(pages_to_process):
            current_page_num = start_index + i + 1
            print(f"   -> Processing Page {current_page_num}...")
            
            raw_page_text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
            page_text = clean_and_normalize_text(raw_page_text)

            if len(page_text.strip()) < 20:
                print("        - Skipping page (not enough content).")
                continue

            paragraphs = split_into_paragraphs(page_text)
            print(f"      - Found {len(paragraphs)} paragraphs to process on this page.")

            for para_text in paragraphs:
                if len(para_text) < 10: continue
                
                paragraph_counter += 1
                
                extracted_data = extract_facts_from_text(para_text, book_title, use_cache=use_cache)

                if extracted_data:
                    cleaned_data = extracted_data.strip().replace("```jsonl", "").replace("```json", "").replace("```", "")
                    sentence_in_para_counter = 0
                    for line in cleaned_data.strip().split('\n'):
                        try:
                            fact_json = json.loads(line)
                            raw_text = fact_json.get('raw_text', '')
                            if not raw_text: continue

                            sentence_in_para_counter += 1
                            fact_id = generate_id(f"{book_title}_{current_page_num}_{paragraph_counter}_{raw_text}")

                            full_row = {
                                "fact_id": fact_id,
                                "paragraph_id": paragraph_counter,
                                "sentence_in_paragraph_id": sentence_in_para_counter,
                                "raw_text": raw_text,
                                "interpretation_text": fact_json.get('interpretation_text', ''),
                                "topic": "",
                                "header": fact_json.get('header', ''),
                                #"mark_for_deletion": fact_json.get('mark_for_deletion', False),
                                "schema_type": json.dumps(fact_json.get('schema_type', [])),
                                "prior_type": json.dumps(fact_json.get('prior_type', [])),
                                "is_conceptual_start": fact_json.get('is_conceptual_start', True),
                                "source_name": book_title,
                                "source_type": source_type,
                                "source_reference": f"Page {current_page_num}",
                                "status": "RAW_ANNOTATED",
                                "last_updated": datetime.now().isoformat(),
                                "notes": "Extracted via v2_ingest_wide.py"
                            }
                            run_interpretations.append(full_row)
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"          - ⚠️ Error parsing JSON line in paragraph {paragraph_counter}: '{line}'")
    return run_interpretations

# --- Main Orchestrator ---
if __name__ == '__main__':
    PDF_FILE = "source_material/books/Gemini - Theme Assessment.pdf"
    BOOK_TITLE = "Gemini"
    SOURCE_TYPE = "AI_ASSISTED_BOOK"
    START_PAGE = 1
    END_PAGE = 1

    print(f"🚀 Starting ingestion for '{BOOK_TITLE}'...")

    all_raw_facts = process_pdf_run(
        pdf_path=PDF_FILE,
        book_title=BOOK_TITLE,
        source_type=SOURCE_TYPE,
        start_page=START_PAGE,
        end_page=END_PAGE,
        use_cache=False 
    )

    if all_raw_facts:
        df = pd.DataFrame(all_raw_facts)
        print(f"\n✅ Ingestion complete. Total raw facts gathered: {len(df)}.")
        
        initial_count = len(df)
        df.drop_duplicates(subset=['raw_text', 'header'], inplace=True, keep='first')
        print(f"✨ Removed {initial_count - len(df)} exact duplicates. Unique facts remaining: {len(df)}.")
        
        print(f"   - Applying final rule for 'is_conceptual_start'...")
        mask_no_keywords = df['raw_text'].apply(lambda x: not has_keywords(x, SCHEMA_KEYWORDS))
        rows_to_correct = len(df[(mask_no_keywords) & (df['is_conceptual_start'] == True)])
        df.loc[mask_no_keywords, 'is_conceptual_start'] = False
        print(f"   - Corrected {rows_to_correct} rows.")

        for col in V2_ANNOTATED_SCHEMA:
            if col not in df.columns:
                df[col] = ''
        df = df[V2_ANNOTATED_SCHEMA]

        output_filename = f"raw_facts_{BOOK_TITLE.replace(' ', '_').lower()}_annotated.csv"
        output_path = os.path.join(RAW_DATA_PATH, output_filename)
        os.makedirs(RAW_DATA_PATH, exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ Successfully saved {len(df)} rows to '{output_path}'.")

    else:
        print("\nℹ️  Processing complete, but no interpretations were extracted.")