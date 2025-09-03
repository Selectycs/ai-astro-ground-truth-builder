import pandas as pd
import pdfplumber
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
import time
from kb_utils import TAXONOMY, CORE_CONCEPT_SCHEMAS, save_interpretations_to_kb, save_core_concepts_to_kb

# --- LOAD API KEY ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-latest')

def extract_facts_from_text(page_text, book_title):
    """
    Uses Gemini to extract both thematic interpretations and highly structured
    core concept definitions from a book page.
    """
    taxonomy_json = json.dumps(TAXONOMY, indent=2)
    core_schemas_json = json.dumps(CORE_CONCEPT_SCHEMAS, indent=2)

    prompt = f"""
    You are a hyper-precise Vedic Astrology knowledge engineering expert. Your task is to analyze the following text from "{book_title}" and break it down into the smallest possible individual facts, following these critical rules.

    **CRITICAL RULES:**
    1.  **UNPACK COMPARISONS:** If the text makes a comparison (e.g., "X for this, versus Y for that"), you MUST create a separate atomic row for the rule related to X and another for the rule related to Y.
    2.  **CHOOSE THE SINGLE BEST CLASSIFICATION:** A single astrological rule might seem to fit multiple `Sub_Themes`. You MUST choose only the ONE `Sub_Theme` that is the most representative of the rule's primary focus. Do NOT create duplicate rows for a single rule by classifying it into multiple different sub-themes.
    3.  **ONE FACT PER ROW:** If a sentence mentions multiple astrological factors (e.g., "Sun, Mars, or Saturn in the 10th..."), you MUST create a separate row for each factor.
    4.  **FULLY UNPACK ALL LISTS:** A single sentence might contain a list of rules separated by semicolons or commas (e.g., "...Venus for arts; Mercury for commerce; Jupiter for law..."). You MUST create a separate, complete, atomic row for EVERY SINGLE ITEM in such a list.
    5.  **INTERPRETATION QUALITY:** The `Interpretation_Text` MUST be a complete, descriptive sentence explaining the astrological principle clearly.
    6.  **SUMMARY QUALITY:** The `AI_Astro_Summary` MUST be a single, clear, plain-language sentence that directly answers the question "What does this likely indicate?".
    7.  **STANDARDIZE FOUNDATION POINTS:** The `Foundation_Point` must be a concise, standardized trigger (e.g., "Sun in 10th House").
    8.  **STRUCTURED REFERENCES:** You must generate a `Chart_Refs_JSON` object representing the foundation point.

    ---
    **OUTPUT FORMATS & TASKS**

    **TASK 1: Thematic Interpretations**
    Format: THEME||[Theme]||[Fact_Group]||[Sub_Themes]||[Foundation_Point]||[Interpretation_Text]||[AI_Astro_Summary]||[Chart_Refs_JSON]

    **TASK 2: Core Concept Definitions (Extract Only)**
    Instruction: If the text explicitly defines a concept (e.g., "The Sun is the karaka for the soul"), extract it.
    Crucial Guardrail: Do NOT generate a definition from your own knowledge. If the text does not define the term, do not create a CORE_CONCEPT entry.
    Format: CORE_CONCEPT||[Concept_Group]||[JSON_string_using_schema_fields]
    
    ---
    **THEMATIC TAXONOMY:**
    {taxonomy_json}

    **CORE CONCEPT SCHEMAS:**
    {core_schemas_json}
    ---
    
    TEXT TO ANALYZE:
    {page_text}
    """
    try:
        generation_config = genai.types.GenerationConfig(
            temperature=0.1,  # Low temp for consistency
            max_output_tokens=8192
        )
        response = model.generate_content(prompt, generation_config=generation_config)
        return response.text
    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        return "NONE"


def process_pdf_multitheme(pdf_path, book_title, source_type="BOOK", start_page=None, end_page=None):
    """
    Processes a PDF, extracts all facts, and saves them to the correct knowledge base.
    """
    print(f"Starting unified processing for '{book_title}'...")
    
    confidence = 0.7 if source_type == "BOOK" else 0.5

    # --- DEFINE DATA HOLDERS BEFORE THE LOOP ---
    all_thematic_interpretations = []
    all_concepts_by_group = {}

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Found {total_pages} pages in the document.")

        for i, page in enumerate(pdf.pages):
            current_page_num = i + 1
            if start_page and current_page_num < start_page:
                continue
            if end_page and current_page_num > end_page:
                print(f"Reached end page {end_page}. Stopping processing.")
                break
            
            print(f"  Processing page {current_page_num} of {total_pages}...")
            page_text = page.extract_text()

            if page_text and len(page_text) > 50:
                extracted_data = extract_facts_from_text(page_text, book_title)
                
                if extracted_data and "NONE" not in extracted_data:
                    for line in extracted_data.strip().split('\n'):
                        
                        if line.startswith("THEME||") and line.count('||') == 7:
                            _, theme, fact_group, sub_themes, foundation_point, interpretation_text, ai_summary, chart_refs_json = [p.strip() for p in line.split('||')]
                            
                            # --- NEW VALIDATION BLOCK STARTS HERE ---
                            try:
                                # This will raise a KeyError if the theme or fact_group is invalid
                                valid_sub_themes = TAXONOMY[theme][fact_group]
                                
                                # This checks if the specific sub_theme exists in the list
                                if sub_themes not in valid_sub_themes:
                                     raise ValueError(f"Sub-theme '{sub_themes}' not found in Fact Group '{fact_group}'")
                            
                            except (KeyError, ValueError) as e:
                                print(f"  - ⚠️ WARNING: Invalid classification path from AI: '{theme} -> {fact_group} -> {sub_themes}'. Skipping fact. Reason: {e}")
                                continue # Skip to the next line in the AI output
                            # --- NEW VALIDATION BLOCK ENDS HERE ---

                            all_thematic_interpretations.append({
                                'theme': theme, 'fact_group': fact_group, 'sub_themes': sub_themes,
                                'foundation_point': foundation_point, 'interpretation_text': interpretation_text,
                                'ai_summary': ai_summary, 'chart_refs_json': chart_refs_json,
                                'confidence': confidence,
                                'reference': f"Page {current_page_num}"
                            })

                        elif line.startswith("CORE_CONCEPT||"):
                            parts = line.split('||', 2)
                            if len(parts) == 3:
                                _, concept_group, json_string = parts
                                concept_group = concept_group.strip()
                                try:
                                    concept_data = json.loads(json_string)
                                    concept_data['reference'] = f"Page {current_page_num}"
                                    if concept_group not in all_concepts_by_group:
                                        all_concepts_by_group[concept_group] = []
                                    all_concepts_by_group[concept_group].append(concept_data)
                                except json.JSONDecodeError:
                                    print(f"    - Warning: Could not decode JSON on page {current_page_num}: {json_string}")
                
                time.sleep(1.5)

    # --- SAVE ALL ACCUMULATED DATA ONCE, AFTER THE LOOP ---
    save_interpretations_to_kb(all_thematic_interpretations, book_title, source_type)
    save_core_concepts_to_kb(all_concepts_by_group, book_title, source_type)

    print(f"Unified processing complete for '{book_title}'.")


if __name__ == '__main__':
    PDF_FILE = "source_material/books/Gemini - Theme Assessment.pdf"
    BOOK_TITLE = "Gemini AI"
    SOURCE_TYPE = "AI"
    
    # --- SET PAGE RANGE HERE ---
    # To process the entire book, set both to None
    START_PAGE = 11
    END_PAGE = 11

    # To process just a single page (e.g., page 2)
    # START_PAGE = 2
    # END_PAGE = 2
    
    if os.path.exists(PDF_FILE):
        process_pdf_multitheme(
            pdf_path=PDF_FILE, 
            book_title=BOOK_TITLE, 
            source_type=SOURCE_TYPE,
            start_page=START_PAGE,
            end_page=END_PAGE
        )
    else:
        print(f"Error: PDF file not found at '{PDF_FILE}'. Please update the path.")