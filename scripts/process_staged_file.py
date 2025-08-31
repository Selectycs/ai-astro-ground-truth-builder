# scripts/process_staged_file.py
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
import argparse
from kb_utils import TAXONOMY, VALID_THEMES, save_facts_to_kb

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-pro-latest')

def process_text_file(file_path):
    """Reads a text file and uses Gemini to extract and classify facts."""
    print(f"Processing staged file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    taxonomy_json = json.dumps(TAXONOMY, indent=2)
    prompt = f"""
    You are an expert in Vedic Astrology and a knowledge engineering assistant.
    Your task is to analyze the following text, extract every distinct astrological rule, and classify it with extreme precision using the provided Taxonomy.

    THE TAXONOMY:
    {taxonomy_json}

    INSTRUCTIONS:
    For each rule you find, you MUST follow these steps:
    1.  Identify the primary 'Theme' from the top-level keys in the Taxonomy.
    2.  Identify the most relevant 'Fact_Group' from the keys within that Theme.
    3.  Identify one or more relevant 'Sub_Themes' from the list within that Fact_Group. If multiple apply, separate them with a comma.
    4.  If a rule clearly belongs to a specific Fact_Group but not any specific Sub_Theme, use that group's 'other' Sub_Theme.
    5.  If and only if the rule does not fit into any of the specific Fact_Groups, classify it under the 'Other' Fact_Group.
    6.  Extract the 'Foundation_Point' (the astrological combination).
    7.  Extract the 'Interpretation_Text' (the result or meaning).

    Structure your final output in the following format, using '||' as a separator:
    Theme||Fact_Group||Sub_Themes||Foundation_Point||Interpretation_Text
    
    If the text contains no valid astrological rules, output the single word "NONE".
    
    Here is the text to analyze:
    ---
    {raw_text}
    ---
    """
    
    try:
        response = model.generate_content(prompt)
        extracted_data = response.text
        
        if "NONE" in extracted_data:
            print("No valid facts found in the file.")
            return

        facts_by_theme = {}
        for line in extracted_data.strip().split('\n'):
            if '||' in line and line.count('||') == 4:
                theme, fact_group, sub_themes, foundation_point, interpretation_text = [p.strip() for p in line.split('||')]
                
                if theme in VALID_THEMES:
                    if theme not in facts_by_theme:
                        facts_by_theme[theme] = []
                    
                    facts_by_theme[theme].append({
                        'fact_group': fact_group,
                        'sub_themes': sub_themes,
                        'foundation_point': foundation_point,
                        'interpretation_text': interpretation_text
                    })
        
        source_name = os.path.basename(file_path)
        source_type = "AI_QUERY" if "ai_queries" in file_path else "MANUAL_NOTE"
        save_facts_to_kb(facts_by_theme, source_name, source_type)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a staged text file.")
    parser.add_argument("filepath", type=str, help="The path to the text file in the staging_area.")
    args = parser.parse_args()
    
    if not os.path.exists(args.filepath):
        print(f"Error: File not found at '{args.filepath}'")
    else:
        process_text_file(args.filepath)