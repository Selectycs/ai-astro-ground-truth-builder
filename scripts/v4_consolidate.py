# v4_consolidate.py (Definitive Version)

import pandas as pd
import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Optional, Dict
from tqdm import tqdm

# --- CONFIGURATION & INITIALIZATION ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = genai.GenerativeModel('gemini-1.5-pro-latest')

CLEANED_DATA_PATH = "kb_pipeline_v4_trusted/cleaned"
CONSOLIDATED_DATA_PATH = "kb_pipeline_v4_trusted/consolidated"
os.makedirs(CONSOLIDATED_DATA_PATH, exist_ok=True)

# --- HELPER FUNCTIONS ---
def save_df(df: pd.DataFrame, path: str, filename: str):
    """Saves a DataFrame to a specified path, creating the directory if needed."""
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    df.to_csv(full_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Successfully saved {len(df)} rows to '{full_path}'.")

def merge_unique_values(series: pd.Series) -> str:
    """
    Takes a pandas Series of strings (which may be comma-separated),
    and returns a single, sorted, comma-separated string of unique values.
    (Adapted from your V2 script)
    """
    all_items = {item.strip() for val in series.dropna() for item in str(val).split(',') if item.strip()}
    return ",".join(sorted(list(all_items)))

# --- AI HELPER PROMPT & FUNCTION ---
def get_synthesis_prompt(group_df: pd.DataFrame) -> str:
    """Creates a prompt to synthesize multiple source texts and summaries into one."""
    
    # Create a formatted block of all unique raw texts
    raw_texts_block = "\n".join([f"- {text}" for text in group_df['Raw Text'].unique()])
    
    # Create a formatted block of all unique existing AI summaries
    summaries_block = "\n".join([f"- {summary}" for summary in group_df['AI Summary'].unique()])
    
    return f"""
You are an expert astrological knowledge engineer. You will be given several passages from different astrological texts, along with their AI-generated summaries. All of these texts describe the same fundamental astrological fact.

Your task is to synthesize all of this information into a single, new, definitive AI Summary.

**Source Passages to Synthesize:**
---
{raw_texts_block}
---

**Existing AI Summaries to Synthesize:**
---
{summaries_block}
---

**Your Tasks:**
1.  **Synthesize a new AI Summary:** Merge the core ideas from all the **Source Passages** and **Existing AI Summaries** into a new, single, eloquent, and comprehensive summary.
2.  **Add Inferred Context:** If the combined information allows for a new astrological inference, add it to your summary in parentheses, like this: (Additional Context: ...).

**CRITICAL:** Your output MUST be a single, valid JSON object with the key "synthesized_summary".

**JSON Response:**
"""

def synthesize_summary_with_ai(group_df: pd.DataFrame) -> str:
    """Calls the AI to generate a new summary from a group of duplicate rows."""
    # If there's only one unique summary to begin with, no need to call the AI
    if len(group_df['AI Summary'].unique()) == 1:
        return group_df['AI Summary'].iloc[0]
        
    prompt = get_synthesis_prompt(group_df)
    try:
        response = MODEL.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        ai_response_obj = json.loads(json_text)
        return ai_response_obj.get('synthesized_summary', "Error during AI summary synthesis.")
    except Exception as e:
        print(f"  - ⚠️ AI synthesis call failed: {e}")
        return "Error during AI summary synthesis."

# --- MAIN ORCHESTRATOR ---
def main():
    print("🚀 Starting V4 Final Consolidation and Synthesis Process...")
    
    # 1. Load and Concatenate All Cleaned Files
    all_dfs = []
    if not os.path.isdir(CLEANED_DATA_PATH):
        print(f"❌ Cleaned data directory not found at '{CLEANED_DATA_PATH}'. Exiting.")
        return
        
    for filename in os.listdir(CLEANED_DATA_PATH):
        #if filename == "interpretations.final.csv": # Process only the final cleaned files
        if filename.endswith('.csv'): # Process ALL cleaned csv files    
            print(f"  -> Loading '{filename}'")
            file_path = os.path.join(CLEANED_DATA_PATH, filename)
            all_dfs.append(pd.read_csv(file_path, dtype=str))

    if not all_dfs:
        print("❌ No 'interpretations.final.csv' files found to consolidate. Exiting.")
        return

    df = pd.concat(all_dfs, ignore_index=True).fillna('')
    
    # 2. Filter for "Passed" Rows Only
    df = df[df['Status'] == 'Passed'].copy()
    print(f"\n📄 Processing {len(df)} total 'Passed' rows for consolidation.")

    # 3. Group by the JSON trigger (adapted from your V2 script)
    grouped = df.groupby('JSON')

    consolidated_rows = []
    print(f"  -> Found {len(grouped)} unique interpretation groups to process.")

    for json_key, group_df in tqdm(grouped, desc="Consolidating Groups"):
        # If a fact is already unique, just pass it through
        if len(group_df) == 1:
            consolidated_rows.append(group_df.iloc[0].to_dict())
            continue

        # If we have duplicates, merge them
        primary_row = group_df.iloc[0].copy() # Use the first row as the base
        
        # 4. Aggregate and Merge Data using robust V2 logic
        primary_row['Raw Text'] = " | ".join(group_df['Raw Text'].unique())
        primary_row['Source'] = merge_unique_values(group_df['Source'])
        primary_row['Page'] = merge_unique_values(group_df['Page'])
        primary_row['Theme'] = merge_unique_values(group_df['Theme'])
        primary_row['Group'] = merge_unique_values(group_df['Group'])
        primary_row['Sub-Theme'] = merge_unique_values(group_df['Sub-Theme'])
        
        # 5. Generate New Synthesized AI Summary
        new_summary = synthesize_summary_with_ai(group_df)
        primary_row['AI Summary'] = new_summary
        
        primary_row['Notes'] = f"Consolidated from {len(group_df)} facts."
        
        consolidated_rows.append(primary_row.to_dict())

    # 6. Create and Save Final DataFrame
    final_df = pd.DataFrame(consolidated_rows)
    
    # Create a new, clean, sequential Fact ID for the final database
    final_df.reset_index(drop=True, inplace=True)
    final_df['Fact ID'] = [f"KB_FACT_{i+1:06}" for i in final_df.index]
    
    print("\n✅ Consolidation and synthesis complete.")
    output_filename = "production_knowledge_base.csv"
    save_df(final_df, CONSOLIDATED_DATA_PATH, output_filename)

if __name__ == "__main__":
    main()