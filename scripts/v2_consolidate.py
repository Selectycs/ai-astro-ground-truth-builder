import pandas as pd
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Optional

# --- Import V2 utilities ---
from v2_kb_utils import (
    V2_PRODUCTION_SCHEMA,
    VALIDATED_DATA_PATH,
    CONSOLIDATED_DATA_PATH,
    save_df
)

# --- CONFIGURATION & INITIALIZATION ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = genai.GenerativeModel('gemini-1.5-pro-latest')

def get_consolidation_prompt(cluster_df: pd.DataFrame, trigger_json_str: str) -> str:
    """
    Creates the prompt for the AI to synthesize multiple interpretations into one.
    """
    # Create a formatted block of text with all interpretations to be merged
    blocks = [f"- {row['interpretation_text']}" for _, row in cluster_df.iterrows()]
    facts_text = "\n".join(blocks)
    
    # --- NEW: Create a block for existing summaries to preserve their context ---
    summary_blocks = [
        f"- {row['interpretation_summary_ai']}"
        for _, row in cluster_df.iterrows() if row['interpretation_summary_ai']
    ]
    summaries_text = "\n".join(summary_blocks)
    
    # Add the trigger to the prompt for context
    trigger_obj = json.loads(trigger_json_str)
    
    return f"""
    You are an expert astrological knowledge engineer. You are given an astrological trigger, several detailed interpretations, and their existing summaries. Your task is to consolidate all of this information into a single, definitive fact.

    **Astrological Trigger to Synthesize:**
    ---
    {json.dumps(trigger_obj, indent=2)}
    ---

    **Full Interpretations to Consolidate:**
    ---
    {facts_text}
    ---

    **Existing Summaries & Context to Consolidate:**
    ---
    {summaries_text}
    ---

    **Your Tasks:**
    1.  **Synthesize Interpretation Text:** Merge the core meanings from the **Full Interpretations** into a single, eloquent, and comprehensive paragraph. Do not lose any unique nuances. The final text should be a definitive, standalone explanation grounded in the provided astrological trigger.
    2.  **Create AI Summary:** Synthesize the core ideas from the **Existing Summaries & Context** into a new, definitive summary sentence (max 25 words). It should reflect the most important nuances and inferred context from the source material.

    **CRITICAL: Your output MUST be a single, valid JSON object with the keys:** "merged_interpretation_text" and "merged_summary_ai".

    **JSON Response:**
    """

def consolidate_cluster_with_ai(cluster_df: pd.DataFrame, trigger_json_str: str) -> Optional[Dict]:
    """
    Sends a cluster of facts to the AI for text consolidation.
    """
    prompt = get_consolidation_prompt(cluster_df, trigger_json_str)
    try:
        generation_config = genai.types.GenerationConfig(temperature=0.2, max_output_tokens=4096)
        response = MODEL.generate_content(prompt, generation_config=generation_config)
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_text)
    except Exception as e:
        print(f"  - ❌ AI text consolidation failed: {e}")
        return None

def merge_unique_values(series: pd.Series) -> str:
    """
    Takes a pandas Series of strings (which may be comma-separated),
    and returns a single, sorted, comma-separated string of unique values.
    """
    all_items = {item.strip() for val in series.dropna() for item in str(val).split(',') if item.strip()}
    return ",".join(sorted(list(all_items)))

def main():
    print("🚀 Starting V2 Final Consolidation Process...")
    
    input_file = os.path.join(VALIDATED_DATA_PATH, "interpretations.validated_canonical.csv")
    if not os.path.exists(input_file):
        print(f"Validated file not found at {input_file}. Please run reconcile script first.")
        return

    df = pd.read_csv(input_file, dtype=str).fillna("")
    print(f"  - Loaded {len(df)} validated rows.")
    
    grouped = df.groupby('astrological_trigger_json')
    
    consolidated_rows = []

    for trigger_json_str, group_df in grouped:
        if len(group_df) == 1:
            row_dict = group_df.iloc[0].to_dict()
            consolidated_rows.append(row_dict)
            continue

        print(f"  -> Consolidating group for trigger... ({len(group_df)} members)")
        
        primary_fact_row = group_df.loc[group_df['confidence_score'].astype(float).idxmax()].copy()
        
        merged_text_data = consolidate_cluster_with_ai(group_df, trigger_json_str)
        
        if merged_text_data:
            primary_fact_row['interpretation_text'] = merged_text_data['merged_interpretation_text']
            primary_fact_row['interpretation_summary_ai'] = merged_text_data['merged_summary_ai']
        
        primary_fact_row['theme'] = merge_unique_values(group_df['theme'])
        primary_fact_row['theme_group'] = merge_unique_values(group_df['theme_group']) 
        primary_fact_row['sub_theme'] = merge_unique_values(group_df['sub_theme'])
        primary_fact_row['raw_text'] = merge_unique_values(group_df['raw_text'])
        primary_fact_row['source_name'] = merge_unique_values(group_df['source_name'])
        primary_fact_row['source_reference'] = merge_unique_values(group_df['source_reference'])
        
        primary_fact_row['notes'] = f"Consolidated from {len(group_df)} facts."
        primary_fact_row['status'] = 'CONSOLIDATED'
        
        #consolidated_rows.append(primary_fact_row)
        consolidated_rows.append(primary_fact_row.to_dict())
    
    if consolidated_rows:
        final_df = pd.DataFrame(consolidated_rows)
        print(f"\n✅ Consolidation complete. Total consolidated facts: {len(final_df)}.")
        
        final_df = final_df[V2_PRODUCTION_SCHEMA]
        
        save_df(final_df, CONSOLIDATED_DATA_PATH, "interpretations.production.csv")
    else:
        print("No rows were produced after consolidation.")

if __name__ == "__main__":
    main()