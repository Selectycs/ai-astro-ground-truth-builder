# v2_consolidate.py

import pandas as pd
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
import time
from typing import List, Dict, Optional

from v2_kb_utils import V2_INTERPRETATIONS_SCHEMA, VALIDATED_DATA_PATH, CONSOLIDATED_DATA_PATH, generate_id, save_df

# (Configuration is the same)
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = genai.GenerativeModel('gemini-1.5-pro-latest')
RATE_LIMIT_DELAY = 1.5

# --- NEW: Conflict Detection Keywords ---
CONFLICT_KEYWORDS = {
    'positive': ['good', 'benefic', 'success', 'wealth', 'gain', 'positive', 'strong', 'fame', 'rise'],
    'negative': ['bad', 'malefic', 'failure', 'poverty', 'loss', 'negative', 'weak', 'inauspicious', 'fall']
}

def detect_conflict(texts: List[str]) -> bool:
    """Simple keyword-based conflict detection."""
    has_positive = False
    has_negative = False
    for text in texts:
        lower_text = text.lower()
        if any(word in lower_text for word in CONFLICT_KEYWORDS['positive']):
            has_positive = True
        if any(word in lower_text for word in CONFLICT_KEYWORDS['negative']):
            has_negative = True
    return has_positive and has_negative

def get_consolidation_prompt(cluster_df: pd.DataFrame, is_conflict: bool) -> str:
    blocks = [f"- Source Interpretation: \"{row['interpretation_text']}\"" for _, row in cluster_df.iterrows()]
    facts_text = "\n\n".join(blocks)
    
    # --- NEW: Dynamic Prompt based on conflict detection ---
    conflict_instruction = ""
    if is_conflict:
        conflict_instruction = "**Conflict Detected:** The source interpretations appear to contradict each other. Your synthesized text MUST acknowledge the differing viewpoints. For example: 'While some authorities state this gives wealth, other sources suggest it can lead to challenges...'"

    return f"""
    You are an expert astrological knowledge engineer. You are given several interpretations for the exact same astrological trigger. Your task is to consolidate them into a single, comprehensive primary fact.

    {conflict_instruction}

    **Interpretations to Consolidate:**
    ---
    {facts_text}
    ---

    **Your Tasks:**
    1.  **Synthesize Interpretation Text:** Merge the meanings from all source interpretations into a single, eloquent, and comprehensive paragraph. Do not lose any unique nuances.
    2.  **Create AI Summary:** Write a new, single, plain-language summary sentence (max 25 words).

    **CRITICAL: Your output MUST be a single, valid JSON object with the keys:** "merged_interpretation_text" and "merged_summary_ai".
    """

def consolidate_cluster_text(cluster_df: pd.DataFrame) -> Optional[Dict]:
    texts = cluster_df['interpretation_text'].tolist()
    is_conflict = detect_conflict(texts)
    
    prompt = get_consolidation_prompt(cluster_df, is_conflict)
    
    try:
        generation_config = genai.types.GenerationConfig(temperature=0.3, max_output_tokens=4096)
        response = MODEL.generate_content(prompt, generation_config=generation_config)
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        result = json.loads(json_text)
        result['is_conflict'] = is_conflict # Pass the conflict status back
        return result
    except Exception as e:
        print(f"  - ❌ AI text consolidation failed: {e}")
        return None

def main():
    print("🚀 Starting V2 Consolidation Process...")
    input_file = os.path.join(VALIDATED_DATA_PATH, "interpretations.validated.csv")
    if not os.path.exists(input_file):
        print("Validated file not found...")
        return
    df = pd.read_csv(input_file, dtype=str).fillna("")
    
    df['interpretation_group'] = df['astrological_trigger_json'].apply(generate_id)
    
    consolidated_rows = []
    grouped = df.groupby('interpretation_group')

    for group_name, group_df in grouped:
        print(f"  -> Processing group '{group_name}' ({len(group_df)} members)...")
        
        if len(group_df) == 1:
            row = group_df.iloc[0].to_dict()
            row['status'] = 'CONSOLIDATED'
            row['conflict_status'] = 'NO_CONFLICT' # <-- Set default
            row['schema_version'] = 'v2.0'       # <-- Set version
            consolidated_rows.append(row)
            continue

        primary_fact = group_df.loc[group_df['confidence_score'].astype(float).idxmax()]
        merged_data = consolidate_cluster_text(group_df)
        
        if not merged_data:
            # Fallback logic if AI fails
            continue

        new_row = primary_fact.to_dict()
        new_row['interpretation_text'] = merged_data['merged_interpretation_text']
        new_row['interpretation_summary_ai'] = merged_data['merged_summary_ai']
        
        # --- NEW: Set conflict status and schema version ---
        new_row['conflict_status'] = 'RESOLVED_BY_AI' if merged_data['is_conflict'] else 'NO_CONFLICT'
        new_row['schema_version'] = 'v2.0'
        
        # --- NEW: Dynamic Confidence Score ---
        base_score = float(primary_fact['confidence_score'])
        num_sources = len(group_df['source_name'].unique())
        if merged_data['is_conflict']:
            # Down-weight for ambiguity
            new_row['confidence_score'] = max(0.5, base_score - 0.15)
        else:
            # Boost for corroboration
            new_row['confidence_score'] = min(0.98, base_score + (0.02 * (num_sources - 1)))
            
        # Merge sub_themes
        all_sub_themes = {s.strip() for sub_str in group_df['sub_theme'].dropna() for s in sub_str.split(',') if s.strip()}
        new_row['sub_theme'] = ",".join(sorted(list(all_sub_themes)))

        new_row['source_reference'] = "; ".join(group_df['source_reference'].unique())
        new_row['notes'] = f"Consolidated from {len(group_df)} facts. Original IDs: {list(group_df['fact_id'])}"
        new_row['primary_fact_id'] = new_row['fact_id']
        new_row['status'] = 'CONSOLIDATED'
        
        consolidated_rows.append(new_row)
        time.sleep(RATE_LIMIT_DELAY)
    
    if consolidated_rows:
        consolidated_df = pd.DataFrame(consolidated_rows)
        consolidated_df = consolidated_df[V2_INTERPRETATIONS_SCHEMA]
        save_df(consolidated_df, CONSOLIDATED_DATA_PATH, "interpretations.consolidated.csv")

if __name__ == "__main__":
    main()