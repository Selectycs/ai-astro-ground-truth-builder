# v2_reconcile.py

import pandas as pd
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
import time
from typing import Dict, Optional
from jsonschema import validate, ValidationError # <-- NEW IMPORT

from v2_kb_utils import V2_INTERPRETATIONS_SCHEMA, RAW_DATA_PATH, VALIDATED_DATA_PATH, generate_id, save_df
from v2_clean import TRIGGER_JSON_SCHEMA # <-- Import the schema from clean script

# (Configuration remains the same)
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = genai.GenerativeModel('gemini-1.5-pro-latest')
RATE_LIMIT_DELAY = 1.5

def get_refinement_prompt(row: pd.Series) -> str:
    raw_trigger_json = row['astrological_trigger_json']
    interpretation_text = row['interpretation_text']

    return f"""
    You are an expert Vedic Astrology knowledge engineer specializing in data modeling. Your task is to analyze the provided astrological fact and refine its trigger JSON.

    **Fact to Analyze:**
    - Interpretation Text: "{interpretation_text}"
    - Raw Trigger JSON: ```json
    {raw_trigger_json}
    ```

    **Your Tasks:**
    1.  **Refine the Trigger JSON:** Rewrite the `Raw Trigger JSON` to be as detailed and semantically precise as possible, following the examples. The refined JSON must adhere to the Golden Rule (contain `calculation_context` with a `type` key, and `components`). Do not add external knowledge not present in the text.
    2.  **Assign Confidence:** Based on the text's nature, assign a confidence score: 0.9 for a specific, checkable rule (e.g., 'Sun in the 9th house'); 0.75 for a general interpretation; 0.6 for a broad, conceptual statement.

    **CRITICAL: Your output MUST be a single, valid JSON object with ONLY TWO keys:** "refined_trigger_json" and "confidence_score".

    **JSON Response:**
    `{{"refined_trigger_json": {{...}}, "confidence_score": 0.xx }}`
    """

def refine_row_json(row: pd.Series) -> Optional[Dict]:
    prompt = get_refinement_prompt(row)
    try:
        generation_config = genai.types.GenerationConfig(temperature=0.0, max_output_tokens=4096)
        response = MODEL.generate_content(prompt, generation_config=generation_config)
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_text)
    except Exception as e:
        print(f"  - ❌ Error calling AI for Fact ID {row['fact_id']}: {e}")
        return None

def main():
    print("🚀 Starting V2 Reconciliation (JSON Refinement & Scoring) Process...")
    # ... (file loading logic is the same)
    raw_files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith("_combined.csv")]
    if not raw_files:
        print("No combined raw files found...")
        return
    input_file = os.path.join(RAW_DATA_PATH, sorted(raw_files)[-1])
    df = pd.read_csv(input_file, dtype=str).fillna("")

    validated_rows = []
    for index, row in df.iterrows():
        print(f"  -> Refining row {index + 1}/{len(df)} (Fact ID: {row['fact_id']})...")
        
        ai_feedback = refine_row_json(row)
        
        updated_row = row.to_dict()
        if ai_feedback and ai_feedback.get("refined_trigger_json"):
            refined_json_obj = ai_feedback["refined_trigger_json"]
            
            # --- NEW: Early Schema Validation ---
            try:
                validate(instance=refined_json_obj, schema=TRIGGER_JSON_SCHEMA)
                
                # --- Success Case ---
                refined_json_str = json.dumps(refined_json_obj)
                updated_row['astrological_trigger_json'] = refined_json_str
                updated_row['interpretation_group'] = generate_id(refined_json_str)
                updated_row['confidence_score'] = ai_feedback.get('confidence_score', 0.75) # Use stratified score
                updated_row['status'] = 'VALIDATED'
                updated_row['notes'] = "JSON refined and scored by AI."

            except (ValidationError, json.JSONDecodeError) as e:
                print(f"  - ⚠️ AI produced invalid JSON for Fact ID {row['fact_id']}. Using fallback. Error: {e}")
                updated_row['status'] = 'VALIDATED'
                updated_row['notes'] = "JSON refinement produced invalid schema; using original raw JSON."

        else:
            # --- Auto-Fallback Logic ---
            print(f"  - ⚠️ Refinement failed for Fact ID {row['fact_id']}. Using original raw JSON.")
            updated_row['status'] = 'VALIDATED'
            updated_row['notes'] = "JSON refinement AI call failed; using original raw JSON."
        
        validated_rows.append(updated_row)
        time.sleep(RATE_LIMIT_DELAY)

    if validated_rows:
        validated_df = pd.DataFrame(validated_rows)
        validated_df = validated_df[V2_INTERPRETATIONS_SCHEMA]
        save_df(validated_df, VALIDATED_DATA_PATH, "interpretations.validated.csv")

if __name__ == "__main__":
    main()