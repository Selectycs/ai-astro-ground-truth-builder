import pandas as pd
import json
import re
import os
import asyncio
import time  # Used to add a delay between AI batches
import google.generativeai as genai
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# --- Load environment variables from .env file ---
load_dotenv()

# --- Configuration ---
INPUT_PATH = "kb_pipeline_v2/03_cleaned/"
INPUT_CSV = "interpretations.cleaned.csv"
OUTPUT_PATH = "kb_pipeline_v2/04_canonical/"
OUTPUT_CSV = "interpretations.canonical_for_validation.csv"

INPUT_FILE = os.path.join(INPUT_PATH, INPUT_CSV)
OUTPUT_FILE = os.path.join(OUTPUT_PATH, OUTPUT_CSV)
BATCH_SIZE = 50
AI_BATCH_SIZE = 25  # 🔽 Reduced to respect quota limits

# --- ASTROLOGICAL ENTITY LISTS ---
NAKSHATRAS = ["Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashirsha", "Ardra", "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni", "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha", "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishtha", "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"]
SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
VARGAS = ["Rasi", "Hora", "Drekkana", "Chaturthamsa", "Saptamsa", "Navamsa", "Dasamsa", "Dvadasamsa", "Shodasamsa", "Vimsamsa", "Siddhamsa", "Trimsamsa", "Shashtyamsa", "D1", "D2", "D3", "D4", "D7", "D9", "D10", "D12", "D16", "D20", "D24", "D30", "D60", "D-1", "D-2", "D-3", "D-4", "D-7", "D-9", "D-10", "D-12", "D-16", "D-20", "D-24", "D-30", "D-60"]
SPECIAL_POINTS = ["Arudha Lagna", "Upapada Lagna", "Bhava Lagna", "Hora Lagna", "Ghati Lagna"]
SPECIAL_CHARTS = ["Bhava Chalit", "Chandra Lagna", "Surya Lagna"]
PLANETS = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]

# --- AI Configuration ---
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("⚠️ Warning: GEMINI_API_KEY not found. Ensure you have a .env file with the key. AI processing will be skipped.")
    else:
        genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"❌ Error configuring Gemini AI: {e}. AI processing will be skipped.")
    GEMINI_API_KEY = None

# --- Keyword Definitions ---
SCHEMA_KEYWORDS: Dict[str, List[str]] = {
    "ashtakavarga": ["ashtakavarga", "bindus"],
    "nakshatra_placement": ["nakshatra"] + NAKSHATRAS,
    "vimsopaka": ["vimsopaka"],
    "special_chart_placement": ["chalit chart"] + SPECIAL_CHARTS,
    "avastha": ["avastha", "state of"],
    "karaka": ["karaka", "significator"],
    "special_point": ["special point"] + SPECIAL_POINTS,
    "house_significance": ["significance of"],
    "dasha": ["dasha", "mahadasha", "antardasha"],
    "lordship": ["lord", "ruler of", "house ruled"],
    "strength": ["strong", "weak", "shadbala", "dignity", "exalted", "debilitated", "vargottama"],
    "conjunction": ["conjunct", "conjunction", "together with"],
    "aspect": ["aspect", "aspected by", "drishti"],
    "varga_placement": ["varga", "chart"] + VARGAS,
    "yoga": ["yoga"],
    "dosha": ["dosha"],
    "sign_trait": ["sign"] + SIGNS,
    "placement": ["placed in", "in the sign of", "in house"],
}

# --- Entity Extraction & JSON Builders ---
def safe_json_loads(s: str) -> Dict:
    try:
        return json.loads(s.replace("'", "\"")) if isinstance(s, str) else {}
    except (json.JSONDecodeError, TypeError):
        return {}

def to_int(v: Any) -> Optional[int]:
    try:
        return int(float(v))
    except (ValueError, TypeError, AttributeError):
        return None

def to_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (ValueError, TypeError, AttributeError):
        return None

def extract_entity(text: str, entity_list: List[str]) -> Optional[str]:
    for entity in entity_list:
        if re.search(r'\b' + re.escape(entity) + r'\b', text, re.IGNORECASE):
            return entity
    return None

def extract_from_text(pattern: str, text: str, group_num: int = 1) -> Optional[str]:
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(group_num).strip() if match else None

def build_canonical_json(schema_type: str, text: str, old_trigger: Dict) -> Dict:
    # --- Universal fallbacks ---
    planet = extract_entity(text, PLANETS) or old_trigger.get("planet_name") or old_trigger.get("planet")
    house = to_int(extract_from_text(r'(\d{1,2})(st|nd|rd|th)? house', text, 1)) \
        or to_int(old_trigger.get("house") or old_trigger.get("placement", {}).get("house"))
    sign = extract_entity(text, SIGNS) or old_trigger.get("sign")

    if schema_type == "placement":
        return {"type": "placement", "planet_name": planet, "house": house, "sign": sign}

    if schema_type == "lordship":
        lord_of = to_int(extract_from_text(r'lord of the (\d{1,2})', text)) \
            or to_int(old_trigger.get("lord_of_house") or old_trigger.get("house"))
        placed_in = to_int(extract_from_text(r'placed in house (\d{1,2})', text)) \
            or to_int(old_trigger.get("placed_in_house") or old_trigger.get("placement", {}).get("house"))
        return {"type": "lordship", "lord_of_house": lord_of, "placed_in_house": placed_in, "planet_name": planet}

    if schema_type == "aspect":
        target_house = to_int(extract_from_text(r'aspects the (\d{1,2})', text)) \
            or to_int(old_trigger.get("target_house"))
        aspect_type = extract_from_text(r'(\w+ drishti)', text) or old_trigger.get("aspect_type")
        return {"type": "aspect", "source_planet": planet, "target_house": target_house, "aspect_type": aspect_type}

    if schema_type == "conjunction":
        planets = sorted(list(set(re.findall(r'\b(' + '|'.join(PLANETS) + r')\b', text, re.IGNORECASE))))
        return {"type": "conjunction", "planet_names": planets if len(planets) > 1 else old_trigger.get("planet_names", []), "house": house}

    if schema_type == "yoga":
        name = extract_from_text(r'([\w\s]+) Yoga', text) or old_trigger.get("name")
        return {"type": "yoga", "name": f"{name} Yoga" if name and "Yoga" not in name else name}

    if schema_type == "dosha":
        name = extract_from_text(r'([\w\s]+) Dosha', text) or old_trigger.get("name")
        return {"type": "dosha", "name": f"{name} Dosha" if name and "Dosha" not in name else name}

    if schema_type == "karaka":
        karaka_type = extract_from_text(r'(Atmakaraka|Amatyakaraka|Bhratrukaraka|Matrukaraka|Putrakaraka|Gnatikaraka|Darakaraka)', text) \
            or old_trigger.get("karaka_type")
        return {"type": "karaka", "karaka_type": karaka_type, "planet_name": planet}

    if schema_type == "special_point":
        point_name = extract_entity(text, SPECIAL_POINTS) or old_trigger.get("point_name")
        return {"type": "special_point", "point_name": point_name, "sign": sign}

    if schema_type == "varga_placement":
        varga = extract_entity(text, VARGAS) or old_trigger.get("varga")
        return {"type": "varga_placement", "varga": varga, "planet_name": planet, "sign": sign}

    if schema_type == "strength":
        strength_type = extract_from_text(r'(shadbala|dignity|exalted|debilitated|vargottama)', text) or old_trigger.get("strength_type")
        value = to_float(extract_from_text(r'value of ([\d\.]+)', text)) or to_float(old_trigger.get("value"))
        return {"type": "strength", "planet_name": planet, "strength_type": strength_type, "value": value}

    if schema_type == "vimsopaka":
        score = to_float(extract_from_text(r'score of ([\d\.]+)', text)) or to_float(old_trigger.get("score"))
        return {"type": "vimsopaka", "planet_name": planet, "score": score}

    if schema_type == "dasha":
        system = extract_from_text(r'(Vimshottari|Ashtottari|Yogini)', text) or old_trigger.get("system", "Vimshottari")
        mahadasha_lord = extract_entity(text, PLANETS) or old_trigger.get("mahadasha_lord") or old_trigger.get("planet_name")
        return {"type": "dasha", "system": system, "mahadasha_lord": mahadasha_lord, "planet_name": planet}

    if schema_type == "avastha":
        state = extract_from_text(r'(Bala|Kumara|Yuva|Vriddha|Mrita)', text) or old_trigger.get("state")
        return {"type": "avastha", "planet_name": planet, "state": state}

    if schema_type == "ashtakavarga":
        scope = extract_from_text(r'(sarvashtakavarga|bhinnaashtakavarga)', text) or old_trigger.get("scope")
        bindus = to_int(extract_from_text(r'(\d+) bindus', text)) or to_int(old_trigger.get("bindus"))
        return {"type": "ashtakavarga", "scope": scope, "house": house, "bindus": bindus}

    if schema_type == "special_chart_placement":
        chart_name = extract_entity(text, SPECIAL_CHARTS) or old_trigger.get("chart_name")
        return {"type": "special_chart_placement", "chart_name": chart_name, "planet_name": planet, "house": house}

    if schema_type == "sign_trait":
        element = extract_from_text(r'(Fire|Earth|Air|Water)', text) or old_trigger.get("element")
        return {"type": "sign_trait", "sign": sign, "element": element, "planet_name": planet, "house": house}

    if schema_type == "nakshatra_placement":
        nakshatra = extract_entity(text, NAKSHATRAS) or old_trigger.get("nakshatra")
        return {"type": "nakshatra_placement", "planet_name": planet, "nakshatra": nakshatra}

    if schema_type == "house_significance":
        significance_match = extract_from_text(r'significance of.*?\[([^\]]+)\]', text)
        significance = significance_match.split(',') if significance_match else old_trigger.get("significance", [])
        return {"type": "house_significance", "house": house, "significance": significance}

    return {"type": schema_type, "error": "Builder logic is missing."}

def build_unstructured_json(old_trigger_str: str) -> Dict:
    return {"type": "unstructured_trigger", "original_trigger_text": old_trigger_str}

# --- Main Heuristic Processing ---
def process_chunk_heuristically(df_chunk: pd.DataFrame) -> pd.DataFrame:
    processed_rows = []
    for _, row in df_chunk.iterrows():
        interpretation_text = str(row.get('interpretation_text', ''))
        old_trigger_str = str(row.get('astrological_trigger_json', '{}'))
        old_trigger_dict = safe_json_loads(old_trigger_str)
        matched_schemas = []
        for schema_type, keywords in SCHEMA_KEYWORDS.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', interpretation_text, re.IGNORECASE):
                    if schema_type not in matched_schemas:
                        matched_schemas.append(schema_type)
        original_fact_id = row['fact_id']
        if not matched_schemas:
            new_row = row.to_dict()
            new_row['canonical_trigger_json'] = json.dumps(build_unstructured_json(old_trigger_str))
            new_row['normalization_source'] = 'fallback'
            new_row['parent_fact_id'] = None
            processed_rows.append(new_row)
        else:
            for i, schema_type in enumerate(matched_schemas):
                new_row = row.to_dict()
                canonical_json = build_canonical_json(schema_type, interpretation_text, old_trigger_dict)
                new_row['canonical_trigger_json'] = json.dumps(canonical_json)
                new_row['normalization_source'] = 'heuristic'
                new_row['parent_fact_id'] = original_fact_id if len(matched_schemas) > 1 else None
                new_row['fact_id'] = f"{original_fact_id}_{i}" if len(matched_schemas) > 1 else original_fact_id
                processed_rows.append(new_row)
    return pd.DataFrame(processed_rows)

# --- Asynchronous AI Processing ---
async def ai_transform_worker(text: str, original_json_str: str, index: int, model):
    prompt = f"""
    You are a data structuring expert. Your task is to analyze the following astrological interpretation text 
    and identify ALL relevant astrological concepts within it.

    **Canonical Schema Examples:** ... (omitted here for brevity) ...

    **Interpretation Text:**
    "{text}"

    **Original JSON:**
    {original_json_str}

    RULES:
    - Use interpretation text first.
    - Use original JSON only as fallback for attributes.
    - For EACH concept found, generate JSON object(s).
    - Respond with ONLY JSON object or JSON array (no markdown).
    """
    try:
        response = await model.generate_content_async(prompt, generation_config={"temperature": 0.0})
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        parsed_data = json.loads(json_text)
        if isinstance(parsed_data, dict):
            parsed_data = [parsed_data]
        return index, parsed_data, 'AI_SUCCESS'
    except Exception as e:
        print(f"   - ❌ AI worker failed for index {index}: {e}")
        fallback_json = [{"type": "unstructured_trigger", "original_trigger_text": original_json_str}]
        return index, fallback_json, 'AI_FAILED'

async def process_fallbacks_with_ai_async(df: pd.DataFrame) -> pd.DataFrame:
    if not GEMINI_API_KEY:
        print("Skipping AI processing as API key is not configured.")
        return df

    fallback_rows = df[df['normalization_source'] == 'fallback']
    if fallback_rows.empty:
        print("No fallback rows to process with AI.")
        return df

    print(f"\n🚀 Starting AI normalization for {len(fallback_rows)} fallback rows...")
    model = genai.GenerativeModel('gemini-1.5-pro-latest')

    results = []
    tasks = []
    for index, row in fallback_rows.iterrows():
        task = ai_transform_worker(
            text=str(row.get('interpretation_text', '')),
            original_json_str=str(row.get('astrological_trigger_json', '{}')),
            index=index,
            model=model
        )
        tasks.append(task)

    # Process in batches to respect AI_BATCH_SIZE
    for i in range(0, len(tasks), AI_BATCH_SIZE):
        batch = tasks[i:i + AI_BATCH_SIZE]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)
        time.sleep(1)  # slight delay between batches

    print("   - ✅ AI tasks completed. Integrating results...")

    indices_to_drop = []
    new_rows_data = []
    for index, parsed_jsons, status in results:
        indices_to_drop.append(index)
        original_row = df.loc[index]
        for i, parsed_json in enumerate(parsed_jsons):
            new_row = original_row.to_dict()
            new_row['canonical_trigger_json'] = json.dumps(parsed_json)
            new_row['normalization_source'] = 'AI' if status == 'AI_SUCCESS' else 'AI_FAILED'
            if len(parsed_jsons) > 1:
                new_row['parent_fact_id'] = original_row['fact_id']
                new_row['fact_id'] = f"{original_row['fact_id']}_ai_{i}"
            new_rows_data.append(new_row)

    df.drop(indices_to_drop, inplace=True)
    new_rows_df = pd.DataFrame(new_rows_data)
    final_df = pd.concat([df, new_rows_df], ignore_index=True)

    return final_df

# --- Main Execution ---
def main():
    print("🚀 Starting normalization pipeline...")
    try:
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        print(f"Input:  '{INPUT_FILE}'")
        print(f"Output: '{OUTPUT_FILE}'")
        chunk_iterator = pd.read_csv(INPUT_FILE, chunksize=BATCH_SIZE, keep_default_na=False)
        print(f"\n1. Running heuristic normalization...")
        all_chunks = [process_chunk_heuristically(chunk) for chunk in chunk_iterator]
        heuristic_df = pd.concat(all_chunks, ignore_index=True)
        print("   - Heuristic pass complete.")
        print("\n2. Running AI normalization for remaining fallbacks...")
        final_df = asyncio.run(process_fallbacks_with_ai_async(heuristic_df))
        original_cols = [col for col in heuristic_df.columns if col not in ['canonical_trigger_json', 'normalization_source', 'parent_fact_id']]
        new_order = ['fact_id', 'parent_fact_id', 'canonical_trigger_json', 'normalization_source'] + original_cols
        final_df = final_df.reindex(columns=new_order)
        final_df.to_csv(OUTPUT_FILE, index=False)

        print("\n📊 Final Output Summary:")
        def get_schema_type(json_str):
            try:
                return json.loads(json_str).get('type', 'unknown_json_format')
            except (json.JSONDecodeError, TypeError):
                return 'malformed_json'

        final_df['schema_type'] = final_df['canonical_trigger_json'].apply(get_schema_type)
        print(final_df['schema_type'].value_counts())

        print(f"\n🎉 Normalization complete!")
        print(f"   - Total rows generated: {len(final_df)}")
        print(f"   - Output saved to: {OUTPUT_FILE}")

    except FileNotFoundError:
        print(f"❌ ERROR: Input file not found at '{INPUT_FILE}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
