import os, re, json, time, uuid, hashlib
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

# ---------------- CONFIG ---------------- #
KB_DIR = "knowledge-base"
INPUT_CSV = os.path.join(KB_DIR, "interpretations.csv")
OUTPUT_CSV = os.path.join(KB_DIR, "interpretations.consolidated.csv")
PROVENANCE_CSV = os.path.join(KB_DIR, "interpretations.provenance_links.csv")
MODEL_NAME = "gemini-1.5-pro-latest"
TEMPERATURE = 0.1
MAX_TOKENS = 8192
RATE_DELAY_SEC = 1.0

# ---------------- MODEL INIT ---------------- #
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("GEMINI_API_KEY missing in environment")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# ---------------- UTILITIES ---------------- #
PLANETS = ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn","Rahu","Ketu", "Lagna Lord","Ascendant Lord"]
REL_WORDS = ["in","aspect","aspects","aspecting","conjunct","conjunction","opposes", "trine","square","sextile","lord","exchange"]
ORD2NUM = {"1st":1,"2nd":2,"3rd":3,"4th":4,"5th":5,"6th":6,"7th":7,"8th":8,"9th":9, "10th":10,"11th":11,"12th":12}
HOUSE_RX = re.compile(r'\b(1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th|11th|12th)\s+house\b', re.I)
YOGA_RX = re.compile(r'\b(Raja Yoga|Pancha Mahapurusha Yoga|Gajakesari Yoga)\b', re.I)

def uid() -> str:
    return uuid.uuid4().hex

def norm_subthemes(val: str) -> str:
    if pd.isna(val) or not str(val).strip():
        return ""
    toks = [t.strip().lower() for t in str(val).split(",") if t.strip()]
    return ",".join(sorted(set(toks)))

def extract_signature(fp: str) -> str:
    """Return a canonical trigger signature so we only merge truly duplicate triggers."""
    if not fp:
        return "sig:unknown"
    s = fp.strip()
    low = s.lower()

    # --- ADDED: Condition/valence detection ---
    positive_keywords = ["strong", "well-placed", "bright", "exalted"]
    negative_keywords = ["afflicted", "malefic", "weak", "debilitated"]
    condition = ""
    if any(kw in low for kw in positive_keywords):
        condition = "|condition=strong"
    elif any(kw in low for kw in negative_keywords):
        condition = "|condition=afflicted"
    # ----------------------------------------

    # --- ADDED: Element detection ---
    elements = ["Fiery", "Earthy", "Airy", "Watery"]
    found_element = ""
    for el in elements:
        if el.lower() in low:
            found_element = el
            break
    # --------------------------------

    ym = YOGA_RX.search(s)
    if ym:
        return f"yoga={ym.group(1).strip()}"

    found = [p for p in PLANETS if p.lower() in low]
    p1 = found[0] if found else ""
    p2 = found[1] if len(found) > 1 else ""

    hm = HOUSE_RX.search(s)
    hnum = ORD2NUM.get(hm.group(1).lower(), "") if hm else ""
    
    if hnum and found_element:
        return f"house={hnum}|element={found_element}"

    rel = ""
    for w in REL_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", low):
            rel = w
            break

    # Build signature with new condition logic
    if p1 and rel == "in" and hnum:
        return f"planet={p1}|rel=in|house={hnum}{condition}"
    if p1 and p2 and rel in ("conjunct","conjunction"):
        return f"planet={p1}|rel=conjunct|planet2={p2}|house={hnum or ''}{condition}"
    if p1 and "aspect" in rel and hnum:
        return f"planet={p1}|rel=aspect|house={hnum}{condition}"
    if p1 and hnum and not rel:
        return f"planet={p1}|house={hnum}{condition}"
    if p1 and not hnum and rel:
        return f"planet={p1}|rel={rel}{condition}"
    if p1:
        return f"planet={p1}{condition}"
    if hnum:
        return f"house={hnum}{condition}"

    key = re.sub(r"\s+", " ", low)[:80]
    return "sig:" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]

def get_consolidation_prompt(fact_group_df: pd.DataFrame) -> str:
    blocks = []
    for _, row in fact_group_df.iterrows():
        fp = str(row.get("Foundation_Point", "")).strip()
        it = str(row.get("Interpretation_Text", "")).strip()
        blocks.append(f"- Foundation: {fp}\n  Interpretation: {it}")
    facts_text = "\n\n".join(blocks)

    return f"""
You are an expert Vedic astrology knowledge engineer.

Consolidate the following facts into exactly ONE canonical fact. IMPORTANT:
• All items share the SAME trigger signature. Merge only phrasing, not meaning.
• Keep the trigger explicit in Foundation_Point.
• Interpretation_Text should be precise and comprehensive.
• AI_Astro_Summary must be ≤ 20 words, plain-language.
• Chart_Refs_JSON must include consolidated structured refs.

STRICT OUTPUT: JSON with keys:
"Foundation_Point","Interpretation_Text","AI_Astro_Summary","Chart_Refs_JSON"

FACTS:
---
{facts_text}
---
""".strip()

def safe_load_json(txt: str) -> dict:
    t = (txt or "").strip()
    t = t.replace("```json", "").replace("```", "").strip()
    if "{" in t and "}" in t:
        t = t[t.find("{"): t.rfind("}") + 1]
    return json.loads(t)

def consolidate_signature_cluster(cluster_df: pd.DataFrame) -> dict | None:
    prompt = get_consolidation_prompt(cluster_df)
    try:
        resp = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=TEMPERATURE,
                max_output_tokens=MAX_TOKENS
            )
        )
        data = safe_load_json(getattr(resp, "text", "") or "")
        return data
    except Exception:
        return None

def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _recalc_conf(gsig: pd.DataFrame) -> str:
    vals = []
    for _, r in gsig.iterrows():
        try:
            vals.append(float(r.get("Confidence_Score","0") or 0))
        except Exception:
            vals.append(0.0)
    base = max(vals) if vals else 0.5
    base = max(base, 0.55)
    base = min(0.9, base + 0.10)
    return f"{base:.2f}"

def main():
    if not os.path.exists(INPUT_CSV):
        raise SystemExit(f"Missing input file: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")
    df["Sub_Themes_norm"] = df["Sub_Themes"].apply(norm_subthemes)
    df["Signature"] = df["Foundation_Point"].apply(extract_signature)

    consolidated_rows, provenance = [], []
    outer = df.groupby(["Theme","Fact_Group","Sub_Themes_norm"], dropna=False)

    for (theme, group, subs), g in outer:
        for sig, gsig in g.groupby("Signature", dropna=False):
            if len(gsig) == 1:
                consolidated_rows.append(gsig.iloc[0].to_dict())
                continue

            data = consolidate_signature_cluster(gsig)
            if not data:
                consolidated_rows.extend(gsig.to_dict("records"))
                continue
            
            all_sub_themes = gsig['Sub_Themes'].dropna().unique()
            merged_sub_themes = ','.join(sorted(list(set(
                st.strip() for sublist in all_sub_themes for st in sublist.split(',') if st.strip()
            ))))

            base = gsig.iloc[0].to_dict()
            new_id = uid()
            base.update({
                "Fact_ID": new_id,
                "Sub_Themes": merged_sub_themes,
                "Foundation_Point": data.get("Foundation_Point","").strip(),
                "Interpretation_Text": data.get("Interpretation_Text","").strip(),
                "AI_Astro_Summary": data.get("AI_Astro_Summary","").strip(),
                "Chart_Refs_JSON": json.dumps(data.get("Chart_Refs_JSON", {}), ensure_ascii=False),
                "Status": "NEEDS_REVIEW",
                "Confidence_Score": _recalc_conf(gsig),
                "Source_Name": "AI Consolidation",
                "Source_Type": "AI_MODEL",
                "Source_Reference": f"{MODEL_NAME}|signature={sig}",
                "Last_Updated": _now_iso(),
                "Notes": f"Consolidated from {len(gsig)} facts; signature={sig}"
            })
            consolidated_rows.append(base)
            for _, r in gsig.iterrows():
                provenance.append((r["Fact_ID"], new_id))
            time.sleep(RATE_DELAY_SEC)

    out_df = pd.DataFrame(consolidated_rows)
    out_df.drop(columns=[c for c in ["Sub_Themes_norm","Signature"] if c in out_df.columns], inplace=True, errors="ignore")
    os.makedirs(KB_DIR, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    pd.DataFrame(provenance, columns=["Original_Fact_ID","Primary_Fact_ID"]).to_csv(PROVENANCE_CSV, index=False)

    print(f"✅ Wrote {len(out_df)} rows → {OUTPUT_CSV}")
    print(f"🔗 Provenance map → {PROVENANCE_CSV}")

if __name__ == "__main__":
    main()