import os, re, json, datetime as dt
import pandas as pd

KB_DIR = "knowledge-base"
INPUT = os.path.join(KB_DIR, "interpretations.consolidated.csv")
OUTPUT = os.path.join(KB_DIR, "interpretations.cleaned.csv")

AI_SUMMARY_MAX_WORDS = 20

PLANETS = [
    "Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn","Rahu","Ketu",
    "Lagna Lord","Ascendant Lord"
]
PLANET_SET = {p.lower(): p for p in PLANETS}

ORD2NUM = {"1st":1,"2nd":2,"3rd":3,"4th":4,"5th":5,"6th":6,"7th":7,"8th":8,"9th":9,
           "10th":10,"11th":11,"12th":12}

HOUSE_PATTERN = re.compile(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+house\b', flags=re.I)
PLANET_PATTERN = re.compile(r'\b(Sun|Moon|Mars|Mercury|Jupiter|Venus|Saturn|Rahu|Ketu|Lagna Lord|Ascendant Lord)\b', flags=re.I)
LAGNA_IN_TEXT = re.compile(r'\b(lagna|ascendant|asc)\b', flags=re.I)
NAKSHATRA_HINT = re.compile(r'\bnakshatra\b', flags=re.I)
YOGA_PATTERN = re.compile(r'\b(Raja Yoga|Pancha Mahapurusha Yoga|Gajakesari Yoga)\b', flags=re.I)
LORD_CONN_PATTERN = re.compile(
    r'\b((?:\d{1,2}(?:st|nd|rd|th)\s+Lord|Lagna Lord|Ascendant Lord))\s+'
    r'(connected to|conjunct|joins|yuti|linked to|associated with|in sambandha with|in yoga with)\s+'
    r'((?:\d{1,2}(?:st|nd|rd|th)\s+Lord|Lagna Lord|Ascendant Lord))\b',
    flags=re.I
)
ATMAKARAKA_PATTERN = re.compile(r'\bAtmakaraka\b', flags=re.I)

def now_iso():
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def fix_mojibake(s: str) -> str:
    if s is None:
        return s
    s = str(s)
    repl = {
        "â€™": "’", "â€˜": "‘",
        "â€œ": "“", "â€": "”", "â€ť": "”",
        "â€“": "–", "â€”": "—",
        "â€¢": "•", "Ã—": "×", "Â": "",
    }
    out = s
    for k, v in repl.items():
        out = out.replace(k, v)
    return re.sub(r'\s+', ' ', out).strip()

def canonicalize_foundation_point(fp: str, interpretation: str) -> str:
    raw = fix_mojibake(fp or "")
    low = raw.lower()
    if re.fullmatch(r"Lagna\s*\(1st House\)", raw, flags=re.I):
        return "Condition of Lagna (Ascendant: sign, degree, planets in 1st)"
    if low == "lagna":
        return "Condition of Lagna (Ascendant: sign, degree, planets in 1st)"
    if low == "lagna lord":
        return "Condition of the Lagna Lord (dignity, house, aspects)"
    if low == "sun":
        return "Sun’s dignity and house placement"
    if low == "moon":
        return "Moon’s sign, nakshatra, and phase"
    return raw

def clean_summary(text: str, interpretation: str, foundation: str = "") -> str:
    t = fix_mojibake(text or "")
    if not t:
        src = re.split(r'[.!?]', fix_mojibake(interpretation).strip())[0]
        t = src
    t = re.sub(r'\s+', ' ', t).strip()
    words = t.split()
    if len(words) > AI_SUMMARY_MAX_WORDS:
        t = " ".join(words[:AI_SUMMARY_MAX_WORDS]).rstrip(",;:") + "."
    if not t.endswith((".", "!", "?")):
        t += "."
    if "dignity" in (foundation or "").lower() and "dignity" not in t.lower():
        t = re.sub(r'\.$', '', t) + " (by dignity)."
    return t

def parse_json_safe(s):
    if not s or not str(s).strip():
        return {}
    try:
        d = json.loads(s)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}

def normalize_list(v):
    """Normalize any list to a flat list of strings (except we don't enforce for houses here)."""
    out = []
    if isinstance(v, list):
        for x in v:
            xs = str(x).strip()
            if xs.startswith("{") and xs.endswith("}"):
                continue
            if xs:
                out.append(xs)
    return sorted(set(out))

def _to_int_list(seq):
    ints = []
    for x in (seq or []):
        try:
            if isinstance(x, int):
                ints.append(x)
            else:
                xs = str(x).strip()
                # handle tokens like "10", or accidental "10.0"
                if re.fullmatch(r'\d+', xs):
                    ints.append(int(xs))
                elif re.fullmatch(r'\d+\.0', xs):
                    ints.append(int(float(xs)))
        except Exception:
            continue
    return sorted(set(ints))

def infer_chart_refs(foundation: str, interpretation: str, summary: str) -> dict:
    joined = " ".join([foundation or "", interpretation or "", summary or ""])
    out = {"houses": [], "planets": [], "yogas": [], "nakshatras": [], "special": []}

    # Houses like "10th house"
    for m in HOUSE_PATTERN.finditer(joined):
        n = int(m.group(1))
        if 1 <= n <= 12:
            out["houses"].append(n)

    # Lagna implies 1st house
    if LAGNA_IN_TEXT.search(joined):
        out["houses"].append(1)

    # Planets
    for m in PLANET_PATTERN.finditer(joined):
        out["planets"].append(PLANET_SET.get(m.group(1).lower(), m.group(1)))

    # Yogas
    for m in YOGA_PATTERN.finditer(joined):
        out["yogas"].append(m.group(1))

    # Nakshatra hint
    if NAKSHATRA_HINT.search(joined):
        out["nakshatras"].append("Moon Nakshatra")

    # Lords connections
    for m in LORD_CONN_PATTERN.finditer(joined):
        a, _, b = m.groups()
        out["planets"].extend([a.strip(), b.strip()])

    # Atmakaraka
    if ATMAKARAKA_PATTERN.search(joined):
        out["special"].append("Atmakaraka")

    # Dedupe/sort with proper typing
    out["houses"] = _to_int_list(out["houses"])
    out["planets"] = sorted(set(str(x) for x in out["planets"]))
    out["yogas"] = sorted(set(str(x) for x in out["yogas"]))
    out["nakshatras"] = sorted(set(str(x) for x in out["nakshatras"]))
    out["special"] = sorted(set(str(x) for x in out["special"]))

    # Drop empties
    return {k: v for k, v in out.items() if v}

def merge_chart_refs(existing: dict, inferred: dict) -> dict:
    """Coerce houses→int, others→str, then merge and sort."""
    existing = existing or {}
    inferred = inferred or {}
    out = {}

    # Houses → ints
    houses_a = _to_int_list(existing.get("houses", []))
    houses_b = _to_int_list(inferred.get("houses", []))
    houses = sorted(set(houses_a) | set(houses_b))
    if houses:
        out["houses"] = houses

    # Stringy lists
    for k in ["planets", "yogas", "nakshatras", "special", "vargas"]:
        a = existing.get(k, [])
        b = inferred.get(k, [])
        aset = set(str(x) for x in (a or []))
        bset = set(str(x) for x in (b or []))
        merged = sorted(aset | bset)
        if merged:
            out[k] = merged

    return out

def clean_chart_refs_json(raw_json: str, foundation: str, interpretation: str, summary: str) -> str:
    d = parse_json_safe(raw_json)
    # normalize existing arrays (keep as-strings; houses will be coerced in merge)
    for key in ["houses","planets","yogas","nakshatras","special","vargas"]:
        d[key] = normalize_list(d.get(key, []))
    inferred = infer_chart_refs(foundation, interpretation, summary)
    merged = merge_chart_refs(d, inferred)
    return json.dumps(merged, ensure_ascii=False)

def adjusted_confidence(current: str, notes: str) -> str:
    try:
        base = float(current)
    except Exception:
        base = 0.5
    if "Consolidated from" in (notes or ""):
        base = max(base, 0.55)
        base = min(0.9, base + 0.10)
    return f"{base:.2f}"

def main():
    if not os.path.exists(INPUT):
        raise SystemExit(f"Missing {INPUT}")

    df = pd.read_csv(INPUT, dtype=str).fillna("")
     # --- ADD THIS LINE HERE ---
    df.drop_duplicates(subset=['Fact_ID'], keep='first', inplace=True)
    # -------------------------
    cleaned_rows = []
    for _, row in df.iterrows():
        fp = canonicalize_foundation_point(row.get("Foundation_Point",""), row.get("Interpretation_Text",""))
        interp = fix_mojibake(row.get("Interpretation_Text",""))
        summ = clean_summary(row.get("AI_Astro_Summary",""), interp, fp)
        cref = clean_chart_refs_json(row.get("Chart_Refs_JSON",""), fp, interp, summ)
        conf = adjusted_confidence(row.get("Confidence_Score","0.5"), row.get("Notes",""))
        status = (row.get("Status","") or "NEEDS_REVIEW").strip()
        new_row = row.to_dict()
        new_row.update({
            "Foundation_Point": fp,
            "Interpretation_Text": interp,
            "AI_Astro_Summary": summ,
            "Chart_Refs_JSON": cref,
            "Confidence_Score": conf,
            "Status": status,
            "Last_Updated": now_iso(),
        })
        cleaned_rows.append(new_row)

    out_df = pd.DataFrame(cleaned_rows, columns=df.columns)

    # Final mojibake sweep across all string columns
    for col in out_df.columns:
        if out_df[col].dtype == object:
            out_df[col] = out_df[col].apply(lambda x: fix_mojibake(x) if pd.notna(x) else x)

    out_df.to_csv(OUTPUT, index=False)
    print(f"✅ Cleaned {len(out_df)} rows → {OUTPUT}")

if __name__ == "__main__":
    main()