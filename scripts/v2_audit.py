# -*- coding: utf-8 -*-
"""
v2_audit.py (Definitive HTML Version)

This script runs a comprehensive audit on a cleaned knowledge base file
and generates a single, detailed HTML report on the KB's completeness,
including an "At a Glance" summary of low-coverage areas.
"""

import pandas as pd
import os
import json
from jinja2 import Template
from datetime import datetime

# --- Import KB Utilities ---
from v2_kb_utils import (
    TAXONOMY,
    CLEANED_DATA_PATH
)

# --- MASTER CHECKLISTS FOR AUDIT ---
PLANETS = ["sun", "moon", "mars", "mercury", "jupiter", "venus", "saturn", "rahu", "ketu"]
HOUSES = list(range(1, 13))
SIGNS = [
    "aries", "taurus", "gemini", "cancer", "leo", "virgo", "libra",
    "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"
]
NAKSHATRAS = [
    "ashwini", "bharani", "krittika", "rohini", "mrigashira", "ardra", "punarvasu",
    "pushya", "ashlesha", "magha", "purva_phalguni", "uttara_phalguni", "hasta",
    "chitra", "swati", "vishakha", "anuradha", "jyeshtha", "mula", "purva_ashadha",
    "uttara_ashadha", "shravana", "dhanishtha", "shatabhisha", "purva_bhadrapada",
    "uttara_bhadrapada", "revati"
]
CALCULATION_MODULES = {
    "Core Placements": ["rashi_bhava_placements", "natal_planet_to_planet_conjunctions", "static_placement"],
    "Divisional & Special Charts": ["divisional_charts", "special_charts", "arudha_padas", "upagrahas", "special_lagnas"],
    "Strength Metrics": ["shadbala", "avasthas", "ishta_kashta_bala", "vimsopaka_score", "vargottama_status", "retro_combust_atichara", "strength_calculation"],
    "Predictive Systems": ["vimshottari_dashas", "other_dasha_systems", "sade_sati_panoti_timelines", "life_span_estimation", "dasha_system"],
    "Panchanga & Nakshatras": ["navatara", "nakshatra_pada_per_planet"],
    "Aspects & Relationships": ["graha_drishti_tables", "friendship", "aspect", "relative_placement"],
    "Special Combinations": ["yogas", "doshas", "yoga_status", "dosha_check"],
    "Point-Based Systems": ["ashtakavarga_point_grids", "shodashavarga_summary", "point_system"],
    "Core Birth Facts": ["atmakaraka", "karak_summary"]
}

def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """Loads the cleaned CSV and safely parses the JSON column."""
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path, dtype=str).fillna("")
    def safe_json_loads(s):
        try: return json.loads(s)
        except (json.JSONDecodeError, TypeError): return {}
    df['trigger_obj'] = df['astrological_trigger_json'].apply(safe_json_loads)
    return df

def get_theme_data(df: pd.DataFrame) -> dict:
    if df.empty: return {}
    df_exploded = df.assign(sub_theme=df['sub_theme'].str.split(',')).explode('sub_theme')
    df_exploded['sub_theme'] = df_exploded['sub_theme'].str.strip()
    return df_exploded['sub_theme'].value_counts().to_dict()

def audit_at_a_glance(theme_counts: dict, planet_data: dict, module_data: dict, lordship_data: dict) -> dict:
    LOW_COVERAGE_THRESHOLD = 5
    zero_coverage, low_coverage = [], []
    for theme, groups in TAXONOMY.items():
        for group, sub_themes in groups.items():
            for sub_theme in sub_themes:
                count = theme_counts.get(sub_theme, 0)
                item = {"type": "Sub-Theme", "name": f"{theme} -> {sub_theme}", "count": count}
                if count == 0: zero_coverage.append(item)
                elif count < LOW_COVERAGE_THRESHOLD: low_coverage.append(item)
    for planet in planet_data.get("planets", []):
        for item in planet.get("houses", []) + planet.get("signs", []) + planet.get("nakshatras", []):
            if not item['found']:
                zero_coverage.append({"type": "Planetary Placement", "name": f"{planet['name']} in {item['name']}", "count": 0})
    for item in lordship_data.get("house_lords_missing", []):
        zero_coverage.append({"type": "Lordship Placement", "name": item, "count": 0})
    for category in module_data.get("categories", []):
        for module in category.get("modules", []):
            item = {"type": "Calculation Module", "name": module['name'], "count": module['count']}
            if module['count'] == 0: zero_coverage.append(item)
            elif module['count'] < LOW_COVERAGE_THRESHOLD: low_coverage.append(item)
    return {
        "zero_coverage_items": sorted(zero_coverage, key=lambda x: x['type']),
        "low_coverage_items": sorted(low_coverage, key=lambda x: (x['type'], x['count'])),
        "threshold": LOW_COVERAGE_THRESHOLD
    }

def audit_theme_coverage(df: pd.DataFrame, sub_theme_counts: dict) -> dict:
    results = []
    for theme, groups in TAXONOMY.items():
        theme_total = df[df['theme'] == theme].shape[0]
        theme_data = {"name": theme, "total": theme_total, "groups": []}
        for group, sub_themes in groups.items():
            group_total = sum(sub_theme_counts.get(st, 0) for st in sub_themes)
            group_data = {"name": group, "total": group_total, "sub_themes": []}
            for sub_theme in sub_themes:
                count = sub_theme_counts.get(sub_theme, 0)
                group_data["sub_themes"].append({"name": sub_theme, "count": count})
            theme_data["groups"].append(group_data)
        results.append(theme_data)
    return {"themes": results}

def audit_planet_coverage(df: pd.DataFrame) -> dict:
    results = []
    for planet in PLANETS:
        planet_data = {"name": planet.upper(), "houses": [], "signs": [], "nakshatras": []}
        # Houses, Signs, and Nakshatras
        for house in HOUSES:
            is_found = any(p.get('planet') == planet and p.get('house') == house for c in df['trigger_obj'].apply(lambda x: x.get('components', [])) for p in c)
            planet_data["houses"].append({"name": house, "found": is_found})
        for sign in SIGNS:
            is_found = any(p.get('planet') == planet and p.get('sign') == sign for c in df['trigger_obj'].apply(lambda x: x.get('components', [])) for p in c)
            planet_data["signs"].append({"name": sign.capitalize(), "found": is_found})
        for nakshatra in NAKSHATRAS:
            is_found = any(p.get('planet') == planet and p.get('nakshatra') == nakshatra for c in df['trigger_obj'].apply(lambda x: x.get('components', [])) for p in c)
            planet_data["nakshatras"].append({"name": nakshatra.capitalize(), "found": is_found})
        results.append(planet_data)
    return {"planets": results}

def audit_calculation_module_coverage(df: pd.DataFrame) -> dict:
    module_counts = {}
    for trigger in df['trigger_obj']:
        context = trigger.get('calculation_context', {})
        if context:
            key = context.get('name') or context.get('type')
            if key: module_counts[key] = module_counts.get(key, 0) + 1
    results = []
    for category, modules in CALCULATION_MODULES.items():
        category_data = {"name": category, "modules": []}
        for module in modules:
            count = module_counts.get(module, 0)
            category_data["modules"].append({"name": module, "count": count})
        results.append(category_data)
    return {"categories": results}

def audit_lordship_coverage(df: pd.DataFrame) -> dict:
    found_count, total_count, missing_items = 0, 0, []
    for lord_of_house in HOUSES:
        for in_house in HOUSES:
            total_count += 1
            is_found = any(
                p.get('type') == 'lord_in_house' and p.get('lord_of_house') == lord_of_house and p.get('house') == in_house
                for components in df['trigger_obj'].apply(lambda x: x.get('components', [])) for p in components
            )
            if is_found: found_count += 1
            else: missing_items.append(f"Lord of {lord_of_house} in House {in_house}")
    return {
        "house_lords_found": found_count, "house_lords_total": total_count,
        "house_lords_missing": missing_items[:20]
    }

def generate_html_report(audit_data: dict, output_path: str):
    """Generates the complete, styled HTML report from the audit data."""
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8"><title>Knowledge Base Audit Report</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 2em; background-color: #f8f9fa; color: #212529; }
            h1, h2, h3, h4 { color: #0056b3; border-bottom: 2px solid #007bff; padding-bottom: 8px;}
            h4 { border-bottom: 1px solid #dee2e6; }
            .container { background: #fff; padding: 2em; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); max-width: 1400px; margin: auto; }
            .section { margin-bottom: 2.5em; }
            .theme-block, .category-block { margin-bottom: 1.5em; border-left: 3px solid #007bff; padding-left: 15px; }
            .theme-header, .category-header { font-size: 1.3em; font-weight: 600; color: #0056b3; }
            .group-header { font-weight: bold; margin-top: 1em; color: #343a40; }
            .item-list { list-style-type: none; padding-left: 20px; }
            .item-list li { display: flex; justify-content: space-between; padding: 8px; border-bottom: 1px solid #dee2e6; transition: background-color 0.2s; }
            .item-list li:hover { background-color: #f1f3f5; }
            .zero-count { color: #dc3545; font-weight: bold; }
            .low-count { color: #fd7e14; font-weight: bold; }
            .planet-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 1.5em; }
            .planet-card { border: 1px solid #ced4da; padding: 1em; border-radius: 5px; background: #fff; }
            .status-grid { display: grid; gap: 5px; margin-top: 0.5em; }
            .status-grid-houses { grid-template-columns: repeat(6, 1fr); }
            .status-grid-signs { grid-template-columns: repeat(4, 1fr); }
            .status-grid-nakshatras { grid-template-columns: repeat(9, 1fr); font-size: 0.75em; }
            .status-box { text-align: center; padding: 4px; border-radius: 3px; }
            .found { background-color: #28a745; color: white; }
            .missing { background-color: #dc3545; color: white; }
            .summary-table { width: 100%; border-collapse: collapse; margin-top: 1em; }
            .summary-table th, .summary-table td { border: 1px solid #dee2e6; padding: 8px; text-align: left; }
            .summary-table th { background-color: #e9ecef; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Knowledge Base Audit Report</h1>
            <p>Generated on: {{ audit_data.timestamp }}</p>
            <p><strong>Total Facts Audited: {{ audit_data.total_facts }}</strong></p>

            <div class="section">
                <h2>📊 At a Glance: Gaps & Low Coverage Areas</h2>
                <h3>Zero Coverage Items (Count = 0)</h3>
                {% if audit_data.at_a_glance.zero_coverage_items %}<table class="summary-table">
                    <tr><th>Type</th><th>Name</th></tr>
                    {% for item in audit_data.at_a_glance.zero_coverage_items %}<tr><td>{{ item.type }}</td><td>{{ item.name }}</td></tr>{% endfor %}
                </table>{% else %}<p style="color: #28a745;"><strong>Excellent! No zero-coverage areas found.</strong></p>{% endif %}
                <h3 style="margin-top: 2em;">Low Coverage Items (Count < {{ audit_data.at_a_glance.threshold }})</h3>
                {% if audit_data.at_a_glance.low_coverage_items %}<table class="summary-table">
                    <tr><th>Type</th><th>Name</th><th>Count</th></tr>
                    {% for item in audit_data.at_a_glance.low_coverage_items %}<tr><td>{{ item.type }}</td><td>{{ item.name }}</td><td class="low-count">{{ item.count }}</td></tr>{% endfor %}
                </table>{% else %}<p style="color: #28a745;"><strong>Excellent! No low-coverage areas found.</strong></p>{% endif %}
            </div>

            <div class="section">
                <h2>🪐 Planetary Placement Coverage</h2>
                <div class="planet-grid">
                {% for planet in audit_data.planets.planets %}
                    <div class="planet-card">
                        <h3>{{ planet.name }}</h3><h4>Houses</h4><div class="status-grid status-grid-houses">
                        {% for item in planet.houses %}<div class="status-box {{ 'found' if item.found else 'missing' }}">{{ item.name }}</div>{% endfor %}
                        </div><h4>Signs</h4><div class="status-grid status-grid-signs">
                        {% for item in planet.signs %}<div class="status-box {{ 'found' if item.found else 'missing' }}">{{ item.name }}</div>{% endfor %}
                        </div><h4>Nakshatras</h4><div class="status-grid status-grid-nakshatras">
                        {% for item in planet.nakshatras %}<div class="status-box {{ 'found' if item.found else 'missing' }}">{{ item.name }}</div>{% endfor %}
                        </div></div>
                {% endfor %}
                </div>
            </div>
            
            <div class="section">
                <h2>📜 Lordship Placement Coverage</h2>
                {% set data = audit_data.lordships %}
                <p><strong>Lord of House in House Coverage:</strong> {{ data.house_lords_found }} / {{ data.house_lords_total }} Covered ({{ "%.1f"|format((data.house_lords_found/data.house_lords_total)*100) }}%)</p>
                {% if data.house_lords_missing %}
                    <h4>First {{ data.house_lords_missing|length }} Missing Combinations:</h4>
                    <ul class="item-list" style="max-height: 200px; overflow-y: auto; border: 1px solid #eee;">
                    {% for item in data.house_lords_missing %}<li>{{ item }}</li>{% endfor %}
                    </ul>
                {% endif %}
            </div>

            <div class="section">
                <h2>🔬 Hierarchical Thematic Coverage</h2>
                {% for theme in audit_data.themes.themes %}<div class="theme-block">
                    <div class="theme-header">{{ theme.name }} (Total: {{ theme.total }})</div>
                    {% for group in theme.groups %}<p class="group-header">{{ group.name }} (Group Total: {{ group.total }})</p>
                    <ul class="item-list">
                        {% for sub_theme in group.sub_themes %}
                        <li><span>{{ sub_theme.name }}</span><span class="{{ 'zero-count' if sub_theme.count == 0 }}">{{ sub_theme.count }}</span></li>
                        {% endfor %}
                    </ul>{% endfor %}
                </div>{% endfor %}
            </div>

            <div class="section">
                <h2>⚙️ Calculation Module Coverage</h2>
                {% for category in audit_data.modules.categories %}<div class="category-block">
                    <div class="category-header">{{ category.name }}</div>
                    <ul class="item-list module-list">
                        {% for module in category.modules %}
                        <li><span>{{ module.name }}</span><span class="{{ 'zero-count' if module.count == 0 }}">{{ module.count }}</span></li>
                        {% endfor %}
                    </ul></div>{% endfor %}
            </div>
        </div>
    </body>
    </html>
    """
    
    template = Template(html_template)
    rendered_html = template.render(audit_data=audit_data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(rendered_html)
    print(f"\n✅ HTML report generated successfully at '{output_path}'")

def main():
    """Main function to run the full audit."""
    input_file = os.path.join(CLEANED_DATA_PATH, "interpretations.cleaned.csv")
    df = load_and_prepare_data(input_file)
    
    if df is None or df.empty:
        print("Audit cannot proceed without data.")
        return

    sub_theme_counts = get_theme_data(df)
    planet_data = audit_planet_coverage(df)
    module_data = audit_calculation_module_coverage(df)
    lordship_data = audit_lordship_coverage(df)
    
    audit_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_facts": len(df),
        "at_a_glance": audit_at_a_glance(sub_theme_counts, planet_data, module_data, lordship_data),
        "themes": audit_theme_coverage(df, sub_theme_counts),
        "planets": planet_data,
        "modules": module_data,
        "lordships": lordship_data
    }

    output_path = os.path.join(CLEANED_DATA_PATH, "audit_report.html")
    generate_html_report(audit_data, output_path)

if __name__ == "__main__":
    main()