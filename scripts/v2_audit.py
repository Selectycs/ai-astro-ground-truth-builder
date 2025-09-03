# -*- coding: utf-8 -*-
"""
v2_audit.py (Definitive HTML Version)

This script runs a comprehensive audit on a cleaned knowledge base file
and generates a single, detailed HTML report on the KB's completeness.
"""

import pandas as pd
import os
import json
from jinja2 import Template

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

# Using your full, categorized list of calculation modules
CALCULATION_MODULES = {
    "Core Placements": ["rashi_bhava_placements", "natal_planet_to_planet_conjunctions"],
    "Divisional & Special Charts": ["divisional_charts", "special_charts", "arudha_padas", "upagrahas", "special_lagnas"],
    "Strength Metrics": ["shadbala", "avasthas", "ishta_kashta_bala", "vimsopaka_score", "vargottama_status", "retro_combust_atichara"],
    "Predictive Systems": ["vimshottari_dashas", "other_dasha_systems", "sade_sati_panoti_timelines", "life_span_estimation"],
    "Panchanga & Nakshatras": ["navatara", "nakshatra_pada_per_planet"],
    "Aspects & Relationships": ["graha_drishti_tables", "friendship"],
    "Special Combinations": ["yogas", "doshas"],
    "Point-Based Systems": ["ashtakavarga_point_grids", "shodashavarga_summary"],
    "Core Birth Facts": ["atmakaraka", "karak_summary"]
}


def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """Loads the cleaned CSV and prepares it for auditing."""
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path, dtype=str).fillna("")
    return df

def audit_theme_coverage(df: pd.DataFrame) -> dict:
    """Audits and structures the distribution of facts across the full taxonomy hierarchy."""
    if df.empty: return {}
    
    df_exploded = df.assign(sub_theme=df['sub_theme'].str.split(',')).explode('sub_theme')
    df_exploded['sub_theme'] = df_exploded['sub_theme'].str.strip()
    sub_theme_counts = df_exploded['sub_theme'].value_counts().to_dict()

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
    """Audits and structures the KB for facts about each planet in each house and sign."""
    if df.empty: return {}
    
    results = []
    for planet in PLANETS:
        planet_data = {"name": planet.upper(), "houses": [], "signs": []}
        
        # Houses
        for house in HOUSES:
            search_str_1 = f'"planet": "{planet}"'
            search_str_2 = f'"house": {house}'
            is_found = not df[(df['astrological_trigger_json'].str.contains(search_str_1, case=False)) &
                              (df['astrological_trigger_json'].str.contains(search_str_2, case=False))].empty
            planet_data["houses"].append({"name": house, "found": is_found})
            
        # Signs
        for sign in SIGNS:
            search_str_1 = f'"planet": "{planet}"'
            search_str_2 = f'"sign": "{sign}"'
            is_found = not df[(df['astrological_trigger_json'].str.contains(search_str_1, case=False)) &
                              (df['astrological_trigger_json'].str.contains(search_str_2, case=False))].empty
            planet_data["signs"].append({"name": sign.capitalize(), "found": is_found})
        results.append(planet_data)
    return {"planets": results}

def audit_calculation_module_coverage(df: pd.DataFrame) -> dict:
    """Audits and structures coverage for specific, named calculation modules."""
    if df.empty: return {}
    
    results = []
    for category, modules in CALCULATION_MODULES.items():
        category_data = {"name": category, "modules": []}
        for module in modules:
            # A more robust search for the specific key-value pair
            search_str = f'"name": "{module}"'
            count = len(df[df['astrological_trigger_json'].str.contains(search_str, regex=False)])
            category_data["modules"].append({"name": module, "count": count})
        results.append(category_data)
    return {"categories": results}

def generate_html_report(audit_data: dict, output_path: str):
    """Generates a styled HTML report from the audit data."""
    
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Knowledge Base Audit Report</title>
        <style>
            body { font-family: sans-serif; margin: 2em; background-color: #f4f4f9; color: #333; }
            h1, h2, h3 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px;}
            .container { background: #fff; padding: 2em; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.1); }
            .section { margin-bottom: 2em; }
            .theme-block, .category-block { margin-bottom: 1.5em; }
            .theme-header, .category-header { font-size: 1.2em; font-weight: bold; color: #3498db; }
            .group-header { font-weight: bold; margin-top: 1em; }
            .sub-theme-list, .module-list { list-style-type: none; padding-left: 20px; }
            .sub-theme-list li, .module-list li { display: flex; justify-content: space-between; padding: 5px; border-bottom: 1px solid #eee; }
            .sub-theme-list li:last-child { border-bottom: none; }
            .zero-count { color: #e74c3c; font-weight: bold; }
            .planet-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 1.5em; }
            .planet-card { border: 1px solid #ddd; padding: 1em; border-radius: 5px; }
            .status-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 5px; margin-top: 0.5em; }
            .status-box { text-align: center; padding: 2px; border-radius: 3px; font-size: 0.8em; }
            .found { background-color: #2ecc71; color: white; }
            .missing { background-color: #e74c3c; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Knowledge Base Audit Report</h1>
            <p>Generated on: {{ audit_data.timestamp }}</p>
            <p><strong>Total Facts Audited: {{ audit_data.total_facts }}</strong></p>

            <div class="section">
                <h2>🔬 AUDIT 2: HIERARCHICAL THEMATIC COVERAGE</h2>
                {% for theme in audit_data.themes.themes %}
                <div class="theme-block">
                    <div class="theme-header">{{ theme.name }} (Total: {{ theme.total }})</div>
                    {% for group in theme.groups %}
                    <p class="group-header">{{ group.name }} (Group Total: {{ group.total }})</p>
                    <ul class="sub-theme-list">
                        {% for sub_theme in group.sub_themes %}
                        <li>
                            <span>{{ sub_theme.name }}</span>
                            <span class="{{ 'zero-count' if sub_theme.count == 0 }}">{{ sub_theme.count }}</span>
                        </li>
                        {% endfor %}
                    </ul>
                    {% endfor %}
                </div>
                {% endfor %}
            </div>

            <div class="section">
                <h2>🔬 AUDIT 1: PLANETARY PLACEMENT COVERAGE</h2>
                <div class="planet-grid">
                {% for planet in audit_data.planets.planets %}
                    <div class="planet-card">
                        <h3>{{ planet.name }}</h3>
                        <h4>Houses</h4>
                        <div class="status-grid">
                        {% for house in planet.houses %}
                            <div class="status-box {{ 'found' if house.found else 'missing' }}">{{ house.name }}</div>
                        {% endfor %}
                        </div>
                        <h4>Signs</h4>
                        <div class="status-grid">
                        {% for sign in planet.signs %}
                            <div class="status-box {{ 'found' if sign.found else 'missing' }}">{{ sign.name }}</div>
                        {% endfor %}
                        </div>
                    </div>
                {% endfor %}
                </div>
            </div>

            <div class="section">
                <h2>🔬 AUDIT 3: CALCULATION MODULE COVERAGE</h2>
                {% for category in audit_data.modules.categories %}
                <div class="category-block">
                    <div class="category-header">{{ category.name }}</div>
                    <ul class="module-list">
                        {% for module in category.modules %}
                        <li>
                            <span>{{ module.name }}</span>
                            <span class="{{ 'zero-count' if module.count == 0 }}">{{ module.count }}</span>
                        </li>
                        {% endfor %}
                    </ul>
                </div>
                {% endfor %}
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
    from datetime import datetime
    
    input_file = os.path.join(CLEANED_DATA_PATH, "interpretations.cleaned.csv")
    df = load_and_prepare_data(input_file)
    
    if df is None or df.empty:
        print("Audit cannot proceed without data.")
        return

    # Gather all audit data first
    audit_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_facts": len(df),
        "themes": audit_theme_coverage(df),
        "planets": audit_planet_coverage(df),
        "modules": audit_calculation_module_coverage(df)
    }

    # Generate the final HTML report
    output_path = os.path.join(CLEANED_DATA_PATH, "audit_report.html")
    generate_html_report(audit_data, output_path)

if __name__ == "__main__":
    main()