# -*- coding: utf-8 -*-
"""
v2_kb_utils.py

This utility module provides shared constants, schemas, and helper functions
for the V2 AI Astrology Knowledge Base (KB) build pipeline. It serves as the
single source of truth for data structures and file paths used across all
pipeline scripts (ingest, reconcile, consolidate, clean).
"""

import os
import pandas as pd
from datetime import datetime
import hashlib
import json
from typing import Dict, List, Any, Optional

# --- V2 PIPELINE PATHS ---
# A structured directory for managing intermediate files from the pipeline.
KB_PIPELINE_DIR = "kb_pipeline_v2"
RAW_DATA_PATH = os.path.join(KB_PIPELINE_DIR, "00_raw")
VALIDATED_DATA_PATH = os.path.join(KB_PIPELINE_DIR, "01_validated")
CONSOLIDATED_DATA_PATH = os.path.join(KB_PIPELINE_DIR, "02_consolidated")
CLEANED_DATA_PATH = os.path.join(KB_PIPELINE_DIR, "03_cleaned")
CORE_CONCEPTS_PATH = os.path.join(KB_PIPELINE_DIR, "core_concepts")


# --- V2 CORE SCHEMAS ---

# The definitive schema for the final interpretations table in PostgreSQL.
# In v2_kb_utils.py
# In v2_kb_utils.py

V2_ANNOTATED_SCHEMA = [
    "fact_id",
    "raw_text",
    "interpretation_text",
    "topic",                 # ADDED
    "header",
    "mark_for_deletion",
    "schema_type",
    "prior_type",            # Added back
    "is_conceptual_start",
    "source_name",
    "source_type",
    "source_reference",
    "paragraph_id",
    "sentence_in_paragraph_id",
    "status",
    "last_updated",
    "notes"
]

# The Final Canonical Schema remains the same
# In v2_kb_utils.py

# This schema defines the full output of the v2_reconcile.py script
V2_CANONICAL_SCHEMA = [
    'fact_id', 'parent_fact_id', 'astrological_trigger_json', 'raw_text',
    'interpretation_text', 'interpretation_summary_ai', 'topic', 'header',
    'theme', 'theme_group', 'sub_theme', 'schema_type', 'is_conceptual_start',
    'source_name', 'source_type', 'source_reference', 'status', 'notes',
    'confidence_score', 'last_updated'
]

# Schemas for core concept files (largely unchanged, but maintained for clarity).
CORE_CONCEPT_SCHEMAS = {
    "planets": [
        "fact_id", "concept_group", "concept_name", "sanskrit_name", "keywords",
        "general_description", "dignities_json", "karaka_json", "attributes_json",
        "source_name", "source_type", "source_reference", "last_updated"
    ],
    "signs": [
        "fact_id", "concept_group", "concept_name", "sanskrit_name", "keywords",
        "description", "ruling_planet", "element", "modality", "gender",
        "body_part", "symbol",
        "source_name", "source_type", "source_reference", "last_updated"
    ],
    "houses": [
        "fact_id", "concept_group", "concept_name", "sanskrit_name", "keywords",
        "description", "karaka_planets_json", "related_houses_json",
        "source_name", "source_type", "source_reference", "last_updated"
    ],
    "nakshatras": [
        "fact_id", "concept_group", "concept_name", "ruling_planet", "symbol", "deity", "description",
        "source_name", "source_type", "source_reference", "last_updated"
    ],
    "yogas_doshas": [
        "fact_id", "concept_group", "concept_name", "type", "definition",
        "source_name", "source_type", "source_reference", "last_updated"
    ],
    "timing_systems": [
        "fact_id", "concept_group", "concept_name", "system_type", "definition",
        "source_name", "source_type", "source_reference", "last_updated"
    ],
    "astrological_terms": [
        "fact_id", "concept_group", "concept_name", "definition",
        "source_name", "source_type", "source_reference", "last_updated"
    ]
}


# --- THEMATIC TAXONOMY ---
# This is the master classification system used by the AI to categorize interpretations.
TAXONOMY = {
    'Foundational Concepts': {
        'Astrological Principles': ['classification', 'definition', 'general_principle']
    },
    'core_self_vitality': {
        'Identity & Personality': [
            'self_identity_character', 'vitality_energy_levels', 'physical_appearance',
            'confidence_charisma', 'other_identity_personality'
        ],
        'Purpose & Direction': [
            'life_path_purpose', 'spiritual_self_alignment', 'resilience_adaptability',
            'independence_autonomy', 'other_purpose_direction'
        ],
        'Other': ['other_core_self_related']
    },
    'relationships': {
        'Romantic & Marriage': [
            'marriage_spouse', 'love_relationships', 'intimacy_sexuality', 'other_romantic_marriage'
        ],
        'Social & Family Relations': [
            'friendships_social_circle', 'siblings_relations', 'parental_relations',
            'children_relations', 'inlaws_extended_family', 'other_social_family'
        ],
        'Public Interaction': [
            'partnerships_collaborations', 'social_influence_presence', 'disputes_conflicts',
            'other_public_interaction'
        ],
        'Other': ['other_relationships_related']
    },
    'career_public_life': {
        'Prominence & Governance': [
            'authority_leadership', 'public_image_reputation', 'politics_public_service',
            'networking_influence', 'military_defense_security', 'other_prominence_governance'
        ],
        'Profession & Field': [
            'independent_business_entrepreneurship', 'service_employment', 'creative_arts_entertainment',
            'communication_media_publishing', 'tech_science_engineering', 'education_academia_research',
            'healing_medical_professions', 'finance_trade_commerce', 'legal_professions',
            'spiritual_occult_professions', 'sports_athletics', 'other_profession_type'
        ],
        'Career Dynamics': [
            'income_from_profession', 'career_stability_longevity', 'career_mobility_changes',
            'career_peak_timing', 'foreign_career_relocation', 'remote_work_global_teams',
            'other_career_dynamics'
        ],
        'Other': ['other_career_related']
    },
    'wealth_assets_resources': {
        'Income & Earnings': [
            'income_primary', 'income_secondary_sources', 'speculative_gains_losses',
            'inheritance_unearned_income', 'dhana_yogas', 'other_income_earnings'
        ],
        'Assets & Property': [
            'property_real_estate', 'movable_assets', 'savings_investments',
            'debts_liabilities', 'other_assets_property', 'luxury_possessions'
        ],
        'Financial Dynamics': [
            'wealth_stability', 'sudden_gains_losses', 'financial_risks', 'other_financial_dynamics'
        ],
        'Other': ['other_wealth_related']
    },
    'learning_education_creativity': {
        'Formal Education': [
            'primary_education', 'higher_education', 'research_academic_success',
            'foreign_education', 'other_formal_education'
        ],
        'Skills & Creativity': [
            'arts_performance', 'writing_expression', 'intellectual_skills',
            'innovation_invention', 'other_skills_creativity'
        ],
        'Knowledge & Wisdom': [
            'spiritual_learning','spiritual_growth', 'philosophical_understanding', 'practical_skills',
            'lifelong_learning', 'other_knowledge_wisdom'
        ],
        'Other': ['other_learning_related']
    },
    'family_lineage': {
        'Parents & Ancestors': [
            'father_status_influence', 'mother_status_influence', 'ancestral_heritage',
            'family_reputation', 'other_parents_ancestors'
        ],
        'Children & Descendants': [
            'childbearing_potential', 'child_success', 'lineage_continuity', 'other_children_descendants'
        ],
        'Extended Family Dynamics': [
            'siblings_influence', 'cousins_extended_lineage', 'family_support_conflicts',
            'other_extended_family'
        ],
        'Other': ['other_family_related']
    },
    'home_property_vehicles': {
        'Residence & Property': [
            'home_comforts', 'property_acquisition', 'debts_liabilities', 'relocation_residence_changes', 'other_residence_property'
        ],
        'Vehicles & Movables': [
            'vehicle_ownership', 'travel_comforts', 'luxury_possessions', 'movable_assets', 'other_vehicles_movables'
        ],
        'Domestic Environment': [
            'domestic_harmony', 'parental_care', 'family_security', 'other_domestic_environment'
        ],
        'Other': ['other_home_related']
    },
    'health_risks_longevity': {
        'Physical Health': [
            'general_health', 'chronic_diseases', 'accidents_injuries', 'immunity_strength', 'vitality_energy_levels', 'other_physical_health'
        ],
        'Mental & Emotional Health': [
            'stress_anxiety', 'emotional_resilience', 'psychological_balance', 'other_mental_emotional_health'
        ],
        'Longevity & Risks': [
            'lifespan_overview', 'risk_periods', 'sudden_endings', 'other_longevity_risks'
        ],
        'Other': ['other_health_related']
    },
    'travel_dharma_spirituality': {
        'Travel & Journeys': [
            'short_travel_local', 'long_travel_foreign', 'pilgrimage_spiritual_travel',
            'relocation_abroad', 'other_travel_journeys'
        ],
        'Dharma & Spiritual Path': [
            'religious_alignment', 'spiritual_learning', 'spiritual_growth', 'philosophical_beliefs',
            'charitable_service', 'other_dharma_spirituality'
        ],
        'Other': ['other_travel_spirituality_related']
    },
    'karma_past_life': {
        'Past Life Patterns': [
            'past_life_professions', 'past_life_relationships', 'karmic_debts_lessons',
            'karmic_blessings', 'rahu_ketu_affliction', 'other_past_life_patterns','pitru_dosha'
        ],
        'Moksha & Liberation': [
            'spiritual_progress', 'obstacles_to_moksha', 'soul_purpose_alignment', 'moksha_factors',
            'other_moksha_factors'
        ],
        'Ancestral & Collective Karma': [
            'ancestral_karmas', 'family_lineage_debts', 'collective_past_life_influence',
            'other_karmic_influences'
        ],
        'Other': ['other_karma_related']
    },
    'yogas': {
        'Strength & Success Yogas': ['raja_yogas', 'dhana_yogas', 'gajakesari_yoga', 'pancha_mahapurusha_yogas','other_strength_success_yogas'],
        'Spiritual & Knowledge Yogas': ['saraswati_yoga', 'chandra_guru_yoga', 'moksha_yogas', 'other_spiritual_knowledge_yogas'],
        'Miscellaneous Yogas': ['uncommon_special_yogas', 'other_miscellaneous_yogas'],
        'Other': ['other_yogas_related']
    },
    'doshas': {
        'Affliction Doshas': [
            'mangal_dosha', 'kaal_sarpa_dosha', 'pitru_dosha', 'grahan_dosha', 'other_affliction_doshas'
        ],
        'Malefic Influences': [
            'saturn_affliction', 'rahu_ketu_affliction', 'other_malefic_influences'
        ],
        'Other': ['other_doshas_related']
    },
    'timing_windows': {
        'Dashas & Sub-Dashas': [
            'vimshottari_dasha', 'yogini_dasha', 'chara_dasha', 'other_dashas'
        ],
        'Transits & Periods': [
            'sade_sati', 'saturn_transits', 'jupiter_transits', 'eclipses_periods', 'other_transits_periods'
        ],
        'Panchang & Lunar Windows': [
            'tithi_yoga_karana', 'eclipses_nakshatra_windows', 'other_lunar_windows'
        ],
        'Other': ['other_timing_related']
    },
    'other': {
    'Other': ['other']
}
}


# --- HELPER FUNCTIONS ---

def generate_id(text_to_hash: str, length: int = 16) -> str:
    """
    Generates a unique and deterministic ID from a string.

    Best practice is to hash a stable, unique combination of fields, such as
    the canonical string representation of the Astrological_Trigger_JSON.

    Args:
        text_to_hash: The string to be hashed.
        length: The desired length of the output hex digest.

    Returns:
        A unique hexadecimal string ID.
    """
    return hashlib.sha256(text_to_hash.encode('utf-8')).hexdigest()[:length]


def save_df(df: pd.DataFrame, file_path: str, file_name: str):
    """
    Saves a Pandas DataFrame to a CSV file, creating the directory if needed.
    It appends to the file if it exists, otherwise creates a new one with a header.

    Args:
        df: The Pandas DataFrame to save.
        file_path: The directory where the file should be saved.
        file_name: The name of the CSV file.
    """
    if df.empty:
        print(f"INFO: DataFrame is empty. Skipping save for '{file_name}'.")
        return

    # Ensure the directory exists
    os.makedirs(file_path, exist_ok=True)
    
    output_csv = os.path.join(file_path, file_name)
    file_exists = os.path.exists(output_csv)

    try:
        df.to_csv(output_csv, mode='a', header=not file_exists, index=False)
        print(f"✅ Successfully saved {len(df)} rows to '{output_csv}'.")
    except Exception as e:
        print(f"❌ ERROR: Failed to save DataFrame to '{output_csv}'. Reason: {e}")

        # --- ASTROLOGICAL ENTITY LISTS ---
NAKSHATRAS = ["Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashirsha", "Ardra", "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni", "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha", "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishtha", "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"]
SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
VARGAS = ["Rasi", "Hora", "Drekkana", "Chaturthamsa", "Saptamsa", "Navamsa", "Dasamsa", "Dvadasamsa", "Shodasamsa", "Vimsamsa", "Siddhamsa", "Trimsamsa", "Shashtyamsa", "D1", "D2", "D3", "D4", "D7", "D9", "D10", "D12", "D16", "D20", "D24", "D30", "D60", "D-1", "D-2", "D-3", "D-4", "D-7", "D-9", "D-10", "D-12", "D-16", "D-20", "D-24", "D-30", "D-60"]
SPECIAL_POINTS = ["Arudha Lagna", "Upapada Lagna", "Bhava Lagna", "Hora Lagna", "Ghati Lagna"]
SPECIAL_CHARTS = ["Bhava Chalit", "Chandra Lagna", "Surya Lagna"]
PLANETS_TO = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]

# --- KEYWORD DEFINITIONS ---
SCHEMA_KEYWORDS: Dict[str, List[str]] = {
    "ashtakavarga": ["ashtakavarga", "bindus"], 
    "nakshatra_placement": ["nakshatra"] + NAKSHATRAS, 
    "vimsopaka": ["vimsopaka"], 
    "special_chart_placement": ["chalit chart"] + SPECIAL_CHARTS, 
    "avastha": ["avastha", "state of"], 
    "karaka": ["karaka", "significator"], 
    "special_point": ["special point"] + SPECIAL_POINTS, 
    "dasha": ["dasha", "mahadasha", "antardasha", "MD", "AD"], 
    "lordship": ["lord", "lordship", "ruler of", "house ruled"], 
    "shadbala": ["shadbala", "vargottama"], 
    "conjunction": ["conjunct", "conjunction", "together with"], 
    "aspect": ["aspect", "aspected by", "drishti", "aspects", "aspected", "aspects:","casts a full aspect"], 
    "varga": ["varga", "chart"] + VARGAS, 
    "yoga": ["yoga"], "dosha": ["dosha"], 
    "sign_trait": ["sign", "rashi", "modality"] + SIGNS, 
    "placement": ["placed in", "in the sign of", "in house"], 
    "house_group_definition": ["Kendra", "Trikona", "Dusthana", "Upachaya", "Angular", "Trinal"],
    "house_profile": ["Lagna", "Bhava", "house", "houses"],
    "planet_profile": PLANETS_TO + ["Planet", "Graha", "Grahas", "Exalted", "Debilitated", "Moolatrikona", "Own", "Neutral", "Enemy"],
    "varga_group_definition": ["Shadvargas", "Saptavargas", "Dashavargas", "Shodashavargas", "Varga Group"]
}

# Place this constant near the top of your v2_reconcile.py script
CANONICAL_SCHEMA_EXAMPLES_TEXT = """
1.  Placement: {"type": "placement", "planet_name": "Sun", "house": 7, "sign": "Sagittarius"}
2.  Lordship: {"type": "lordship", "lord_of_house": 10, "placed_in_house": 12, "planet_name": "Mercury"}
3.  Aspect: {"type": "aspect", "source_planet": "Mars", "target_house": 8, "relative_aspect_number": 8, "planet_nature": "Malefic", "source_sign": "Aries", "target_sign": "Taurus", "strength": "full"}
4.  Conjunction: {"type": "conjunction", "planet_names": ["Moon", "Saturn"], "house": 7}
5.  Yoga: {"type": "yoga", "name": "Gajakesari Yoga"}
6.  Dosha: {"type": "dosha", "name": "Mangal Dosha"}
7.  Karaka: {"type": "karaka", "karaka_type": "Atmakaraka", "planet_name": "Sun"}
8.  Special Point: {"type": "special_point", "point_name": "Arudha Lagna", "sign": "Aquarius"}
9.  Varga: {"type": "varga", "varga": "D9", "varga_name": "Hora", "domain": ["Wealth", "income", "resources"], "planet_name": "Venus", "sign": "Libra"}
10. Shadbala: {"type": "shadbala", "planet_name": "Saturn", "value_rupas": 1.15}
11. Vimsopaka: {"type": "vimsopaka", "planet_name": "Jupiter", "score": 15.5}
12. Dasha: {"type": "dasha", "system": "Vimshottari", "mahadasha_lord": "Saturn", "duration_years": 6, "antardasha_lord": "Sun", "pratyantardasha_lord": null, "condition": "general"}
13. Avastha: {"type": "avastha", "planet_name": "Mars", "state": "Bala"}
14. Ashtakavarga: {"type": "ashtakavarga", "scope": "sarvashtakavarga", "house": 7, "bindus": 32}
15. Special Chart Placement: {"type": "special_chart_placement", "chart_name": "Bhava Chalit", "planet_name": "Moon", "house": 4}
16. Sign Trait: {"type": "sign_trait", "sign": "Pisces", "element": "Water", "planet_name": "Sun", "house": 1, "modality": "Cardinal", "ruler": "Mars"}
17. Nakshatra Placement: {"type": "nakshatra_placement", "planet_name": "Venus", "nakshatra": "Purva Ashadha"}
18. House Group: {"type": "house_group_definition", "group_name": "Kendra", "houses": [1, 4, 7, 10], "description": "The pillars of the chart, representing the most active and important areas of life."}
19. House Profile: {"type": "house_profile", "house": 1, "classifications": ["Kendra", "Trikona"], "karakas": ["Sun"], "significations": ["The Self", "physical body", "appearance"]}
20. Planet Profile: {"type": "planet_profile", "planet_name": "Sun", "karaka_for": ["soul", "father", "status"], "state": "exalted", "sign": "Aries"}
21. Varga Group: {"type": "varga_group_definition", "group_name": "Shadvargas", "varga_count": 6, "included_vargas": ["D1", "D2", "D3", "D9", "D12", "D30"]}
"""

# In v2_kb_utils.py, add or replace with this schema definition

V2_PRODUCTION_SCHEMA = [
    "fact_id",
    "theme",
    "theme_group",
    "sub_theme",
    "raw_text",
    "astrological_trigger_json",
    "interpretation_text",
    "interpretation_summary_ai",
    "confidence_score",
    "source_name",
    "source_reference",
    "last_updated"
]


# v2_kb_utils.py

CANONICAL_SCHEMAS = {
    "placement": {
        "attributes": {
            "type": "string",
            "planet_name": "string",
            "house": "integer",
            "sign": "string"
        }
    },
    "lordship": {
        "attributes": {
            "type": "string",
            "lord_of_house": "integer",
            "placed_in_house": "integer",
            "planet_name": "string"
        }
    },
    "aspect": {
        "attributes": {
            "type": "string",
            "source_planet": "string",
            "target_house": "integer",
            "relative_aspect_number": "integer",
            "planet_nature": "string",
            "source_sign": "string",
            "target_sign": "string",
            "strength": "string"
        }
    },
    "conjunction": {
        "attributes": {
            "type": "string",
            "planet_names": "list[string]",
            "house": "integer"
        }
    },
   "yoga": {
        "attributes": {
            "type": "string",
            "name": "string",
            "components": "list[string]"   # e.g., ["Moon", "Jupiter", "4th house"]
        }
    },
    "dosha": {
        "attributes": {
            "type": "string",
            "name": "string",
            "components": "list[string]"   # e.g., ["Mars", "7th house"]
        }
    },
    "karaka": {
        "attributes": {
            "type": "string",
            "karaka_type": "string",
            "planet_name": "string",
            "signification": "string"      # e.g., "father", "soul", "status"
        }
    },
    "special_point": {
        "attributes": {
            "type": "string",
            "point_name": "string",
            "sign": "string"
        }
    },
    "varga": {
        "attributes": {
            "type": "string",
            "varga": "string",
            "varga_name": "string",
            "domain": "list[string]",
            "planet_name": "string",
            "sign": "string"
        }
    },
    "shadbala": {
        "attributes": {
            "type": "string",
            "planet_name": "string",
            "value_rupas": "float"
        }
    },
    "vimsopaka": {
        "attributes": {
            "type": "string",
            "planet_name": "string",
            "score": "float"
        }
    },
    "dasha": {
        "attributes": {
            "type": "string",
            "system": "string",
            "mahadasha_lord": "string",
            "duration_years": "float",
            "antardasha_lord": "string",
            "pratyantardasha_lord": "string",
            "condition": "string",
        }
    },
    "avastha": {
        "attributes": {
            "type": "string",
            "planet_name": "string",
            "state": "string"
        }
    },
    "ashtakavarga": {
        "attributes": {
            "type": "string",
            "scope": "string",
            "house": "integer",
            "bindus": "integer"
        }
    },
    "special_chart_placement": {
        "attributes": {
            "type": "string",
            "chart_name": "string",
            "planet_name": "string",
            "house": "integer"
        }
    },
    "sign_trait": {
        "attributes": {
            "type": "string",
            "sign": "string",
            "element": "string",
            "planet_name": "string",
            "house": "integer",
            "modality": "string",
            "ruler": "string"
        }
    },
    "nakshatra_placement": {
        "attributes": {
            "type": "string",
            "planet_name": "string",
            "nakshatra": "string"
        }
    },
    "house_group_definition": {
        "attributes": {
            "type": "string",
            "group_name": "string",
            "houses": "list[integer]",
            "description": "string"
        }
    },
    "house_profile": {
        "attributes": {
            "type": "string",
            "house": "integer",
            "planet_in_house": "string", # <-- ADDED for linking
            "aspected_by_planet": "string",
            "state": "",
            "classifications": "list[string]",
            "karakas": "list[string]",
            "significations": "list[string]"
        }
    },
    "planet_profile": {
        "attributes": {
            "type": "string",
            "planet_name": "string",
            "house": "integer", # <-- ADDED for linking
            "aspects_house": "integer",  # <-- ADD THIS
            "karaka_for": "list[string]",
            "state": "string",
            "sign": "string",
            "friends": "list[string]",      # <-- ADD THIS
            "enemies": "list[string]",      # <-- ADD THIS
            "neutral_to": "list[string]"  # <-- ADD THIS
        }
    },
    "varga_group_definition": {
        "attributes": {
            "type": "string",
            "group_name": "string",
            "varga_count": "integer",
            "included_vargas": "list[string]"
        }
    },
     # --- ADD THIS NEW ENTRY ---
    "unstructured": {
        "attributes": {
            "type": "string",
            "text": "string"
        }
    },
    # --- ADD THIS NEW ENTRY ---
    "prose": {
        "attributes": {
            "type": "string",
            "text": "string"
        }
    }
}


# In v2_kb_utils.py

# --- UPDATED: Canonical Entity Definitions ---
CANONICAL_ENTITIES = {
    "planet": {
        "Sun": ["Sun", "Surya"],
        "Moon": ["Moon", "Chandra"],
        "Mars": ["Mars", "Mangal", "Kuja"],
        "Mercury": ["Mercury", "Budha"],
        "Jupiter": ["Jupiter", "Guru", "Brihaspati"],
        "Venus": ["Venus", "Shukra"],
        "Saturn": ["Saturn", "Shani"],
        "Rahu": ["Rahu", "North Node"],
        "Ketu": ["Ketu", "South Node"]
    },
    "sign": {
        "Aries": ["Aries", "Mesha"],
        "Taurus": ["Taurus", "Vrishabha"],
        "Gemini": ["Gemini", "Mithuna"],
        "Cancer": ["Cancer", "Karka"],
        "Leo": ["Leo", "Simha"],
        "Virgo": ["Virgo", "Kanya"],
        "Libra": ["Libra", "Tula"],
        "Scorpio": ["Scorpio", "Vrishchika"],
        "Sagittarius": ["Sagittarius", "Dhanu"],
        "Capricorn": ["Capricorn", "Makara"],
        "Aquarius": ["Aquarius", "Kumbha"],
        "Pisces": ["Pisces", "Meena"]
    },
    "house_group": {
        "Kendra": ["Kendra", "Kendra Houses", "Angle"],
        "Trikona": ["Trikona", "Trikona Houses", "Trine"],
        "Upachaya": ["Upachaya", "Upachaya Houses"],
        "Dusthana": ["Dusthana", "Dusthana Houses"],
        "Maraka": ["Maraka", "Maraka Houses"],
        "Moksha": ["Moksha", "Moksha Houses"]
    },
    # --- ADDED NEW ENTITIES BELOW ---
    "nakshatra": {
        "Ashwini": ["Ashwini"], "Bharani": ["Bharani"], "Krittika": ["Krittika"],
        "Rohini": ["Rohini"], "Mrigashirsha": ["Mrigashirsha"], "Ardra": ["Ardra"],
        "Punarvasu": ["Punarvasu"], "Pushya": ["Pushya"], "Ashlesha": ["Ashlesha"],
        "Magha": ["Magha"], "Purva Phalguni": ["Purva Phalguni"], "Uttara Phalguni": ["Uttara Phalguni"],
        "Hasta": ["Hasta"], "Chitra": ["Chitra"], "Swati": ["Swati"], "Vishakha": ["Vishakha"],
        "Anuradha": ["Anuradha"], "Jyeshtha": ["Jyeshtha"], "Mula": ["Mula"],
        "Purva Ashadha": ["Purva Ashadha"], "Uttara Ashadha": ["Uttara Ashadha"], "Shravana": ["Shravana"],
        "Dhanishtha": ["Dhanishtha"], "Shatabhisha": ["Shatabhisha"], "Purva Bhadrapada": ["Purva Bhadrapada"],
        "Uttara Bhadrapada": ["Uttara Bhadrapada"], "Revati": ["Revati"]
    },
    "varga": {
        "Rasi": ["Rasi", "D1", "D-1"],
        "Hora": ["Hora", "D2", "D-2"],
        "Drekkana": ["Drekkana", "D3", "D-3"],
        "Chaturthamsa": ["Chaturthamsa", "D4", "D-4"],
        "Saptamsa": ["Saptamsa", "D7", "D-7"],
        "Navamsa": ["Navamsa", "D9", "D-9"],
        "Dasamsa": ["Dasamsa", "D10", "D-10"],
        "Dvadasamsa": ["Dvadasamsa", "D12", "D-12"],
        "Shodasamsa": ["Shodasamsa", "D16", "D-16"],
        "Vimsamsa": ["Vimsamsa", "D20", "D-20"],
        "Siddhamsa": ["Siddhamsa", "D24", "D-24"],
        "Trimsamsa": ["Trimsamsa", "D30", "D-30"],
        "Shashtyamsa": ["Shashtyamsa", "D60", "D-60"]
    },
    "special_point": {
        "Arudha Lagna": ["Arudha Lagna"],
        "Upapada Lagna": ["Upapada Lagna"],
        "Bhava Lagna": ["Bhava Lagna"],
        "Hora Lagna": ["Hora Lagna"],
        "Ghati Lagna": ["Ghati Lagna"]
    },
    "special_chart": {
        "Bhava Chalit": ["Bhava Chalit", "Chalit"],
        "Chandra Lagna": ["Chandra Lagna", "Moon Chart"],
        "Surya Lagna": ["Surya Lagna", "Sun Chart"]
    }
}