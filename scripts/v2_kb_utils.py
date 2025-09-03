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
# This structure is the target output of the entire KB build pipeline.
V2_INTERPRETATIONS_SCHEMA = [
    "fact_id",                   # Unique identifier (SHA256 hash).
    "astrological_trigger_json", # The structured, machine-readable rule (CORE of V2).
    "interpretation_summary_raw",# A brief, human-readable summary of the trigger rule.
    "interpretation_summary_ai", # AI-generated summary for semantic search embedding.
    "interpretation_text",       # The full, original interpretation text from the source.
    "theme",                     # The primary life area this interpretation relates to (from TAXONOMY).
    "sub_theme",                 # A more specific category within the main Theme.
    "interpretation_group",      # A hash of the trigger JSON used to group identical rules for consolidation.
    "source_name",               # Name of the source book/text (e.g., "Brihat Parashara Hora Shastra").
    "source_type",               # Type of source (e.g., "Classical Text", "Modern Commentary").
    "source_reference",          # Specific location in the source (e.g., "Chapter 4, Verse 12").
    "status",                    # The current stage in the pipeline (e.g., RAW, VALIDATED, CONSOLIDATED, CLEANED).
    "confidence_score",          # The AI's confidence in the accuracy of the extraction (0.0 to 1.0).
    "primary_fact_id",           # For consolidated facts, the ID of the chosen primary interpretation.
    "fallback_tags",             # Optional secondary tags for low-confidence classifications.
    "embedding_vector",          # The stored pgvector embedding for semantic search.
    "schema_version",          # <-- ADDED THIS
    "conflict_status",         # <-- ADDED THIS
    "last_updated",              # ISO timestamp of the last modification.
    "notes"                      # Any notes from the automated or manual review process.
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
# Preserved exactly from the original file to ensure full coverage.
TAXONOMY = {
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