import os
import pandas as pd
from datetime import datetime
import hashlib

# --- PATHS ---
KNOWLEDGE_BASE_PATH = "knowledge-base"
CORE_CONCEPTS_PATH = os.path.join(KNOWLEDGE_BASE_PATH, "core_concepts")

# --- SCHEMAS ---
# Schema for the single, consolidated interpretations file
INTERPRETATIONS_SCHEMA = [
    "Fact_ID", "Theme", "Fact_Group", "Sub_Themes", "Foundation_Point", 
    "Interpretation_Text", "AI_Astro_Summary", "Chart_Refs_JSON",
    "Source_Name", "Source_Type", "Source_Reference", "Confidence_Score", 
    "Status", "Conflict_Tag", "Primary_Fact_ID", "Last_Updated", "Notes"
]

# Schemas for the separate core concept files
CORE_CONCEPT_SCHEMAS = {
    "planets": [
        "Fact_ID", "Concept_Group", "Concept_Name", "Sanskrit_Name", "Keywords", 
        "General_Description", "Dignities_JSON", "Karaka_JSON", "Attributes_JSON",
        "Source_Name", "Source_Type", "Source_Reference", "Last_Updated"
    ],
    "signs": [
        "Fact_ID", "Concept_Group", "Concept_Name", "Sanskrit_Name", "Keywords",
        "Description", "Ruling_Planet", "Element", "Modality", "Gender", 
        "Body_Part", "Symbol",
        "Source_Name", "Source_Type", "Source_Reference", "Last_Updated"
    ],
    "houses": [
        "Fact_ID", "Concept_Group", "Concept_Name", "Sanskrit_Name", "Keywords", 
        "Description", "Karaka_Planets_JSON", "Related_Houses_JSON",
        "Source_Name", "Source_Type", "Source_Reference", "Last_Updated"
    ],
    "nakshatras": ["Fact_ID", "Concept_Group", "Concept_Name", "Ruling_Planet", "Symbol", "Deity", "Description", "Source_Name", "Source_Type", "Source_Reference", "Last_Updated"],
    "yogas_doshas": ["Fact_ID", "Concept_Group", "Concept_Name", "Type", "Definition", "Source_Name", "Source_Type", "Source_Reference", "Last_Updated"],
    "timing_systems": ["Fact_ID", "Concept_Group", "Concept_Name", "System_Type", "Definition", "Source_Name", "Source_Type", "Source_Reference", "Last_Updated"],
    "astrological_terms": ["Fact_ID", "Concept_Group", "Concept_Name", "Definition", "Source_Name", "Source_Type", "Source_Reference", "Last_Updated"]
}

# --- THEMATIC TAXONOMY ---
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
            'inheritance_unearned_income', 'other_income_earnings'
        ],
        'Assets & Property': [
            'property_real_estate', 'movable_assets', 'savings_investments',
            'debts_liabilities', 'other_assets_property'
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
            'spiritual_learning', 'philosophical_understanding', 'practical_skills',
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
            'home_comforts', 'property_acquisition', 'relocation_residence_changes', 'other_residence_property'
        ],
        'Vehicles & Movables': [
            'vehicle_ownership', 'travel_comforts', 'luxury_possessions', 'other_vehicles_movables'
        ],
        'Domestic Environment': [
            'domestic_harmony', 'parental_care', 'family_security', 'other_domestic_environment'
        ],
        'Other': ['other_home_related']
    },
    'health_risks_longevity': {
        'Physical Health': [
            'general_health', 'chronic_diseases', 'accidents_injuries', 'immunity_strength', 'other_physical_health'
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
            'religious_alignment', 'spiritual_growth', 'philosophical_beliefs',
            'charitable_service', 'other_dharma_spirituality'
        ],
        'Other': ['other_travel_spirituality_related']
    },
    'yogas': {
        'Strength & Success Yogas': ['raja_yogas', 'dhana_yogas', 'gajakesari_yoga', 'other_strength_success_yogas'],
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

def generate_fact_id(text):
    """Generates a unique ID for a fact based on its content."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

# --- SAVING FUNCTIONS ---

def save_interpretations_to_kb(interpretations_list, source_name, source_type, source_reference_prefix=""):
    """Saves a list of thematic interpretations to a single interpretations.csv file."""
    if not interpretations_list:
        return

    new_facts = []
    for fact_details in interpretations_list:
        # Corrected Fact_ID logic to ensure uniqueness for semantic duplicates
        fact_id_text = f"{fact_details.get('theme', '')}{fact_details.get('sub_themes', '')}{fact_details.get('foundation_point', '')}{fact_details.get('interpretation_text', '')}"
        
        new_fact = {
            "Fact_ID": generate_fact_id(fact_id_text),
            "Theme": fact_details.get('theme', ''),
            "Fact_Group": fact_details.get('fact_group', ''),
            "Sub_Themes": fact_details.get('sub_themes', ''),
            "Foundation_Point": fact_details.get('foundation_point', ''),
            "Interpretation_Text": fact_details.get('interpretation_text', ''),
            "AI_Astro_Summary": fact_details.get('ai_summary', ''),
            "Chart_Refs_JSON": fact_details.get('chart_refs_json', '{}'),
            "Source_Name": source_name,
            "Source_Type": source_type,
            "Source_Reference": f"{source_reference_prefix}{fact_details.get('reference', '')}",
            "Confidence_Score": fact_details.get('confidence', 0.5),
            "Status": "RAW",
            "Conflict_Tag": "",
            "Primary_Fact_ID": "",
            "Last_Updated": datetime.now().isoformat(),
            "Notes": f"Automated processing from {source_type}."
        }
        new_facts.append(new_fact)

    output_csv = os.path.join(KNOWLEDGE_BASE_PATH, "interpretations.csv")
    os.makedirs(KNOWLEDGE_BASE_PATH, exist_ok=True)
    
    file_exists = os.path.exists(output_csv)
    df_new = pd.DataFrame(new_facts, columns=INTERPRETATIONS_SCHEMA)
    df_new.to_csv(output_csv, mode='a', header=not file_exists, index=False)
    
    print(f"    -> Saved {len(df_new)} new interpretations to 'interpretations.csv'.")


def save_core_concepts_to_kb(concepts_by_group, source_name, source_type, source_reference_prefix=""):
    """Saves a dictionary of core concepts using predefined schemas."""
    if not concepts_by_group:
        return
    
    os.makedirs(CORE_CONCEPTS_PATH, exist_ok=True)

    for group, concepts in concepts_by_group.items():
        if group not in CORE_CONCEPT_SCHEMAS:
            print(f"Warning: No schema found for core concept group '{group}'. Skipping.")
            continue

        output_csv = os.path.join(CORE_CONCEPTS_PATH, f"{group}.csv")
        schema = CORE_CONCEPT_SCHEMAS[group]
        
        reformatted_concepts = []
        for concept_details in concepts:
            new_concept = {key: "" for key in schema}
            new_concept.update({
                "Concept_Group": group,
                "Source_Name": source_name,
                "Source_Type": source_type,
                "Source_Reference": f"{source_reference_prefix}{concept_details.get('reference', '')}",
                "Last_Updated": datetime.now().isoformat(),
            })
            
            # Populate with data extracted from the AI JSON
            for key, value in concept_details.items():
                if key in new_concept:
                    new_concept[key] = value
            
            fact_id_text = f"{new_concept.get('Concept_Name', '')}{new_concept.get('Description', new_concept.get('Definition', ''))}"
            new_concept["Fact_ID"] = generate_fact_id(fact_id_text)

            reformatted_concepts.append(new_concept)

        df_new = pd.DataFrame(reformatted_concepts, columns=schema)
        
        file_exists = os.path.exists(output_csv)
        df_new.to_csv(output_csv, mode='a', header=not file_exists, index=False)
        
        print(f"    -> Saved {len(df_new)} new core concepts to '{group}.csv' using schema.")