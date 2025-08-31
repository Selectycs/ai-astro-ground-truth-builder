import os
import pandas as pd
import json
from datetime import datetime

# --- CONFIGURATION ---
KNOWLEDGE_BASE_PATH = "knowledge-base"
DIST_PATH = "dist"  # Distribution folder for final output
BUILD_VERSION = "1.0.0" # You can increment this for each build

# --- GUARDRAIL DEFINITIONS ---
# Define the expected columns for the interpretations file
EXPECTED_COLUMNS = [
    "Fact_ID", "Theme", "Fact_Group", "Sub_Themes", "Foundation_Point", 
    "Interpretation_Text", "AI_Astro_Summary", "Source_Name", "Source_Type", 
    "Source_Reference", "Confidence_Score", "Status", "Conflict_Tag", 
    "Primary_Fact_ID", "Last_Updated", "Notes"
]
# Define valid enum values
VALID_STATUSES = ["RAW", "VERIFIED", "DEPRECATED", "CONFLICT"]
VALID_SOURCE_TYPES = ["BOOK", "AI_QUERY", "MANUAL"]


def validate_dataframe(df):
    """
    Runs a series of validation checks on the DataFrame.
    Returns True if all checks pass, False otherwise.
    """
    print("--- Starting Validation ---")
    is_valid = True

    # 1. Schema Check: Ensure all expected columns are present
    if not all(col in df.columns for col in EXPECTED_COLUMNS):
        print("Error: Schema mismatch. Some expected columns are missing.")
        is_valid = False

    # 2. Non-Empty Checks: Critical fields should not be empty
    critical_fields = ["Fact_ID", "Theme", "Foundation_Point", "Interpretation_Text"]
    for field in critical_fields:
        if df[field].isnull().any() or (df[field] == "").any():
            print(f"Error: Critical field '{field}' contains empty values.")
            is_valid = False

    # 3. Enum & Type Checks
    if not df['Status'].isin(VALID_STATUSES).all():
        print(f"Error: 'Status' column contains invalid values.")
        is_valid = False
    
    # 4. Range and Uniqueness Checks
    if not df['Confidence_Score'].between(0, 1).all():
        print("Error: 'Confidence_Score' is not between 0 and 1.")
        is_valid = False
    
    if not df['Fact_ID'].is_unique:
        print("Error: 'Fact_ID' column contains duplicate values.")
        is_valid = False
    
    print("--- Validation Complete ---")
    return is_valid

def consolidate_data(df):
    """
    Performs deduplication and conflict tagging.
    """
    print("--- Starting Consolidation ---")
    
    # Duplicate Detection
    # Find rows with the same astrological rule (Foundation_Point)
    duplicates = df[df.duplicated(subset=['Foundation_Point'], keep=False)]
    
    if not duplicates.empty:
        print(f"Found {len(duplicates)} rows that are part of duplicate groups.")
        # In a real scenario, you would add logic here to:
        # 1. Group these by 'Foundation_Point'.
        # 2. For each group, assign one 'Fact_ID' as the 'Primary_Fact_ID' for all others.
        # 3. This step is often semi-automated and requires review.
        # For now, we'll just print a message.
        print("Action: Manual review needed to assign Primary_Fact_ID for duplicate groups.")

    # Conflict Tagging (Placeholder)
    # Your logic here would group by 'Foundation_Point' and check for
    # significantly different 'Interpretation_Text' or other conflicting attributes.
    # If a conflict is found, you would set the 'Conflict_Tag' and update 'Status'.
    print("Action: Conflict tagging logic to be implemented based on expert rules.")
    
    print("--- Consolidation Complete ---")
    return df


def main():
    """
    Main function to run the validation, consolidation, and build process.
    """
    interpretations_file = os.path.join(KNOWLEDGE_BASE_PATH, "interpretations.csv")
    
    if not os.path.exists(interpretations_file):
        print(f"Error: '{interpretations_file}' not found. Please run the ingestion script first.")
        return

    # Load the entire thematic knowledge base
    print(f"Loading data from '{interpretations_file}'...")
    df = pd.read_csv(interpretations_file)

    # Run validations
    if not validate_dataframe(df):
        print("\nBuild failed due to validation errors. Please fix the data and retry.")
        return

    # Run consolidation steps
    df = consolidate_data(df)
    
    # This is where your team's manual review (Phase 3) would happen.
    # After review, the 'Status' column in 'interpretations.csv' would be updated.
    # For this script, we'll simulate that by assuming some facts are now 'VERIFIED'.
    print("\nSimulating manual review: Assuming some facts are now 'VERIFIED'.")
    
    # --- Final Build Step (Phase 4) ---
    print("--- Starting Final Build ---")
    
    # 1. Update Confidence Scores based on Status
    df.loc[df['Status'] == 'VERIFIED', 'Confidence_Score'] = 1.0
    df.loc[df['Status'] == 'DEPRECATED', 'Confidence_Score'] = 0.0

    # 2. Filter for only verified facts for the production build
    verified_df = df[df['Status'] == 'VERIFIED'].copy()
    
    if verified_df.empty:
        print("Warning: No facts with status 'VERIFIED' found. No production file will be generated.")
        return

    # 3. Create the distribution directory
    os.makedirs(DIST_PATH, exist_ok=True)
    
    # 4. Export the final, clean knowledge base
    final_kb_path = os.path.join(DIST_PATH, "knowledge_base.json")
    verified_df.to_json(final_kb_path, orient="records", indent=2)
    print(f"Successfully exported {len(verified_df)} verified facts to '{final_kb_path}'.")

    # 5. Semantic versioning
    version_info = {
        "build_version": BUILD_VERSION,
        "build_date": datetime.now().isoformat(),
        "total_facts_verified": len(verified_df)
    }
    version_path = os.path.join(DIST_PATH, "version.json")
    with open(version_path, "w") as f:
        json.dump(version_info, f, indent=2)
    print(f"Build version {BUILD_VERSION} information saved to '{version_path}'.")
    
    print("--- Build Process Complete ---")


if __name__ == "__main__":
    main()