import streamlit as st
import pandas as pd
import os
import json
import psycopg2
from dotenv import load_dotenv
from v2_kb_utils import FINAL_DATA_PATH # Assuming this is in your utils

# --- Load environment variables from .env file ---
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Astro KB Viewer")

# --- DATA LOADING ---
@st.cache_data
def load_data_from_csv(file_path):
    """Loads data from the final CSV file."""
    if not os.path.exists(file_path):
        st.error(f"Error: CSV file not found at {file_path}")
        return pd.DataFrame()
    return pd.read_csv(file_path, dtype=str).fillna("")

# Updated: This function now uses a single DATABASE_URL environment variable
@st.cache_data
def load_data_from_postgres():
    """Loads data directly from the PostgreSQL database using a connection URL."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        st.error("Error: DATABASE_URL not found in your .env file.")
        return pd.DataFrame()
        
    try:
        conn = psycopg2.connect(db_url)
        # Assumes your final, consolidated table is named 'interpretations'
        query = "SELECT * FROM interpretations ORDER BY fact_id;"
        df = pd.read_sql(query, conn)
        conn.close()
        # Parse the JSON column for easier filtering
        df['trigger_obj'] = df['astrological_trigger_json'].apply(json.loads)
        return df
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return pd.DataFrame()

# --- MAIN APP ---
st.title("AI Astro Knowledge Base Viewer")

# --- DATA SOURCE SELECTION ---
source_option = st.radio(
    "Select Data Source",
    ('PostgreSQL Database', 'CSV File'),
    horizontal=True
)

df = pd.DataFrame()
if source_option == 'PostgreSQL Database':
    df = load_data_from_postgres()
else:
    final_csv_path = os.path.join(FINAL_DATA_PATH, "interpretations.production.csv")
    df = load_data_from_csv(final_csv_path)
    if not df.empty:
        df['trigger_obj'] = df['astrological_trigger_json'].apply(json.loads)


if not df.empty:
    # Extract filter options from the dataframe
    def get_unique_values(key):
        return sorted(df['trigger_obj'].apply(lambda x: x.get(key)).dropna().unique())

    planets = get_unique_values('planet_name')
    signs = get_unique_values('sign')
    houses = sorted(pd.to_numeric(get_unique_values('house'), errors='coerce').dropna().astype(int).unique())
    nakshatras = get_unique_values('nakshatra')
    vargas = get_unique_values('varga')
    schema_types = get_unique_values('type')
    
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filters")
    selected_schema_type = st.sidebar.selectbox("Schema Type", options=["All"] + schema_types)
    selected_planet = st.sidebar.selectbox("Planet", options=["All"] + planets)
    selected_sign = st.sidebar.selectbox("Sign", options=["All"] + signs)
    selected_house = st.sidebar.selectbox("House", options=["All"] + houses)
    selected_nakshatra = st.sidebar.selectbox("Nakshatra", options=["All"] + nakshatras)
    selected_varga = st.sidebar.selectbox("D-Chart (Varga)", options=["All"] + vargas)
    search_text = st.sidebar.text_input("Search in Interpretation Text")

    # --- FILTERING LOGIC ---
    filtered_df = df.copy()

    # Apply filters
    if selected_schema_type != "All":
        filtered_df = filtered_df[filtered_df['trigger_obj'].apply(lambda x: x.get('type') == selected_schema_type)]
    if selected_planet != "All":
        filtered_df = filtered_df[filtered_df['trigger_obj'].apply(lambda x: x.get('planet_name') == selected_planet)]
    if selected_sign != "All":
        filtered_df = filtered_df[filtered_df['trigger_obj'].apply(lambda x: x.get('sign') == selected_sign)]
    if selected_house != "All":
        filtered_df = filtered_df[filtered_df['trigger_obj'].apply(lambda x: x.get('house') == selected_house)]
    if selected_nakshatra != "All":
        filtered_df = filtered_df[filtered_df['trigger_obj'].apply(lambda x: x.get('nakshatra') == selected_nakshatra)]
    if selected_varga != "All":
        filtered_df = filtered_df[filtered_df['trigger_obj'].apply(lambda x: x.get('varga') == selected_varga)]
    if search_text:
        filtered_df = filtered_df[filtered_df['interpretation_text'].str.contains(search_text, case=False, na=False)]

    # --- DISPLAY RESULTS ---
    st.header("Filtered Results")
    st.metric(label="Total Rows Found", value=len(filtered_df))
    
    display_cols = ['fact_id', 'theme', 'sub_theme', 'interpretation_text', 'astrological_trigger_json']
    st.dataframe(filtered_df[[col for col in display_cols if col in filtered_df.columns]], use_container_width=True)

else:
    st.warning("No data loaded. Please check your data source configuration and ensure the selected source contains data.")