import pandas as pd
import pdfplumber
import google.generativeai as genai
import os
from dotenv import load_dotenv
import hashlib
from datetime import datetime
import time

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
# Configure the Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")
genai.configure(api_key=GEMINI_API_KEY)

# Define the generative model - UPDATED
model = genai.GenerativeModel('gemini-2.5-pro-latest')

def generate_fact_id(text):
    """Generates a unique ID for a fact based on its content."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

def extract_facts_from_text(page_text, book_title, theme):
    """Uses Gemini to extract astrological facts from a piece of text."""
    prompt = f"""
    You are an expert in Vedic Astrology. Your task is to act as a knowledge engineering assistant.
    Analyze the following text from the book "{book_title}" and extract every distinct astrological rule or interpretation you can find related to the theme of '{theme}'.

    For each rule you find, structure it in the following format, using '||' as a separator:
    Foundation_Point||Interpretation_Text

    Example:
    Sun in 10th House||The native will achieve high status and recognition in their career.
    Mars in 7th House||This placement can cause conflicts in marriage.

    If you find no rules on this page, output the single word "NONE".

    Here is the text to analyze:
    ---
    {page_text}
    ---
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        return "NONE"

def process_pdf(pdf_path, output_csv, book_title, theme):
    """Processes a PDF, extracts facts, and saves them to a CSV."""
    print(f"Starting processing for '{book_title}'...")
    
    # Check if output CSV exists, if not create it with headers
    if not os.path.exists(output_csv):
        df_header = pd.DataFrame(columns=["Fact_ID", "Theme", "Fact_Group", "Sub_Themes", "Foundation_Point", 
                                          "Interpretation_Text", "AI_Astro_Summary", "Source_Name", "Source_Type", 
                                          "Source_Reference", "Confidence_Score", "Status", "Conflict_Tag", 
                                          "Primary_Fact_ID", "Last_Updated", "Notes"])
        df_header.to_csv(output_csv, index=False)

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            print(f"  Processing page {i+1} of {len(pdf.pages)}...")
            page_text = page.extract_text()

            if page_text and len(page_text) > 50: # Process only if there's meaningful text
                extracted_data = extract_facts_from_text(page_text, book_title, theme)
                
                if "NONE" not in extracted_data:
                    new_facts = []
                    for line in extracted_data.strip().split('\n'):
                        if '||' in line:
                            parts = line.split('||')
                            foundation_point = parts[0].strip()
                            interpretation_text = parts[1].strip()
                            
                            fact_id_text = f"{foundation_point}{interpretation_text}"
                            
                            new_fact = {
                                "Fact_ID": generate_fact_id(fact_id_text),
                                "Theme": theme,
                                "Fact_Group": "", # To be filled in later
                                "Sub_Themes": "", # To be filled in later
                                "Foundation_Point": foundation_point,
                                "Interpretation_Text": interpretation_text,
                                "AI_Astro_Summary": "", # To be filled in later
                                "Source_Name": book_title,
                                "Source_Type": "BOOK",
                                "Source_Reference": f"Page {i+1}",
                                "Confidence_Score": 0.75, # Default for a book source
                                "Status": "RAW",
                                "Conflict_Tag": "",
                                "Primary_Fact_ID": "",
                                "Last_Updated": datetime.now().isoformat(),
                                "Notes": "Automated extraction."
                            }
                            new_facts.append(new_fact)
                    
                    if new_facts:
                        df_new = pd.DataFrame(new_facts)
                        df_new.to_csv(output_csv, mode='a', header=False, index=False)
                        print(f"    -> Found and saved {len(new_facts)} new facts.")
                
                # To respect API rate limits
                time.sleep(2) # A 2-second delay between API calls

    print(f"Processing complete. Data saved to '{output_csv}'.")

if __name__ == '__main__':
    # --- DEFINE YOUR JOB HERE ---
    PDF_FILE = "source_material/books/your_book_name.pdf"
    OUTPUT_CSV_FILE = "knowledge-base/career.csv"
    BOOK_TITLE = "The Full Title of Your Book"
    THEME = "career"
    
    process_pdf(PDF_FILE, OUTPUT_CSV_FILE, BOOK_TITLE, THEME)