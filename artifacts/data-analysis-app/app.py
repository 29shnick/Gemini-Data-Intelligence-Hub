"""
Gemini Data Intelligence Hub - Dashboard Application
Handles Streamlit UI and secure Gemini AI integration.
"""
import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import io
import pdfplumber

# --- CORE FUNCTIONS ---

@st.cache_resource
def get_gemini_model():
    """Initializes the Gemini model using secure environment variables."""
    auth_val = os.environ.get("GEMINI_API_KEY")
    if not auth_val:
        return None
    genai.configure(api_key=auth_val)
    return genai.GenerativeModel("gemini-1.5-flash")

def extract_pdf_text(file_bytes):
    """Extracts text, page count, and word count from PDF bytes."""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages_text = [page.extract_text() or "" for page in pdf.pages]
            full_text = "\n\n".join(pages_text).strip()
            p_count = len(pdf.pages)
            w_count = len(full_text.split())
            return full_text, p_count, w_count
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return "", 0, 0

def build_data_context(df: pd.DataFrame) -> str:
    """Converts a dataframe into a text summary for the AI context."""
    buf = io.StringIO()
    buf.write(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n")
    buf.write("Columns and Data Types:\n")
    for col, dtype in df.dtypes.items():
        buf.write(f"- {col}: {dtype}\n")
    buf.write("\nStatistical Summary (numeric columns):\n")
    buf.write(df.describe().to_string())
    return buf.getvalue()

# --- UI LOGIC ---

def render_pdf_view(model, uploaded_file):
    """Handles the rendering and analysis of PDF documents."""
    state = st.session_state
    
    if state.get("pdf_text") is None:
        with st.spinner("Reading PDF..."):
            raw = uploaded_file.read()
            text, pages, words = extract_pdf_text(raw)
            state.pdf_text = text
            state.pdf_pages = pages
            state.pdf_words = words

    st.write(f"**Pages:** {state.pdf_pages} | **Words:** {state.pdf_words}")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📊 Analysis", "💬 Chat with AI"])
    
    with tab1:
        if state.get("pdf_analysis") is None:
            with st.spinner("Analyzing document with Gemini..."):
                prompt = f"Analyze this document text and provide a summary: {state.pdf_text[:10000]}"
                response = model.generate_content(prompt)
                state.pdf_analysis = response.text
        st.markdown(state.pdf_analysis)

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Gemini Data Intelligence Hub")
    st.title("🚀 Gemini Data Intelligence Hub")
    
    model = get_gemini_model()
    if not model:
        st.error("Gemini API Key not found. Please check your .env file.")
        return

    uploaded_file = st.file_uploader("Upload a PDF or CSV file", type=["pdf", "csv"])
    
    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            render_pdf_view(model, uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            context = build_data_context(df)
            st.info("Data loaded successfully.")

if __name__ == "__main__":
    main()
    