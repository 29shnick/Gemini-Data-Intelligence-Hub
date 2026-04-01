import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import io
import pdfplumber

st.set_page_config(
    page_title="Data Analysis AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }

    [data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #2a2f3e;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown p { color: #c8cdd8; }

    h1, h2, h3 { color: #ffffff; font-weight: 700; letter-spacing: -0.5px; }
    h1 { font-size: 1.9rem; }
    h2 { font-size: 1.4rem; }
    h3 { font-size: 1.1rem; }

    [data-testid="stMetric"] {
        background: #1a2035;
        border: 1px solid #2a3050;
        border-radius: 10px;
        padding: 16px 20px;
    }
    [data-testid="stMetricLabel"] { color: #8892a4; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.6px; }
    [data-testid="stMetricValue"] { color: #4f9cf9; font-size: 1.7rem; font-weight: 700; }

    [data-testid="stDataFrame"] { border: 1px solid #2a3050; border-radius: 10px; overflow: hidden; }

    [data-testid="stFileUploader"] {
        background: #141929;
        border: 2px dashed #2e3a5c;
        border-radius: 12px;
        padding: 20px;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploader"]:hover { border-color: #4f9cf9; }

    [data-testid="stChatMessage"] { border-radius: 12px; margin-bottom: 8px; padding: 6px 10px; }
    [data-testid="stChatMessage"][data-testid*="user"] { background: #1e2d50; }
    [data-testid="stChatMessage"][data-testid*="assistant"] { background: #171e2e; }

    [data-testid="stChatInput"] textarea {
        background: #1a2035 !important;
        color: #e0e0e0 !important;
        border: 1px solid #2e3a5c !important;
        border-radius: 10px !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #4f9cf9 !important;
        box-shadow: 0 0 0 2px rgba(79,156,249,0.15) !important;
    }

    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background: #141929; border-radius: 10px; gap: 4px; padding: 4px;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        background: transparent; color: #8892a4; border-radius: 8px;
        padding: 6px 16px; font-size: 0.9rem;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        background: #1e2d50 !important; color: #4f9cf9 !important; font-weight: 600;
    }

    [data-testid="stExpander"] { background: #141929; border: 1px solid #2a3050; border-radius: 10px; }
    hr { border-color: #2a2f3e; }
    [data-testid="stAlert"] { border-radius: 10px; border-left-width: 4px; }

    code {
        background: #1a2035; color: #79d7f7;
        border-radius: 4px; padding: 1px 5px; font-size: 0.85em;
    }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0e1117; }
    ::-webkit-scrollbar-thumb { background: #2e3a5c; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #4f9cf9; }

    .dtype-pill {
        display: inline-block; background: #1e2d50; color: #79b8f9;
        border-radius: 20px; padding: 2px 10px;
        font-size: 0.78rem; font-family: monospace; margin: 2px 3px;
    }

    .card {
        background: #141929; border: 1px solid #2a3050;
        border-radius: 12px; padding: 20px 24px; margin-bottom: 16px;
    }

    .analysis-box {
        background: #141929; border: 1px solid #2a3050;
        border-radius: 12px; padding: 24px 28px; line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)


MAX_PDF_CHARS = 120_000


@st.cache_resource
def get_gemini_model():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


def extract_pdf_text(file_bytes: bytes) -> tuple[str, int, int]:
    """Return (full_text, page_count, word_count)."""
    pages_text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
    full_text = "\n\n".join(pages_text).strip()
    word_count = len(full_text.split())
    return full_text, page_count, word_count


def build_data_context(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    buf.write(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n")
    buf.write("Columns and Data Types:\n")
    for col, dtype in df.dtypes.items():
        buf.write(f"  - {col}: {dtype}\n")
    buf.write("\nStatistical Summary (numeric columns):\n")
    buf.write(df.describe().to_string())
    non_numeric = df.select_dtypes(exclude=["number"])
    if not non_numeric.empty:
        buf.write("\n\nCategorical / Text Columns Summary:\n")
        for col in non_numeric.columns:
            n_unique = df[col].nunique()
            top_vals = df[col].value_counts().head(5).to_dict()
            buf.write(f"\n  {col} ({n_unique} unique values):\n")
            for val, count in top_vals.items():
                buf.write(f"    '{val}': {count} occurrences\n")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        buf.write("\n\nMissing Values:\n")
        for col, count in missing.items():
            pct = (count / len(df)) * 100
            buf.write(f"  - {col}: {count} missing ({pct:.1f}%)\n")
    else:
        buf.write("\n\nMissing Values: None\n")
    buf.write("\n\nFirst 5 rows sample:\n")
    buf.write(df.head(5).to_string())
    return buf.getvalue()


def ask_gemini_data(model, question: str, data_context: str, chat_history: list) -> str:
    system_prompt = f"""You are a skilled data analyst assistant. The user has uploaded a spreadsheet or CSV dataset.
Here is a full summary of the dataset:

--- DATASET CONTEXT ---
{data_context}
--- END CONTEXT ---

Answer the user's questions clearly and concisely based on this data.
If you perform calculations, show your reasoning. Use markdown formatting.
If a question cannot be answered from the data provided, say so honestly."""

    history_text = ""
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    full_prompt = f"{system_prompt}\n\nConversation so far:\n{history_text}\nUser: {question}\nAssistant:"
    return model.generate_content(full_prompt).text


def generate_pdf_analysis(model, pdf_text: str, file_name: str) -> str:
    truncated = pdf_text[:MAX_PDF_CHARS]
    was_truncated = len(pdf_text) > MAX_PDF_CHARS

    prompt = f"""You are an expert document analyst. The user has uploaded a PDF document called "{file_name}".
{"Note: The document is very long so only the first portion is shown below." if was_truncated else ""}

--- DOCUMENT CONTENT ---
{truncated}
--- END DOCUMENT ---

Please provide a comprehensive analysis of this document. Structure your response with these sections:

## 📋 Summary
A concise overview of what this document is about (2-4 sentences).

## 🔑 Key Points
The most important facts, findings, or takeaways as a bullet list.

## 📊 Data & Figures
Any notable numbers, statistics, dates, or quantitative information mentioned.

## 🧠 Insights & Observations
Deeper analysis — patterns, implications, notable aspects, or anything worth highlighting.

## ❓ Suggested Questions
3-5 questions a reader might want to ask about this document.

Use clear markdown formatting throughout."""

    return model.generate_content(prompt).text


def ask_gemini_pdf(model, question: str, pdf_text: str, file_name: str, chat_history: list) -> str:
    truncated = pdf_text[:MAX_PDF_CHARS]
    was_truncated = len(pdf_text) > MAX_PDF_CHARS

    system_prompt = f"""You are an expert document analyst. The user has uploaded a PDF called "{file_name}".
{"Note: Only the first portion of the document is available due to length." if was_truncated else ""}

--- DOCUMENT CONTENT ---
{truncated}
--- END DOCUMENT ---

Answer the user's questions based on the document above.
Be specific and cite relevant parts of the document when useful.
Use markdown formatting. If something cannot be determined from the document, say so clearly."""

    history_text = ""
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    full_prompt = f"{system_prompt}\n\nConversation so far:\n{history_text}\nUser: {question}\nAssistant:"
    return model.generate_content(full_prompt).text


def render_pdf_sidebar(file_name: str, page_count: int, word_count: int):
    with st.sidebar:
        st.markdown("## 📄 Document Info")
        st.markdown(f"**File:** `{file_name}`")
        st.markdown(f"**Pages:** `{page_count}`")
        st.markdown(f"**Words:** `{word_count:,}`")
        st.markdown("---")
        st.markdown("**💡 Tips**")
        st.markdown("""
- Ask for a summary of any section
- Request key dates, names, or figures
- Ask to compare or contrast ideas
- Ask for clarification on any point
""")


def render_data_sidebar(df):
    with st.sidebar:
        st.markdown("## 📁 Dataset Info")
        st.markdown(f"**Rows:** `{df.shape[0]:,}`")
        st.markdown(f"**Columns:** `{df.shape[1]}`")
        st.markdown("---")
        st.markdown("**Column Types**")
        for col, dtype in df.dtypes.items():
            st.markdown(f'<span class="dtype-pill">{col}: {dtype}</span>', unsafe_allow_html=True)
        st.markdown("---")
        missing_total = df.isnull().sum().sum()
        if missing_total > 0:
            st.warning(f"⚠️ {missing_total:,} missing values found")
        else:
            st.success("✅ No missing values")
        st.markdown("---")
        st.markdown("**💡 Tips**")
        st.markdown("""
- Ask about trends, averages, or distributions
- Request correlations between columns
- Ask for insights or anomalies
- Request data quality assessments
""")


def render_pdf_view(model, uploaded_file):
    state = st.session_state

    if state.pdf_text is None:
        with st.spinner("Reading PDF…"):
            try:
                raw = uploaded_file.read()
                text, pages, words = extract_pdf_text(raw)
                state.pdf_text = text
                state.pdf_pages = pages
                state.pdf_words = words
            except Exception as e:
                st.error(f"Could not read PDF: {e}")
                return

        if not state.pdf_text.strip():
            st.error("No readable text found in this PDF. It may be a scanned image — please use a text-based PDF.")
            return

    render_pdf_sidebar(uploaded_file.name, state.pdf_pages, state.pdf_words)
    st.markdown("---")

    tab1, tab2 = st.tabs(["📄 Analysis", "💬 Chat with AI"])

    with tab1:
        if state.pdf_analysis is None:
            with st.spinner("Analysing document with Gemini…"):
                try:
                    state.pdf_analysis = generate_pdf_analysis(model, state.pdf_text, uploaded_file.name)
                except Exception as e:
                    st.error(f"❌ Gemini error: {e}")
                    return
        st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
        st.markdown(state.pdf_analysis)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown("### 💬 Ask Questions About This Document")
        st.caption("Gemini will answer based on the full content of your PDF.")

        if not state.messages:
            st.info("👋 Ask me anything about this document! For example: *\"What are the main conclusions?\"* or *\"List all the dates mentioned.\"*")
        else:
            for msg in state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a question about this document…"):
            state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        answer = ask_gemini_pdf(
                            model, prompt, state.pdf_text,
                            uploaded_file.name, state.messages[:-1],
                        )
                        st.markdown(answer)
                        state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        err = f"❌ Gemini error: {e}"
                        st.error(err)
                        state.messages.append({"role": "assistant", "content": err})


def render_data_view(model, uploaded_file, df):
    render_data_sidebar(df)
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📋 Overview", "🔍 Data Preview", "💬 Chat with AI"])

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", f"{df.shape[1]}")
        col3.metric("Numeric Cols", f"{len(df.select_dtypes(include='number').columns)}")
        col4.metric("Missing Values", f"{df.isnull().sum().sum():,}")

        st.markdown("### 📈 Statistical Summary")
        numeric_df = df.select_dtypes(include="number")
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe().round(3), use_container_width=True)
        else:
            st.info("No numeric columns found in this dataset.")

        cat_df = df.select_dtypes(exclude="number")
        if not cat_df.empty:
            st.markdown("### 🏷️ Categorical Summary")
            for col in cat_df.columns[:6]:
                with st.expander(f"📌 {col}  —  {df[col].nunique()} unique values"):
                    counts = df[col].value_counts().head(10).reset_index()
                    counts.columns = [col, "Count"]
                    st.dataframe(counts, use_container_width=True, hide_index=True)

        missing_series = df.isnull().sum()
        missing_cols = missing_series[missing_series > 0]
        if not missing_cols.empty:
            st.markdown("### ⚠️ Missing Values")
            miss_df = pd.DataFrame({
                "Column": missing_cols.index,
                "Missing Count": missing_cols.values,
                "Missing %": (missing_cols.values / len(df) * 100).round(2),
            })
            st.dataframe(miss_df, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### 🗂️ Raw Data")
        n_rows = st.slider("Rows to display", min_value=5, max_value=min(500, len(df)), value=20, step=5)
        st.dataframe(df.head(n_rows), use_container_width=True)
        col_a, col_b = st.columns(2)
        with col_a:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_bytes,
                file_name=f"processed_{uploaded_file.name}",
                mime="text/csv",
            )
        with col_b:
            st.markdown(f"**File:** `{uploaded_file.name}`  |  **Size:** `{uploaded_file.size / 1024:.1f} KB`")

    with tab3:
        st.markdown("### 💬 Ask Questions About Your Data")
        st.caption("Gemini will answer based on your dataset's structure and statistics.")

        if not st.session_state.messages:
            st.info("👋 Ask me anything about your dataset! For example: *\"What are the main trends?\"* or *\"Which column has the most missing values?\"*")
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a question about your data…"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        answer = ask_gemini_data(
                            model, prompt,
                            st.session_state.data_context,
                            st.session_state.messages[:-1],
                        )
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        err = f"❌ Gemini error: {e}"
                        st.error(err)
                        st.session_state.messages.append({"role": "assistant", "content": err})


def main():
    st.markdown("# 📊 Data Analysis AI")
    st.markdown("Upload a CSV, XLS, XLSX, or PDF file and chat with your data using Google Gemini.")

    model = get_gemini_model()
    if model is None:
        st.error("❌ GEMINI_API_KEY is not configured. Please add it to your environment secrets.")
        return

    for key, default in [
        ("messages", []),
        ("df", None),
        ("data_context", None),
        ("file_name", None),
        ("pdf_text", None),
        ("pdf_analysis", None),
        ("pdf_pages", 0),
        ("pdf_words", 0),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    uploaded_file = st.file_uploader(
        "Drop your file here",
        type=["csv", "xls", "xlsx", "pdf"],
        help="Supports CSV, XLS, XLSX, and PDF files",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        if st.session_state.file_name != uploaded_file.name:
            for key in ["messages", "df", "data_context", "pdf_text", "pdf_analysis", "pdf_pages", "pdf_words"]:
                st.session_state[key] = [] if key == "messages" else (0 if key in ("pdf_pages", "pdf_words") else None)
            st.session_state.file_name = uploaded_file.name
            for stale in ["sheet_name", "pdf_table_idx"]:
                st.session_state.pop(stale, None)

        file_ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
        is_excel = file_ext in ("xls", "xlsx")
        is_pdf = file_ext == "pdf"

        if is_pdf:
            render_pdf_view(model, uploaded_file)
            return

        if st.session_state.df is None:
            if is_excel:
                try:
                    xl = pd.ExcelFile(uploaded_file)
                    sheet_names = xl.sheet_names
                except Exception as e:
                    st.error(f"Could not open Excel file: {e}")
                    return

                if len(sheet_names) > 1:
                    selected_sheet = st.selectbox(
                        "This workbook has multiple sheets — choose one to analyse:",
                        sheet_names,
                        key="sheet_name",
                    )
                    if st.button("Load sheet"):
                        with st.spinner("Loading and analysing your dataset…"):
                            try:
                                df = xl.parse(selected_sheet)
                                st.session_state.df = df
                                st.session_state.data_context = build_data_context(df)
                            except Exception as e:
                                st.error(f"Could not parse sheet: {e}")
                                return
                    else:
                        return
                else:
                    with st.spinner("Loading and analysing your dataset…"):
                        try:
                            df = xl.parse(sheet_names[0])
                            st.session_state.df = df
                            st.session_state.data_context = build_data_context(df)
                        except Exception as e:
                            st.error(f"Could not parse Excel file: {e}")
                            return
            else:
                with st.spinner("Loading and analysing your dataset…"):
                    df = None
                    parse_note = None
                    raw = uploaded_file.read()
                    for sep in [None, ",", ";", "\t", "|"]:
                        for bad in ["warn", "skip"]:
                            try:
                                df = pd.read_csv(
                                    io.BytesIO(raw),
                                    sep=sep,
                                    on_bad_lines=bad,
                                    engine="python" if sep is None else "c",
                                )
                                if df.shape[1] < 2 and sep is None:
                                    continue
                                if bad == "skip":
                                    parse_note = "⚠️ Some rows with unexpected formatting were skipped during loading."
                                break
                            except Exception:
                                continue
                        if df is not None and not df.empty:
                            break
                    if df is None or df.empty:
                        st.error("Could not parse this file. Please check that it is a valid CSV.")
                        return
                    if parse_note:
                        st.warning(parse_note)
                    st.session_state.df = df
                    st.session_state.data_context = build_data_context(df)

        render_data_view(model, uploaded_file, st.session_state.df)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 60px 40px; background:#141929; border: 2px dashed #2e3a5c; border-radius:16px; margin-top:20px;">
            <div style="font-size:3.5rem; margin-bottom:16px;">📂</div>
            <h2 style="color:#ffffff; margin-bottom:8px;">Upload a file to get started</h2>
            <p style="color:#8892a4; font-size:1rem; max-width:460px; margin:0 auto;">
                Drop a CSV, XLS, XLSX, or PDF file above. You'll get an instant summary and a chat interface
                powered by Google Gemini to explore your data conversationally.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("""
            <div class="card">
                <h3 style="color:#4f9cf9;">📊 Smart Summary</h3>
                <p style="color:#8892a4; font-size:0.9rem;">Instant statistics, data types, missing values, and categorical distributions.</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="card">
                <h3 style="color:#4f9cf9;">📄 PDF Analysis</h3>
                <p style="color:#8892a4; font-size:0.9rem;">Upload any PDF — Gemini reads and analyses the full content automatically.</p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="card">
                <h3 style="color:#4f9cf9;">🤖 AI Chat</h3>
                <p style="color:#8892a4; font-size:0.9rem;">Ask natural language questions and get intelligent answers powered by Gemini.</p>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown("""
            <div class="card">
                <h3 style="color:#4f9cf9;">🔍 Data Explorer</h3>
                <p style="color:#8892a4; font-size:0.9rem;">Browse raw data, control row counts, and download the processed file.</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
