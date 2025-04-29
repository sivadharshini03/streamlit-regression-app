import streamlit as st
import pandas as pd
from streamlit import switch_page

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Upload Data",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {
            display: none !important;
        }
        .css-18e3th9 { margin-left: 0 !important; }

        /* Subtitle styling */
        h3.subtitle {
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            color: #4e4e4e;
        }

        /* Next button container: simple margin, no fixed positioning */
        .next-button-container {
            margin-top: 20px;
        }
        .next-button-container button {
            background-color: #0E1117;
            color: white;
            padding: 0.6em 1.4em;
            font-size: 1rem;
            border-radius: 8px;
            border: none;
        }
    </style>
""", unsafe_allow_html=True)

# ── PAGE TITLE ─────────────────────────────────────────────────────────────────
st.markdown('<h1 style="text-align:center;">📂 Upload Your Dataset</h1>', unsafe_allow_html=True)

# ── PAGE LAYOUT ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown('<h3 class="subtitle">🗂️ Select File Type</h3>', unsafe_allow_html=True)
    file_type = st.radio("", ["-- Select File Type --", "CSV", "Excel", "JSON"], horizontal=True)
    file = None
    if file_type != "-- Select File Type --":
        exts = {"CSV": "csv", "Excel": "xlsx", "JSON": "json"}
        file = st.file_uploader("Drag and drop your file here", type=[exts[file_type]])
    else:
        st.info("ℹ️ Please select a valid file type to enable uploading.")

with col2:
    st.markdown('<h3 class="subtitle">📊 Dataset Preview</h3>', unsafe_allow_html=True)

    df = None
    if file is not None:
        try:
            df = (
                pd.read_csv(file) if file_type == "CSV"
                else pd.read_excel(file) if file_type == "Excel"
                else pd.read_json(file)
            )
            st.session_state.uploaded_data = df
            st.session_state.uploaded_filename = file.name
            st.success(f"✅ Uploaded: {file.name}")
        except Exception as e:
            st.error(f"⚠️ Error loading file: {e}")
    elif 'uploaded_data' in st.session_state:
        df = st.session_state.uploaded_data
        fn = st.session_state.get("uploaded_filename", "your data")
        st.success(f"✅ Previously uploaded: {fn}")

    # If we have a dataframe, show its shape and preview
    if df is not None:
        st.markdown(f"**🔢 Rows:** {df.shape[0]} &nbsp;&nbsp;&nbsp; **📈 Columns:** {df.shape[1]}")
        st.dataframe(df.head(), use_container_width=True)

        # ── NEXT BUTTON ────────────────────────────────────────────────────────────
        st.markdown('<div class="next-button-container">', unsafe_allow_html=True)
        if st.button("➡️ Next"):
            st.session_state.navigate_to_preprocessing = True
            switch_page("pages/2_🧹_Preprocessing.py")

        st.markdown('</div>', unsafe_allow_html=True)
