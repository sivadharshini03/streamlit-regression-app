import streamlit as st 
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, LabelEncoder,
    StandardScaler, MinMaxScaler, RobustScaler, Normalizer
)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from scipy import stats

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Preprocessing",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── HIDE SIDEBAR ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
      [data-testid="stSidebar"] { display: none; }
      .css-18e3th9 { margin-left: 0px; }
      [data-testid="stToolbar"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown(""" 
<style>
    .page-title {
        font-size: 40px;
        font-weight: bold;
        color: #4F8BF9;
        margin-bottom: 10px;
    }
    .info-text {
        font-size: 18px;
        color: #555;
        margin-bottom: 25px;
    }
    .section-title {
        font-size: 28px;
        color: #FF6F61;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .stButton>button {
        font-size: 18px;
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        border: none;
    }
    .next-button button {
        font-size: 18px;
        background-color: #007BFF;
        color: white;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        margin-top: 20px;
        border: none;
    }
    .next-button-container {
        position: fixed;
        bottom: 10px;
        right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ── TITLE ────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">🧹 Clean & Preprocess Your Data</div>', unsafe_allow_html=True)
st.markdown('<div class="info-text">Let’s handle missing values, encode categories, and scale your features — all in one place!</div>', unsafe_allow_html=True)

# ── LOAD DATA ────────────────────────────────────────────────────────────────
if 'uploaded_data' not in st.session_state:
    st.warning("⚠️ Please upload your data from the previous page.")
    st.stop()

df = st.session_state.uploaded_data.copy()

# ── FLAGS & OUTPUT OBJECTS ─────────────────────────────────────────────
encoding_done = False
missing_done = False
scaling_done = False
outlier_done = False
saved_encoder = None
saved_label_encoders = {}

# ── OUTPUT FOLDER ──────────────────────────────────────────────────────
os.makedirs("preprocessing_models", exist_ok=True)

# ── TWO-COLUMN LAYOUT ─────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

# LEFT COLUMN ──────────────────────────────────────────────────────────────────
with col1:
    st.markdown('<div class="section-title">🎭 Encode Categorical Features</div>', unsafe_allow_html=True)

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    all_cols = df.columns.tolist()

    selected_cat_cols = st.multiselect(
        "Select columns to encode (💡 Choose from recommended categorical columns if unsure):",
        all_cols,
        default=st.session_state.get('selected_cat_cols', []),
        help=f"Recommended categorical columns: {', '.join(cat_cols)}" if cat_cols else "No object-type columns detected."
    )

    encoding = st.selectbox(
        "Select encoding method:", 
        ["None", "OneHotEncoder", "OrdinalEncoder", "LabelEncoder"],
        index=["None", "OneHotEncoder", "OrdinalEncoder", "LabelEncoder"].index(
            st.session_state.get('encoding', "None")
        )
    )

    if encoding != "None" and selected_cat_cols:
        if encoding == "OneHotEncoder":
            df = pd.get_dummies(df, columns=selected_cat_cols, drop_first=True)
        elif encoding == "OrdinalEncoder":
            encoder = OrdinalEncoder()
            df[selected_cat_cols] = encoder.fit_transform(df[selected_cat_cols])
            joblib.dump(encoder, "preprocessing_models/encoder.pkl")
            saved_encoder = encoder
        elif encoding == "LabelEncoder":
            for col in selected_cat_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                saved_label_encoders[col] = le
            joblib.dump(saved_label_encoders, "preprocessing_models/encoder.pkl")
        encoding_done = True
        st.success("✅ Encoding applied and encoder saved.")
    elif cat_cols:
        st.warning(
            f"""⚠️ You have string-based categorical columns in your dataset that may need encoding.  
            👉 **Recommended categorical columns:**  
            {", ".join(cat_cols)}

            📌 Please select these or other appropriate columns and apply an encoding method.""")

    st.session_state.selected_cat_cols = selected_cat_cols
    st.session_state.encoding = encoding

    st.markdown('<div class="section-title">🧩 Handle Missing Values</div>', unsafe_allow_html=True)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    missing_before = df[num_cols].isnull().sum().sum()

    missing_strategy = st.selectbox(
        "Choose missing value method:", 
        ["None", "Remove", "Fill"],
        index=["None", "Remove", "Fill"].index(st.session_state.get('missing_strategy', "None"))
    )

    if missing_strategy != "None":
        if missing_strategy == "Remove":
            missing_cols = df.isnull().sum()
            missing_cols = missing_cols[missing_cols > 0]
            remove_cols = st.multiselect(
                "Select columns to remove with missing values:",
                missing_cols.index,
                default=st.session_state.get('remove_cols', [])
            )
            df.drop(columns=remove_cols, inplace=True)
            missing_done = True
            st.success(f"✅ Columns removed: {', '.join(remove_cols)}.")
            st.session_state.remove_cols = remove_cols
        elif missing_strategy == "Fill":
            missing_strategy_fill = st.radio(
                "Choose imputation strategy:", 
                ["mean", "median", "most_frequent"], 
                horizontal=True,
                index=["mean", "median", "most_frequent"].index(
                    st.session_state.get('missing_strategy_fill', "mean")
                )
            )
            if missing_strategy_fill:
                if missing_strategy_fill == "mean":
                    imputer = SimpleImputer(strategy="mean")
                elif missing_strategy_fill == "median":
                    imputer = SimpleImputer(strategy="median")
                else:
                    imputer = SimpleImputer(strategy="most_frequent")
                df[num_cols] = imputer.fit_transform(df[num_cols])
                missing_done = True
                st.success("✅ Missing values filled.")
            st.session_state.missing_strategy_fill = missing_strategy_fill

    elif missing_before == 0:
        st.info("ℹ️ No missing values detected in the dataset. You can proceed to the next step.")

    st.session_state.missing_strategy = missing_strategy

# RIGHT COLUMN ─────────────────────────────────────────────────────────────────
with col2:
    st.markdown('<div class="section-title">🎯 Choose Target Column</div>', unsafe_allow_html=True)
    target_col = st.selectbox(
        "Which column is your target variable?", 
        df.columns,
        index=df.columns.get_loc(st.session_state.get('target_column', df.columns[0]))
    )
    st.session_state.target_column = target_col

# ----------------- SCALING / NORMALIZATION -----------------
with col2:
    st.markdown('<div class="section-title">📊 Scaling / Normalization</div>', unsafe_allow_html=True)

    scaling = st.selectbox(
        "Select a scaler:", 
        ["None", "StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer"],
        index=["None", "StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer"].index(
            st.session_state.get('scaling', "None")
        )
    )

    if scaling != "None":

        all_cols_except_target = [col for col in num_cols if col != target_col]

        st.markdown("#### ⚙️ Choose columns to scale:")

        columns_to_scale = st.multiselect(
            "Select columns for scaling:",
            options=["Select All Columns"] + all_cols_except_target,
            default=st.session_state.get('columns_to_scale', [])
        )

        if "Select All Columns" in columns_to_scale:
            columns_to_scale = all_cols_except_target

        if columns_to_scale:
            if scaling == "StandardScaler":
                scaler = StandardScaler()
            elif scaling == "MinMaxScaler":
                scaler = MinMaxScaler()
            elif scaling == "RobustScaler":
                scaler = RobustScaler()
            elif scaling == "Normalizer":
                scaler = Normalizer()

            df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

            joblib.dump(scaler, "preprocessing_models/scaler.pkl")
            scaling_done = True
            st.success(f"✅ Features scaled: {', '.join(columns_to_scale)} and scaler saved.")
        else:
            st.warning("⚠️ No columns selected for scaling.")
        
        st.session_state.columns_to_scale = columns_to_scale

    else:
        st.info("ℹ️ You chose not to scale features. Proceed if your model doesn't require it.")

    st.session_state.scaling = scaling

    # ── SAVE ──────────────────────────────────────────────────────────────────────
    if st.button("💾 Save Preprocessed Data"):
        if missing_before > 0 and not missing_done:
            st.error("❌ Missing values are present and not handled. Please impute or remove columns before proceeding.")
        
        else:
            st.session_state.preprocessed_data = df
            st.success("✅ Preprocessing complete. You're ready for modeling!")

    if "preprocessed_data" in st.session_state and "target_column" in st.session_state:
        st.markdown('<div class="next-button-container">', unsafe_allow_html=True)
        if st.button("➡️ Next: Model Training"):
            from streamlit import switch_page
            switch_page("pages/3_🤖_Model_Training.py")
        st.markdown('</div>', unsafe_allow_html=True)
