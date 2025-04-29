import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime

# ── PAGE SETUP ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔮 Prediction",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide sidebar and collapse arrow
hide_st_style = """
    <style>
    [data-testid="stSidebar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# ── STYLING ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .big-font {font-size:22px; font-weight:bold;}
    .small-font {font-size:16px;}
    .summary-box {background-color: #f0f2f6; padding: 20px; border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── HEADER ──────────────────────────────────────────────────────────────────
st.title("🔮 Predict with Your Trained Model")
st.markdown("---")

# ── MAIN 2-COLUMN LAYOUT ─────────────────────────────────────────────────────
left_col, right_col = st.columns(2)

# LEFT COLUMN — model selection to prediction
with left_col:
    # ── 1) PICK A MODEL BUNDLE ─────────────────────────────────────────────
    model_files = [f for f in os.listdir(".") if f.endswith("_bundle.pkl")]
    default_index = model_files.index(st.session_state["selected_model"]) if "selected_model" in st.session_state and st.session_state["selected_model"] in model_files else 0
    selected_model = st.selectbox("📁 Select a trained model", model_files, index=default_index)
    if not selected_model:
        st.stop()
    st.session_state["selected_model"] = selected_model

    # ── 2) LOAD MODEL + METADATA + y_scaler ────────────────────────────────
    try:
        if "bundle" not in st.session_state or st.session_state.get("bundle_model_file") != selected_model:
            bundle = joblib.load(selected_model)
            st.session_state["bundle"] = bundle
            st.session_state["bundle_model_file"] = selected_model
        else:
            bundle = st.session_state["bundle"]

        model          = bundle["model"]
        expected_feats = bundle["features"]
        model_name     = bundle.get("model_name", selected_model.replace("_bundle.pkl", ""))
        cat_cols       = bundle.get("cat_cols", [])
        y_scaler       = bundle.get("y_scaler", None)

        st.session_state["model"] = model
        st.session_state["expected_feats"] = expected_feats
        st.session_state["model_name"] = model_name
        st.session_state["cat_cols"] = cat_cols
        st.session_state["y_scaler"] = y_scaler

        st.success(f"✅ Loaded model '{model_name}'")
    except Exception as e:
        st.error(f"❌ Could not load model bundle: {e}")
        st.stop()

    # Fallback scaler
    if st.session_state["y_scaler"] is None:
        scaler_path = f"{model_name}_y_scaler.pkl"
        if os.path.exists(scaler_path):
            st.session_state["y_scaler"] = joblib.load(scaler_path)
            st.info("ℹ️ Loaded separate target scaler")
        else:
            st.info("ℹ️ No target scaler found; predictions will remain on model scale")

    # ── 3) LOAD PREPROCESSING OBJECTS ──────────────────────────────────────
    try:
        if "feat_scaler" not in st.session_state or "encoder" not in st.session_state:
            feat_scaler = joblib.load("preprocessing_models/scaler.pkl")
            encoder     = joblib.load("preprocessing_models/encoder.pkl")
            st.session_state["feat_scaler"] = feat_scaler
            st.session_state["encoder"] = encoder
        else:
            feat_scaler = st.session_state["feat_scaler"]
            encoder = st.session_state["encoder"]
    except Exception as e:
        st.error(f"❌ Could not load preprocessing objects: {e}")
        st.stop()

    # ── 4) UPLOAD UNSEEN DATA ──────────────────────────────────────────────
    st.markdown("### 📤 Upload Unseen Data for Prediction")
    file_type = st.radio("File type:", ["--", "CSV", "Excel", "JSON"], horizontal=True)
    df = None

    if file_type != "--":
        ext = {"CSV": "csv", "Excel": "xlsx", "JSON": "json"}[file_type]
        uploaded = st.file_uploader("📎 Drag & drop here", type=[ext])
        if uploaded:
            try:
                if file_type == "CSV":
                    df = pd.read_csv(uploaded)
                elif file_type == "Excel":
                    df = pd.read_excel(uploaded)
                else:
                    df = pd.read_json(uploaded)

                st.session_state["uploaded_df"] = df
                st.session_state["uploaded_filename"] = uploaded.name

                st.subheader("🔍 Uploaded Data Preview")
                st.dataframe(df.head())

                st.text(f"❗ Expected features: {expected_feats}")
                st.text(f"ℹ️ Your columns: {df.columns.tolist()}")

            except Exception as e:
                st.error(f"⚠️ Error reading file: {e}")
                st.stop()
        else:
            st.info("ℹ️ Please upload a file to proceed.")
            st.stop()
    else:
        st.info("ℹ️ Please select a file type to upload.")
        st.stop()

    # Only proceed if df is not None
    if df is not None:
        # ── 5) ENCODE & SCALE ─────────────────────────────────────────────────
        if isinstance(encoder, dict):
            for c, le in encoder.items():
                if c in df.columns:
                    df[c] = le.transform(df[c])
        else:
            obj_cols = df.select_dtypes(include="object").columns
            if len(obj_cols) > 0:
                df[obj_cols] = encoder.transform(df[obj_cols])

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        df[num_cols] = feat_scaler.transform(df[num_cols])

        missing = [c for c in expected_feats if c not in df]
        if missing:
            st.error(f"❌ Missing features: {missing}")
            st.stop()
        extra = [c for c in df if c not in expected_feats]
        if extra:
            st.warning(f"⚠️ Dropping extra columns: {extra}")
        df = df[expected_feats]

        # ── 6) PREDICT & INVERSE-TRANSFORM ────────────────────────────────────
        if st.button("🚀 Predict"):
            preds = model.predict(df)
            df["Prediction"] = preds

            df_rescaled = df.copy()
            df_rescaled[num_cols] = feat_scaler.inverse_transform(df[num_cols])

            if st.session_state["y_scaler"] is not None:
                try:
                    df_rescaled["Prediction_Rescaled"] = st.session_state["y_scaler"].inverse_transform(preds.reshape(-1, 1)).ravel()
                except Exception:
                    df_rescaled["Prediction_Rescaled"] = preds
            else:
                df_rescaled["Prediction_Rescaled"] = preds

            if isinstance(encoder, dict):
                for c, le in encoder.items():
                    if c in df_rescaled:
                        df_rescaled[c] = le.inverse_transform(df_rescaled[c].astype(int))
            else:
                if cat_cols:
                    df_rescaled[cat_cols] = encoder.inverse_transform(df_rescaled[cat_cols])

            st.success("✅ Prediction Complete!")

            st.session_state["pred_scaled"] = df
            st.session_state["pred_rescaled"] = df_rescaled

# RIGHT COLUMN — predictions
with right_col:
    st.markdown("### 📈 Prediction Results")
    if "pred_scaled" in st.session_state:
        st.subheader("🔍 Scaled Input + Raw Prediction")
        st.dataframe(st.session_state["pred_scaled"].head())

        st.subheader("🔁 Original Scale + Decoded Categories")
        st.dataframe(st.session_state["pred_rescaled"].head())

        csv = st.session_state["pred_rescaled"].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions CSV", csv, "predictions.csv", "text/csv")

        st.markdown("---")

        if st.button("📋 Summary"):
            with st.expander("🧾 Summary Details", expanded=True):
                st.markdown("<div class='summary-box'>", unsafe_allow_html=True)
                st.markdown(f"<div class='big-font'>📅 Summary as of {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"<div class='small-font'>🧠 Selected Model: <b>{st.session_state.get('model_name', 'Not selected')}</b></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='small-font'>📄 Uploaded File: <b>{st.session_state.get('uploaded_filename', 'No file uploaded')}</b></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='small-font'>📊 Prediction Done: <b>{'✅ Yes' if 'pred_scaled' in st.session_state else '❌ No'}</b></div>", unsafe_allow_html=True)

                st.markdown("<br><hr>", unsafe_allow_html=True)
                st.markdown("🎉 **Thank you for using the Regression Playground App!**")
                st.markdown("🙌 Proudly crafted by **Cerulean Solutions** 💙")

                st.markdown("</div>", unsafe_allow_html=True)
