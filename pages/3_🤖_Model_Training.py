import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, GridSearchCV, KFold, cross_val_score
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, RANSACRegressor, 
    HuberRegressor, QuantileRegressor, BayesianRidge, SGDRegressor
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Model Training", layout="wide")

# ── HIDE SIDEBAR & STYLES ───────────────────────────────────────────────
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
        .big-title {
            font-size: 36px;
            font-weight: bold;
            color: #4F8BF9;
            text-align: center;
            margin-bottom: 30px;
        }
        .center-sub {
            font-size: 18px;
            text-align: center;
        }
        .sub-title {
            font-size: 24px;
            font-weight: bold;
            color: #FF6F61;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .metric-text {
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# ── PAGE TITLE ──────────────────────────────────────────────────────────
st.markdown("<h1 class='big-title'>🧠 Model Training Module</h1>", unsafe_allow_html=True)

# ── SESSION DATA CHECK ──────────────────────────────────────────────────
if 'preprocessed_data' not in st.session_state or 'target_column' not in st.session_state:
    st.warning("⚠️ Please complete preprocessing step before training.")
    st.stop()

df = st.session_state.preprocessed_data.copy()
target = st.session_state.target_column
X = df.drop(columns=[target])
y = df[target]

# ── LAYOUT SPLIT ────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.2, 1.8])

# ── LEFT SIDE: INPUT PANEL ──────────────────────────────────────────────
with left_col:
    st.markdown("<div class='sub-title'>🎛️ Validation Strategy</div>", unsafe_allow_html=True)
    validation_method = st.selectbox(
        "Choose validation technique:",
        ["None", "Train/Test Split", "KFold Cross-Validation", "GridSearchCV"],
        index=["None", "Train/Test Split", "KFold Cross-Validation", "GridSearchCV"].index(
            st.session_state.get("validation_method", "None")
        )
    )
    st.session_state.validation_method = validation_method

    st.markdown("<div class='sub-title'>🧪 Dataset Split</div>", unsafe_allow_html=True)
    test_size = st.slider(
        "Test set size (%)", min_value=10, max_value=50,
        value=st.session_state.get("test_size", 20), step=5
    )
    st.session_state.test_size = test_size

    st.markdown("<div class='sub-title'>⚙️ Model Type</div>", unsafe_allow_html=True)
    model_type = st.radio(
        "Choose model type:",
        ["-- Select File Type --", "Linear Models", "Ensemble Models"],
        horizontal=True,
        index=0 if st.session_state.get("model_type", "Linear Models") == "-- Select File Type --" else 1
    )
    st.session_state.model_type = model_type

    st.markdown("<div class='sub-title'>📌 Model Selection</div>", unsafe_allow_html=True)
    models = ["None"] + (
        ["LinearRegression", "Ridge", "Lasso", "ElasticNet", "RANSACRegressor",
         "HuberRegressor", "QuantileRegressor", "BayesianRidge", "SGDRegressor"]
        if model_type == "Linear Models"
        else ["RandomForestRegressor", "GradientBoostingRegressor", "AdaBoostRegressor"]
    )
    model_name = st.selectbox(
        "Select model:",
        models,
        index=models.index(st.session_state.get("model_name", models[0]))
    )
    st.session_state.model_name = model_name

    model_dict = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "RANSACRegressor": RANSACRegressor(),
        "HuberRegressor": HuberRegressor(),
        "QuantileRegressor": QuantileRegressor(),
        "BayesianRidge": BayesianRidge(),
        "SGDRegressor": SGDRegressor(),
        "RandomForestRegressor": RandomForestRegressor(),
        "GradientBoostingRegressor": GradientBoostingRegressor(),
        "AdaBoostRegressor": AdaBoostRegressor()
    }

# ── TRAINING BUTTON ─────────────────────────────────────────────────────
if st.button("🚀 Train Model"):
    model = model_dict[model_name]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=42
    )

    # Scale target
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

    if validation_method == "Train/Test Split":
        model.fit(X_train, y_train_scaled)
        y_pred_scaled = model.predict(X_test)
        scores = None

    elif validation_method == "KFold Cross-Validation":
        kf = KFold(n_splits=5)
        scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
        model.fit(X_train, y_train_scaled)
        y_pred_scaled = model.predict(X_test)

    elif validation_method == "GridSearchCV":
        param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0]} if hasattr(model, "alpha") else {}
        if param_grid:
            grid = GridSearchCV(model, param_grid, cv=3)
            grid.fit(X_train, y_train_scaled)
            model = grid.best_estimator_
        model.fit(X_train, y_train_scaled)
        y_pred_scaled = model.predict(X_test)
        scores = None

    else:
        model.fit(X_train, y_train_scaled)
        y_pred_scaled = model.predict(X_test)
        scores = None

    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # Save to session state
    st.session_state.trained_model = model
    st.session_state.y_test = y_test
    st.session_state.y_pred = y_pred
    st.session_state.scores = scores
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_scaler = y_scaler

    # Clean up old bundles
    for f in os.listdir("."):
        if f.endswith("_bundle.pkl"):
            os.remove(f)

    # Save model bundle
    model_bundle = {
        "model": model,
        "features": X_train.columns.tolist(),
        "model_name": model_name,
        "y_scaler": y_scaler,
    }
    model_filename = f"{model_name}_bundle.pkl"
    joblib.dump(model_bundle, model_filename)
    st.session_state.model_saved = model_filename

# ── RIGHT SIDE: RESULTS DASHBOARD ───────────────────────────────────────
with right_col:
    st.markdown("<div class='big-title'>📊 Results Dashboard</div>", unsafe_allow_html=True)

    if "trained_model" in st.session_state:
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.markdown("<div class='sub-title'>📈 Metrics</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-text'>✅ MSE: {mse:.4f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-text'>✅ MAE: {mae:.4f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-text'>✅ R² Score: {r2:.4f}</div>", unsafe_allow_html=True)

        if st.session_state.scores is not None:
            st.markdown(f"**Cross-Validation R² Scores:** {np.round(st.session_state.scores, 3)}")

        st.markdown("<div class='sub-title'>🖼️ Visualizations</div>", unsafe_allow_html=True)
        fig1, ax1 = plt.subplots()
        sns.residplot(x=y_test, y=y_pred, lowess=True, ax=ax1, color="purple")
        ax1.set_title("Residual Plot")
        st.pyplot(fig1)

        st.success(f"✅ Model saved as {st.session_state.model_saved}")

        if st.button("➡️ Next: Predict Unseen Data"):
            from streamlit import switch_page
            switch_page("pages/4_🔮_Prediction.py")
