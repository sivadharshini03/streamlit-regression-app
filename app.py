import streamlit as st

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Regression Playground",
    layout="centered",
    initial_sidebar_state="collapsed"  # Sidebar always hidden
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        body {
            background-color: #f0f8ff;
        }
        .main-title {
            font-size: 52px;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-top: 30px;
        }
        .subtitle {
            font-size: 38px;
            color: #222;
            text-align: center;
            margin-bottom: 10px;
        }
        .tagline {
            font-size: 28px;
            color: #666;
            text-align: center;
            margin-bottom: 40px;
        }
        .login-links {
            position: absolute;
            top: 10px;
            right: 30px;
            font-size: 18px;
        }
        .login-links a {
            margin-left: 15px;
            text-decoration: none;
            color: #007BFF;
        }
        .try-button {
            display: flex;
            justify-content: center;
            margin-top: 30px;
        }
        .ball {
            width: 40px;
            height: 40px;
            background-color: #ff6f61;
            border-radius: 50%;
            position: relative;
            animation: bounce 1.8s infinite alternate ease-in-out;
            margin: 0 auto 20px auto;
        }
        @keyframes bounce {
            0% { transform: translateY(0); }
            100% { transform: translateY(60px); }
        }
        /* Hide sidebar and its toggle */
        [data-testid="stSidebar"] {
            display: none !important;
        }
        [data-testid="collapsedControl"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# ── LOGIN / SIGNUP PLACEHOLDER ────────────────────────────────────────────────
st.markdown("""
<div class="login-links">
    <a href="#">Login</a> | <a href="#">Sign Up</a>
</div>
""", unsafe_allow_html=True)

# ── TITLES & ANIMATION ─────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Cerulean Solutions</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Welcome to Regression Playground 🎯</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">Build, Train, and Predict with Ease!</div>', unsafe_allow_html=True)
st.markdown('<div class="ball"></div>', unsafe_allow_html=True)

# Centered Try the App Button using Streamlit layout
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Try the App", use_container_width=True):
        st.switch_page("pages/1_📂_Upload_Data.py")
