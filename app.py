import streamlit as st

# ======================================================
# GLOBAL PAGE CONFIG (ONLY ONCE)
# ======================================================
st.set_page_config(
    page_title="QC Filename Toolkit",
    layout="wide"
)

st.title("🧰 QC Filename Toolkit")

# ======================================================
# TABS
# ======================================================
tab1, tab2, tab3 = st.tabs([
    "📂 Artwork vs Packshot Filename Rule Checker",
    "📸 Project Number Extractor from Screenshot",
    "🔁 V01 (front) vs V02 (back) Filename Checker"
])

# ======================================================
# TAB 1 — RUN main.py (AS-IS)
# ======================================================
with tab1:
    exec(open(
        "/Users/froquser/Desktop/filename_comparison/main.py",
        encoding="utf-8"
    ).read())

# ======================================================
# TAB 2 — RUN main1.py (AS-IS)
# ======================================================
with tab2:
    exec(open(
        "/Users/froquser/Desktop/filename_comparison/main1.py",
        encoding="utf-8"
    ).read())

# ======================================================
# TAB 3 — RUN main2.py (AS-IS)
# ======================================================
with tab3:
    exec(open(
        "/Users/froquser/Desktop/filename_comparison/main2.py",
        encoding="utf-8"
    ).read())
