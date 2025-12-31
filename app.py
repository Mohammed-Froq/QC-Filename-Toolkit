import streamlit as st

st.set_page_config(
    page_title="QC Filename Toolkit",
    layout="wide"
)

st.title("🧰 QC Filename Toolkit")

tab1, tab2, tab3 = st.tabs([
    "📂 Artwork vs Packshot Filename Rule Checker",
    "📸 Project Number Extractor from Screenshot",
    "🔁 V01 (front) vs V02 (back) Filename Checker"
])

with tab1:
    exec(open("main.py", encoding="utf-8").read())

with tab2:
    exec(open("main1.py", encoding="utf-8").read())

with tab3:
    exec(open("main2.py", encoding="utf-8").read())
