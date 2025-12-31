import streamlit as st
import pytesseract
from PIL import Image
import pandas as pd
import re

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Project Number Extractor (Image)",
    layout="centered"
)

st.title("📸 Project Number Extractor from Screenshot")
st.write(
    "Upload a screenshot containing filenames.\n"
    "The app will extract **only the first project number** from each filename."
)

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
uploaded_image = st.file_uploader(
    "Upload screenshot image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_image:
    image = Image.open(uploaded_image)

    st.subheader("🖼 Uploaded Image")
    st.image(image, use_container_width=True)

    # --------------------------------------------------
    # OCR
    # --------------------------------------------------
    extracted_text = pytesseract.image_to_string(image)

    st.subheader("📄 Detected Text")
    st.text(extracted_text)

    # --------------------------------------------------
    # EXTRACT PROJECT NUMBERS
    # --------------------------------------------------
    project_numbers = []

    for line in extracted_text.splitlines():
        line = line.strip()
        if "_" in line:
            first_part = line.split("_")[0]
            if re.fullmatch(r"\d+", first_part):
                project_numbers.append(first_part)

    if project_numbers:
        df = pd.DataFrame({
            "Project Number": project_numbers
        })

        st.subheader("✅ Extracted Project Numbers")
        st.dataframe(df, use_container_width=True)

        # --------------------------------------------------
        # DOWNLOAD CSV
        # --------------------------------------------------
        csv = df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name="project_numbers.csv",
            mime="text/csv"
        )
    else:
        st.warning("No valid project numbers detected.")
