# app.py
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance

# OCR
try:
    import pytesseract
except Exception:
    pytesseract = None

# Fuzzy
try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

st.set_page_config(page_title="V01 vs V02 — Filename Match Checker", layout="wide")
st.title("V01 (front) vs V02 (back) — Filename Match Checker")

EXT_RE = re.compile(r"\.(jpg|jpeg|png|tif|tiff|pdf|psd|ai|eps)$", re.IGNORECASE)


def auto_crop_text_region(img: Image.Image, pad: int = 12) -> Image.Image:
    g = np.array(img.convert("L"), dtype=np.int16)

    dx = np.abs(g[:, 1:] - g[:, :-1])
    dy = np.abs(g[1:, :] - g[:-1, :])
    edges = np.zeros_like(g, dtype=np.int32)
    edges[:, :-1] += dx
    edges[:-1, :] += dy

    row = edges.sum(axis=1)
    col = edges.sum(axis=0)

    r_thr = max(50_000, int(row.max() * 0.18))
    c_thr = max(50_000, int(col.max() * 0.18))

    def first_run(arr, thr, run=6):
        for i in range(0, len(arr) - run):
            if np.all(arr[i:i + run] > thr):
                return i
        return 0

    def last_run(arr, thr, run=6):
        for i in range(len(arr) - run - 1, 0, -1):
            if np.all(arr[i:i + run] > thr):
                return i + run
        return len(arr)

    top = first_run(row, r_thr)
    bottom = last_run(row, r_thr)
    left = first_run(col, c_thr)
    right = last_run(col, c_thr)

    w, h = img.size
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(w, right + pad)
    bottom = min(h, bottom + pad)

    if right - left < w * 0.3 or bottom - top < h * 0.3:
        return img

    return img.crop((left, top, right, bottom))


def preprocess_for_ocr(img: Image.Image, scale: int = 3) -> Image.Image:
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Contrast(img).enhance(2.2)
    w, h = img.size
    img = img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
    return img.point(lambda p: 255 if p > 170 else 0)


def ocr_image(img: Image.Image) -> str:
    if pytesseract is None:
        return ""
    whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-.()"
    cfg = f"--oem 3 --psm 6 -c tessedit_char_whitelist={whitelist}"
    return pytesseract.image_to_string(img, config=cfg)


def extract_filenames_from_text(text: str) -> list[str]:
    found = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for m in re.finditer(
            r"[A-Za-z0-9][A-Za-z0-9_\-\.()]{8,}\.(?:jpg|jpeg|png|tif|tiff|pdf|psd|ai|eps)",
            line,
            re.I,
        ):
            found.append(m.group(0))

    out, seen = [], set()
    for f in found:
        f = f.strip()
        if f and f not in seen:
            seen.add(f)
            out.append(f)
    return out


def key_strict(name: str, case_insensitive: bool, ignore_ext: bool) -> str:
    p = Path(name)
    base = p.stem if ignore_ext else p.name
    return base.casefold() if case_insensitive else base


def key_loose(name: str, case_insensitive: bool, ignore_ext: bool) -> str:
    p = Path(name)
    base = p.stem if ignore_ext else p.name
    s = base.casefold() if case_insensitive else base
    return re.sub(r"[^a-z0-9]+", "", s)


def build_map(names, make_key):
    m = {}
    for n in names:
        k = make_key(n)
        m.setdefault(k, []).append(n)
    return m


def best_fuzzy_match(k: str, candidates: list[str]) -> tuple[str | None, int]:
    if not candidates:
        return None, 0

    if fuzz is not None:
        best_c, best_s = None, -1
        for c in candidates:
            s = fuzz.ratio(k, c)
            if s > best_s:
                best_c, best_s = c, s
        return best_c, int(best_s)

    import difflib

    best_c, best_s = None, -1
    for c in candidates:
        s = int(difflib.SequenceMatcher(None, k, c).ratio() * 100)
        if s > best_s:
            best_c, best_s = c, s
    return best_c, best_s


# ---------------- UI (ONLY single screenshot) ----------------
st.caption("Upload ONE Finder screenshot. Front window = V01 (base), Back window = V02 (rework).")

case_insensitive = st.checkbox("Case-insensitive compare", value=True)
ignore_ext = st.checkbox("Ignore file extension (compare by stem only)", value=False)

use_loose = st.checkbox("Loose match (ignore spaces/_/-/.)", value=True)
use_fuzzy = st.checkbox("Fuzzy match for OCR typos (recommended)", value=True)
fuzzy_threshold = st.slider("Fuzzy threshold", 80, 100, 94, 1)

img_file = st.file_uploader("Upload Finder screenshot (PNG/JPG)", type=["png", "jpg", "jpeg"])

split_ratio = st.slider("Split position (top=V02, bottom=V01)", 0.30, 0.70, 0.48, 0.01)
flip = st.checkbox("Flip (top=V01, bottom=V02)", value=False)
pad = st.slider("Auto-crop padding", 0, 40, 12, 1)

if not img_file:
    st.stop()

if pytesseract is None:
    st.error("OCR not available. Install: pip install pytesseract (and install Tesseract on your OS).")
    st.stop()

img = Image.open(img_file)
w, h = img.size
y = int(h * split_ratio)

top = img.crop((0, 0, w, y))
bottom = img.crop((0, y, w, h))

if flip:
    v01_img, v02_img = top, bottom
else:
    v02_img, v01_img = top, bottom

v01_img_c = auto_crop_text_region(v01_img, pad=pad)
v02_img_c = auto_crop_text_region(v02_img, pad=pad)

c1, c2 = st.columns(2)
with c1:
    st.subheader("V01 crop (front)")
    st.image(v01_img_c, use_container_width=True)
with c2:
    st.subheader("V02 crop (back)")
    st.image(v02_img_c, use_container_width=True)

v01_ocr = ocr_image(preprocess_for_ocr(v01_img_c))
v02_ocr = ocr_image(preprocess_for_ocr(v02_img_c))

v01_names = extract_filenames_from_text(v01_ocr)
v02_names = extract_filenames_from_text(v02_ocr)

with st.expander("OCR text (debug)"):
    st.text_area("V01 OCR", v01_ocr, height=160)
    st.text_area("V02 OCR", v02_ocr, height=160)
    st.write("V01 extracted:", v01_names)
    st.write("V02 extracted:", v02_names)

if not v01_names:
    st.info("No V01 filenames detected.")
    st.stop()

# ---------------- compare ----------------
make_key = (lambda n: key_loose(n, case_insensitive, ignore_ext)) if use_loose else (
    lambda n: key_strict(n, case_insensitive, ignore_ext)
)

v01_map = build_map(v01_names, make_key)
v02_map = build_map(v02_names, make_key)

v01_keys = list(v01_map.keys())
v02_keys_set = set(v02_map.keys())
v02_keys_list = list(v02_map.keys())

rows = []
matched_count = 0

for k in v01_keys:
    v01_files = v01_map[k]
    status = "Not in V02"
    match_detail = ""

    if k in v02_keys_set:
        status = "Matched in V02"
        match_detail = ", ".join(v02_map[k][:3]) + (" ..." if len(v02_map[k]) > 3 else "")
        matched_count += len(v01_files)
    elif use_fuzzy and v02_keys_list:
        best_k, score = best_fuzzy_match(k, v02_keys_list)
        if best_k is not None and score >= fuzzy_threshold:
            status = f"Matched (fuzzy {score}%)"
            match_detail = ", ".join(v02_map[best_k][:3]) + (" ..." if len(v02_map[best_k]) > 3 else "")
            matched_count += len(v01_files)

    for orig in v01_files:
        rows.append({"V01 filename": orig, "Status": status, "V02 match (if any)": match_detail})

df = pd.DataFrame(rows)
missing = (df["Status"] == "Not in V02").sum()

m1, m2, m3, m4 = st.columns(4)
m1.metric("V01 files", len(v01_names))
m2.metric("V02 files", len(v02_names))
m3.metric("Matched", matched_count)
m4.metric("Missing in V02", missing)

st.subheader("Result (V01 as base)")
st.dataframe(df, use_container_width=True, hide_index=True)

# V02-only (extra)
v01_all_keys = set(v01_map.keys())
extra_rows = []
for k in v02_map.keys():
    if k not in v01_all_keys:
        for orig in v02_map[k]:
            extra_rows.append({"V02 filename": orig})

with st.expander("V02-only (extra)"):
    st.dataframe(pd.DataFrame(extra_rows), use_container_width=True, hide_index=True)

st.download_button(
    "Download CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="v01_vs_v02_results.csv",
    mime="text/csv",
)
