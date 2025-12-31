import streamlit as st
from PIL import Image
import pytesseract
import re
import cv2
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Filename Rule Checker", layout="centered")
st.title("📂 Artwork vs Packshot Filename Rule Checker")

uploaded_image = st.file_uploader(
    "Upload image containing filenames",
    type=["png", "jpg", "jpeg"]
)

# =========================
# OCR
# =========================
def extract_text(image: Image.Image) -> str:
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)[1]
    return pytesseract.image_to_string(img)

# =========================
# FILENAME EXTRACTION
# =========================
def extract_filenames(text):
    if not isinstance(text, str):
        return [], []

    text = text.replace("\n", " ")

    jpg_files = re.findall(
        r"[A-Za-z0-9_\-]{10,}\.(?:jpg|jpeg)",
        text,
        flags=re.IGNORECASE
    )

    ai_files = re.findall(
        r"[A-Za-z0-9_\-]{10,}\.ai",
        text,
        flags=re.IGNORECASE
    )

    return sorted(set(jpg_files)), sorted(set(ai_files))

def ai_priority(ai_name: str) -> int:
    name = ai_name.lower()
    if "_back_" in name:  return 1
    if "_front_" in name: return 2
    if "_new_" in name:   return 3
    if "_ty_" in name:    return 4
    return 99

# =========================
# HELPERS
# =========================
def split_segments(filename: str):
    name = re.sub(r"\.(jpg|jpeg|ai)$", "", filename, flags=re.IGNORECASE)
    return name.split("_")

def normalize(s: str) -> str:
    s = s.lower().replace("_", "-")
    s = re.sub(r"[^a-z0-9\-]+", "", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s

def first_qty_index(segments, start_idx: int):
    for i in range(start_idx, len(segments)):
        if re.match(r"^\d+", segments[i]):
            return i
    return None

def qty_number(seg: str) -> str:
    m = re.match(r"^(\d+)", seg)
    return m.group(1) if m else ""

def badge(ok: bool) -> str:
    return "✅" if ok else "❌"

def fmt_seg(seg_list):
    # show with indices so it’s obvious what is what
    return {str(i+1): seg_list[i] for i in range(len(seg_list))}

def html_highlight_segment(seg_list, highlight_idx_set):
    # highlight 1-based indices in red
    parts = []
    for i, val in enumerate(seg_list):
        idx1 = i + 1
        if idx1 in highlight_idx_set:
            parts.append(f'<span style="color:#d00;font-weight:700">{idx1}:{val}</span>')
        else:
            parts.append(f'{idx1}:{val}')
    return " | ".join(parts)

# =========================
# RULE CHECK (WITH DEBUG DETAILS)
# =========================
def check_rules(jpg, ai):
    jpg_seg = split_segments(jpg)
    ai_seg  = split_segments(ai)

    details = []  # list of dicts: {"rule":..., "ok":..., "left":..., "right":..., "jpg_idx":..., "ai_idx":...}

    # Rule 1: segment 1 match
    ok1 = (len(jpg_seg) > 0 and len(ai_seg) > 0 and jpg_seg[0] == ai_seg[0])
    details.append({
        "rule": "Segment 1 match",
        "ok": ok1,
        "left": jpg_seg[0] if len(jpg_seg) > 0 else "",
        "right": ai_seg[0] if len(ai_seg) > 0 else "",
        "jpg_idx": 1, "ai_idx": 1
    })

    # Rule 2: EAN match (jpg seg2 == ai seg3)
    ok2 = (len(jpg_seg) > 1 and len(ai_seg) > 2 and jpg_seg[1] == ai_seg[2])
    details.append({
        "rule": "EAN match (jpg seg 2 == ai seg 3)",
        "ok": ok2,
        "left": jpg_seg[1] if len(jpg_seg) > 1 else "",
        "right": ai_seg[2] if len(ai_seg) > 2 else "",
        "jpg_idx": 2, "ai_idx": 3
    })

    # Product + qty extraction
    jpg_qty_i = first_qty_index(jpg_seg, 3)  # from seg4 onwards
    if jpg_qty_i is None:
        jpg_product = normalize("-".join(jpg_seg[3:4]))
        jpg_qty_num = ""
        jpg_qty_raw = ""
    else:
        jpg_product = normalize("-".join(jpg_seg[3:jpg_qty_i]))
        jpg_qty_raw = jpg_seg[jpg_qty_i]
        jpg_qty_num = qty_number(jpg_qty_raw)

    ai_qty_i = first_qty_index(ai_seg, 4)  # from seg5 onwards (after KV)
    if ai_qty_i is None:
        ai_product = normalize("-".join(ai_seg[4:]))
        ai_qty_num = ""
        ai_qty_raw = ""
    else:
        ai_product = normalize("-".join(ai_seg[4:ai_qty_i]))
        ai_qty_raw = ai_seg[ai_qty_i]
        ai_qty_num = qty_number(ai_qty_raw)

    ok3 = (jpg_product != "" and jpg_product == ai_product)
    details.append({
        "rule": "Product match (normalized)",
        "ok": ok3,
        "left": jpg_product,
        "right": ai_product,
        "jpg_idx": None, "ai_idx": None  # derived, not a single segment
    })

    # Qty rule (only enforced if both sides found numbers)
    if jpg_qty_num and ai_qty_num:
        ok4 = (jpg_qty_num == ai_qty_num)
    else:
        ok4 = True  # do not block match if qty can't be parsed
    details.append({
        "rule": "Qty number match (e.g. 2pcs == 2stuks)",
        "ok": ok4,
        "left": f"{jpg_qty_raw} -> {jpg_qty_num}" if jpg_qty_raw else "(not detected)",
        "right": f"{ai_qty_raw} -> {ai_qty_num}" if ai_qty_raw else "(not detected)",
        "jpg_idx": (jpg_qty_i + 1) if jpg_qty_i is not None else None,
        "ai_idx": (ai_qty_i + 1) if ai_qty_i is not None else None
    })

    results = {d["rule"]: d["ok"] for d in details}
    score = sum(1 for d in details if d["ok"])

    return results, details, jpg_seg, ai_seg, score

# =========================
# MAIN APP
# =========================
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Reading filenames..."):
        text = extract_text(image)

    st.subheader("📄 Detected Text")
    st.code(text)

    jpg_files, ai_files = extract_filenames(text)

    st.subheader("📌 Detected Filenames")

    if not jpg_files:
        st.error("❌ No Packshot (.jpg) files detected")
        st.stop()

    if not ai_files:
        st.error("❌ No Artwork (.ai) files detected")
        st.stop()

    st.subheader("📊 Filename Matching Results")

    sorted_ai_files = sorted(ai_files, key=ai_priority)

    for jpg in jpg_files:
        best = None  # (pass_bool, score, ai, results, details, jpg_seg, ai_seg)

        for ai in sorted_ai_files:
            results, details, jpg_seg, ai_seg, score = check_rules(jpg, ai)
            passed = all(results.values())

            cand = (passed, score, ai, results, details, jpg_seg, ai_seg)
            if best is None:
                best = cand
            else:
                # Prefer PASS, then higher score, then priority order already handled by iteration
                if cand[0] and not best[0]:
                    best = cand
                elif cand[0] == best[0] and cand[1] > best[1]:
                    best = cand

            if passed:
                break

        passed, score, best_ai, results, details, jpg_seg, ai_seg = best

        st.markdown("---")
        if passed:
            st.success(f"📸 Packshot: {jpg}")
            st.success(f"🎨 Artwork: {best_ai}")

            for d in details:
                st.success(f"{badge(d['ok'])} {d['rule']}")

            st.success("🟢 FINAL QC RESULT: PASS")

            st.subheader("🔍 Segment Breakdown")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Packshot segments (indexed)**")
                st.json(fmt_seg(jpg_seg))
            with col2:
                st.write("**Artwork segments (indexed)**")
                st.json(fmt_seg(ai_seg))

        else:
            st.error(f"📸 Packshot: {jpg}")
            st.error("🔴 FINAL QC RESULT: FAIL")
            st.write("**Best candidate artwork (closest match):**")
            st.code(best_ai)

            st.write("**Rule-by-rule mismatches (shows compared values):**")
            for d in details:
                if d["ok"]:
                    st.success(f"{badge(True)} {d['rule']}")
                else:
                    st.error(f"{badge(False)} {d['rule']}")
                st.write(f"- JPG: `{d['left']}`")
                st.write(f"- AI : `{d['right']}`")

            # highlight only the segments involved in failing rules
            jpg_bad = set()
            ai_bad = set()
            for d in details:
                if not d["ok"]:
                    if isinstance(d.get("jpg_idx"), int):
                        jpg_bad.add(d["jpg_idx"])
                    if isinstance(d.get("ai_idx"), int):
                        ai_bad.add(d["ai_idx"])

            st.write("**Segments (red = involved in a failed rule):**")
            st.markdown(
                f"<div><b>JPG:</b> {html_highlight_segment(jpg_seg, jpg_bad)}</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div><b>AI :</b> {html_highlight_segment(ai_seg, ai_bad)}</div>",
                unsafe_allow_html=True
            )

            st.caption(f"Match score: {score}/{len(details)} rules true")
