import io
import re
import hashlib
from io import BytesIO
from typing import List, Tuple

import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
from rapidfuzz import process, fuzz


# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="MRS PDF → BOQ", layout="wide")
st.title("📘 MRS / SoR PDF → BOQ Builder")

st.markdown(
    "Upload your **MRS/SoR PDF**. Type item keywords, select a match, "
    "enter quantity & rate type, then add to BOQ and download Excel."
)


# ---------------------------
# PDF → table parser (inline)
# ---------------------------
CHAPTER_RX = re.compile(
    r'(CHAPTER\s*NO?\.?\s*\d+.*|Chap-?\s*\d+.*|Chapter\s*\d+.*)',
    re.IGNORECASE
)
ITEM_RX = re.compile(r'^\s*(([ivxlcdm]+\))|(\d+\))|([a-z]\)))', re.IGNORECASE)


def _split_unit_value(cell: str) -> Tuple[str, str]:
    """Split a cell like 'Per Rft. 1,003.95' → ('Per Rft.', '1003.95')."""
    if not cell:
        return (None, None)
    parts = cell.split()
    if len(parts) >= 2:
        unit = " ".join(parts[:-1])
        val = parts[-1].replace(",", "")
        return (unit, val)
    return (None, cell.replace(",", ""))


def _find_col(header_norm: List[str], cands: List[str]):
    for i, h in enumerate(header_norm):
        for c in cands:
            if c in h:
                return i
    return None


def extract_tables_with_context_from_bytes(pdf_bytes: bytes) -> pd.DataFrame:
    """
    Parse tables from a PDF (bytes) with page + chapter context.
    Returns a tidy DataFrame ready for search/BOQ.
    """
    rows = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        current_chapter = None
        for page_idx, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""

            # Update current chapter from visible text
            for line in page_text.splitlines():
                m = CHAPTER_RX.search(line)
                if m:
                    current_chapter = m.group(0).strip()
                    break

            # Try extracting tables
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []

            if not tables:
                # Fallback: store lines as unstructured descriptions
                for line in page_text.splitlines():
                    if not line.strip():
                        continue
                    rows.append({
                        "chapter": current_chapter,
                        "section": None,
                        "item_no": None,
                        "description": line.strip(),
                        "unit_british": None,
                        "labour_british": None,
                        "composite_british": None,
                        "unit_metric": None,
                        "labour_metric": None,
                        "composite_metric": None,
                        "spec_no": None,
                        "remarks": None,
                        "page_no": page_idx,
                    })
                continue

            for tbl in tables:
                tbl = [[(c or "").strip() for c in row] for row in tbl]
                if len(tbl) < 2:
                    continue

                # Try to detect header row
                header_idx = None
                for i, row in enumerate(tbl[:3]):
                    joined = " ".join(row).lower()
                    if "rate" in joined and ("unit" in joined or "labour" in joined or "composite" in joined):
                        header_idx = i
                        break

                if header_idx is None:
                    header = [f"col_{i}" for i in range(len(tbl[0]))]
                    data_rows = tbl
                else:
                    header = tbl[header_idx]
                    data_rows = tbl[header_idx + 1:]

                header_norm = [h.lower().strip() for h in header]

                desc_col = _find_col(header_norm, ["description"]) or 0
                brit_col = _find_col(header_norm, ["rate (british system)", "british", "rate (british)"])
                lab_b_col = _find_col(header_norm, ["labour"])
                comp_b_col = _find_col(header_norm, ["composite"])
                metric_col = _find_col(header_norm, ["rate (metric system)", "metric", "rate (metric)"])
                spec_col = _find_col(header_norm, ["spec", "spec."])
                rem_col = _find_col(header_norm, ["remark"])

                def safe(row, i):
                    return row[i].strip() if (i is not None and i < len(row) and row[i]) else None

                for r in data_rows:
                    if all(not (c and c.strip()) for c in r):
                        continue

                    description = r[desc_col] if desc_col is not None and desc_col < len(r) else " ".join(r).strip()

                    item_no = None
                    m = ITEM_RX.match(description or "")
                    if m:
                        item_no = (m.group(0) or "").strip()

                    unit_british, labour_british, composite_british = None, None, None
                    unit_metric, labour_metric, composite_metric = None, None, None

                    # British
                    if brit_col is not None:
                        ub = safe(r, brit_col)
                        unit_british, _ = _split_unit_value(ub)
                    if lab_b_col is not None:
                        lb = safe(r, lab_b_col)
                        labour_british = lb.replace(",", "") if lb else None
                    if comp_b_col is not None:
                        cb = safe(r, comp_b_col)
                        composite_british = cb.replace(",", "") if cb else None

                    # Metric
                    if metric_col is not None:
                        um = safe(r, metric_col)
                        unit_metric, _ = _split_unit_value(um)
                        # Often Labour/Composite follow metric column
                        lm = safe(r, metric_col + 1)
                        cm = safe(r, metric_col + 2)
                        labour_metric = lm.replace(",", "") if lm else None
                        composite_metric = cm.replace(",", "") if cm else None

                    rows.append({
                        "chapter": current_chapter,
                        "section": None,
                        "item_no": item_no,
                        "description": description,
                        "unit_british": unit_british,
                        "labour_british": labour_british,
                        "composite_british": composite_british,
                        "unit_metric": unit_metric,
                        "labour_metric": labour_metric,
                        "composite_metric": composite_metric,
                        "spec_no": safe(r, spec_col),
                        "remarks": safe(r, rem_col),
                        "page_no": page_idx,
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["description"] = (
        df["description"].astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df[df["description"].str.len() > 1].copy()
    return df


# ---------------------------
# Caching by file content
# ---------------------------
@st.cache_data(show_spinner=False)
def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@st.cache_data(show_spinner=True)
def parse_pdf_cached(pdf_bytes: bytes) -> pd.DataFrame:
    return extract_tables_with_context_from_bytes(pdf_bytes)


# ---------------------------
# Sidebar: upload PDF
# ---------------------------
st.sidebar.header("📁 Upload PDF")
pdf_file = st.sidebar.file_uploader("MRS/SoR PDF", type=["pdf"])

df = None
if pdf_file is not None:
    pdf_bytes = pdf_file.read()
    file_hash = _hash_bytes(pdf_bytes)
    with st.spinner("Parsing PDF… (first time only, then cached)"):
        df = parse_pdf_cached(pdf_bytes)

    if df is None or df.empty:
        st.error("No table-like content found. If your PDF is scanned, OCR it first or share a sample page to tune the parser.")
else:
    st.info("Upload your MRS/SoR PDF to begin.")


# ---------------------------
# When we have data, show UI
# ---------------------------
if df is not None and not df.empty:
    st.success(f"Parsed {len(df):,} rows from PDF.")
    # Filters
    c1, c2 = st.columns(2)
    with c1:
        chapters = sorted([c for c in df["chapter"].dropna().unique().tolist() if c])
        ch_pick = st.multiselect("Filter by Chapter", chapters)
    with c2:
        pmin, pmax = int(df["page_no"].min()), int(df["page_no"].max())
        pr = st.slider("Page range", min_value=pmin, max_value=pmax, value=(pmin, pmax))

    dff = df.copy()
    if ch_pick:
        dff = dff[dff["chapter"].isin(ch_pick)]
    dff = dff[(dff["page_no"] >= pr[0]) & (dff["page_no"] <= pr[1])]

    # Search box with fuzzy suggestions
    st.subheader("🔎 Search")
    query = st.text_input("Type item keywords (e.g., 'Class-III 610 mm sewer')", "")
    top_n = st.slider("Suggestions to show", 5, 20, 8)

    def get_suggestions(q, pool, n=8):
        if not q or len(q.strip()) < 2:
            return []
        return process.extract(q, pool, scorer=fuzz.WRatio, limit=n)

    pool = dff["description"].astype(str).tolist()
    suggestions = get_suggestions(query, pool, n=top_n)

    pick = None
    if suggestions:
        labels = [f"{txt}  —  (score {int(score)})" for txt, score, _ in suggestions]
        idx = st.radio("Closest matches", list(range(len(labels))), format_func=lambda i: labels[i], index=0)
        pick = suggestions[idx][0]

    # Show selected item details & add to BOQ
    if pick:
        sel = dff.loc[dff["description"] == pick].head(1).squeeze()

        st.subheader("📄 Selected Item")
        cL, cR = st.columns(2)
        with cL:
            st.markdown(f"**Description:** {sel.get('description','')}")
            st.markdown(f"**Chapter:** {sel.get('chapter','') or ''}")
            st.markdown(f"**Item No.:** {sel.get('item_no','') or ''}")
            st.markdown(f"**Page:** {int(sel.get('page_no', 0)) if pd.notna(sel.get('page_no', np.nan)) else ''}")
        with cR:
            st.markdown(f"**Spec No.:** {sel.get('spec_no','') or ''}")
            st.markdown(f"**Remarks:** {sel.get('remarks','') or ''}")

        st.markdown("**Rates**")
        rate_df = pd.DataFrame({
            "System": ["British", "Metric"],
            "Unit": [sel.get("unit_british",""), sel.get("unit_metric","")],
            "Labour": [sel.get("labour_british",""), sel.get("labour_metric","")],
            "Composite": [sel.get("composite_british",""), sel.get("composite_metric","")]
        })
        st.dataframe(rate_df, use_container_width=True)

        st.divider()
        st.subheader("➕ Add to BOQ")
        qty = st.number_input("Quantity", min_value=0.0, step=1.0, value=0.0)
        rate_choice = st.selectbox(
            "Use rate",
            ["Composite (British)", "Composite (Metric)", "Labour (British)", "Labour (Metric)"]
        )

        def to_float(x):
            try:
                return float(str(x).replace(",", ""))
            except Exception:
                return np.nan

        if rate_choice == "Composite (British)":
            unit = sel.get("unit_british", "")
            rate_val = to_float(sel.get("composite_british", np.nan))
        elif rate_choice == "Composite (Metric)":
            unit = sel.get("unit_metric", "")
            rate_val = to_float(sel.get("composite_metric", np.nan))
        elif rate_choice == "Labour (British)":
            unit = sel.get("unit_british", "")
            rate_val = to_float(sel.get("labour_british", np.nan))
        else:
            unit = sel.get("unit_metric", "")
            rate_val = to_float(sel.get("labour_metric", np.nan))

        amount = (qty or 0.0) * (rate_val if not np.isnan(rate_val) else 0.0)

        st.markdown(f"**Unit:** {unit}")
        st.markdown(f"**Rate:** {'' if np.isnan(rate_val) else rate_val}")
        st.markdown(f"**Amount:** {amount:,.2f}")

        if "boq" not in st.session_state:
            st.session_state.boq = []

        if st.button("Add line"):
            st.session_state.boq.append({
                "Description": sel.get("description",""),
                "Chapter": sel.get("chapter",""),
                "Item No.": sel.get("item_no",""),
                "Page": int(sel.get("page_no",0)) if pd.notna(sel.get("page_no", np.nan)) else "",
                "Unit": unit,
                "Rate": rate_val,
                "Qty": qty,
                "Amount": amount
            })
            st.success("Added to BOQ.")

    # BOQ table + download
    st.divider()
    st.subheader("📊 BOQ")
    boq_df = pd.DataFrame(st.session_state.get("boq", []))
    if boq_df.empty:
        st.info("BOQ is empty. Search an item and click **Add line**.")
    else:
        st.dataframe(boq_df, use_container_width=True)
        total = boq_df["Amount"].sum()
        st.markdown(f"**Total:** {total:,.2f}")

        out = BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            boq_df.to_excel(writer, index=False, sheet_name="BOQ")
        st.download_button(
            "⬇️ Download BOQ.xlsx",
            data=out.getvalue(),
            file_name="BOQ.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
