# ... keep the imports and parsing code exactly as before ...

import re

# ---------- NEW: feature extraction helpers ----------
CLASS_RX = re.compile(r'\bclass[-\s]*([ivx]+)\b', re.IGNORECASE)
MM_RX    = re.compile(r'(\d{2,4})\s*mm\b', re.IGNORECASE)
PSI_RX   = re.compile(r'(\d{3,5})\s*psi\b', re.IGNORECASE)
TYPE_RX  = re.compile(r'\b(R\.?C\.?C\.?|concrete|PVC|HDPE|DI)\b', re.IGNORECASE)

def extract_features(desc: str) -> dict:
    """
    Pull structured attributes from the item description:
    - base: description with sizes/classes removed (for grouping 'family')
    - cls:  roman class like II, III, IV
    - mm:   diameter in mm (int) if present
    - psi:  pressure rating if present (int)
    - typ:  pipe/material hint like RCC, PVC, etc.
    """
    d = desc or ""
    cls = None
    mm  = None
    psi = None
    typ = None

    m = CLASS_RX.search(d)
    if m:
        cls = m.group(1).upper()

    m = MM_RX.search(d)
    if m:
        try: mm = int(m.group(1))
        except: mm = None

    m = PSI_RX.search(d)
    if m:
        try: psi = int(m.group(1))
        except: psi = None

    m = TYPE_RX.search(d)
    if m:
        typ = m.group(1).upper().replace('.', '')

    # Build a "base family" string: remove class + mm + psi tokens for grouping
    base = d
    base = CLASS_RX.sub('', base)
    base = MM_RX.sub('', base)
    base = PSI_RX.sub('', base)
    base = re.sub(r'\s+', ' ', base).strip()

    return {"base": base, "cls": cls, "mm": mm, "psi": psi, "typ": typ}

def enrich(df):
    feats = df["description"].astype(str).apply(extract_features).apply(pd.Series)
    for c in feats.columns:
        df[c] = feats[c]
    # a compact key to ensure exact selection when needed
    df["variant_key"] = (
        (df["cls"].fillna('')) + "|" +
        df["psi"].fillna(0).astype(str) + "|" +
        df["mm"].fillna(0).astype(int).astype(str)
    )
    return df

# --------------------------- (inside your main flow, after df is parsed) ---------------------------
if df is not None and not df.empty:
    df = enrich(df)  # <--- NEW: add structured columns
    st.success(f"Parsed {len(df):,} rows from PDF.")

    # --- Filters as before ---
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

    # --- Search a 'family' with fuzzy ---
    st.subheader("🔎 Search")
    query = st.text_input("Type item keywords (e.g., 'RCC pipe laying Class-III 610 mm')", "")
    top_n = st.slider("Suggestions to show", 5, 20, 8)

    def get_suggestions(q, pool, n=8):
        if not q or len(q.strip()) < 2:
            return []
        from rapidfuzz import process, fuzz
        return process.extract(q, pool, scorer=fuzz.WRatio, limit=n)

    # We search over BASE family strings to group all sub-items under one umbrella
    families = dff["base"].fillna(dff["description"]).astype(str).tolist()
    suggestions = get_suggestions(query, families, n=top_n)

    base_pick = None
    if suggestions:
        # de-duplicate labels that repeat
        uniq = []
        for txt, score, idx in suggestions:
            if txt not in (x[0] for x in uniq):
                uniq.append((txt, score, idx))
        labels = [f"{txt}  —  (score {int(score)})" for (txt, score, idx) in uniq]
        idx = st.radio("Closest base matches (pick one)", list(range(len(labels))), format_func=lambda i: labels[i], index=0)
        base_pick = uniq[idx][0]

    # --- Guided picker for sub-options (class / psi / mm) ---
    selected_row = None
    if base_pick:
        fam_df = dff[dff["base"] == base_pick].copy()
        st.markdown("#### Available variants in this family")

        # Show quick summary table (read-only) for user
        show_cols = ["description", "cls", "psi", "mm", "unit_british", "labour_british", "composite_british", "unit_metric", "labour_metric", "composite_metric", "page_no"]
        st.dataframe(fam_df[show_cols].sort_values(["cls","psi","mm"], na_position="last"), use_container_width=True, height=320)

        # Build filter widgets from available values
        cA, cB, cC, cD = st.columns(4)
        with cA:
            typ_opt = sorted([x for x in fam_df["typ"].dropna().unique().tolist()])
            typ_val = st.selectbox("Type", ["(any)"] + typ_opt, index=0)
        with cB:
            cls_opt = sorted([x for x in fam_df["cls"].dropna().unique().tolist()])
            cls_val = st.selectbox("Class", ["(any)"] + cls_opt, index=0)
        with cC:
            psi_opt = sorted([int(x) for x in fam_df["psi"].dropna().unique().tolist()])
            psi_val = st.selectbox("PSI", ["(any)"] + [str(x) for x in psi_opt], index=0)
        with cD:
            mm_opt = sorted([int(x) for x in fam_df["mm"].dropna().unique().tolist()])
            mm_val = st.selectbox("Diameter (mm)", ["(any)"] + [str(x) for x in mm_opt], index=0)

        # Apply filters progressively
        f = fam_df.copy()
        if typ_val != "(any)":
            f = f[f["typ"] == typ_val]
        if cls_val != "(any)":
            f = f[f["cls"] == cls_val]
        if psi_val != "(any)":
            f = f[f["psi"] == int(psi_val)]
        if mm_val != "(any)":
            f = f[f["mm"] == int(mm_val)]

        if len(f) == 0:
            st.warning("No row matches these filters. Loosen one of them.")
        elif len(f) > 1:
            st.info("Multiple rows match—pick the exact description below:")
            row_labels = [f"{r['description']} (Pg {int(r['page_no'])})" for _, r in f.iterrows()]
            pick_i = st.selectbox("Exact sub-item", list(range(len(row_labels))), format_func=lambda i: row_labels[i])
            selected_row = f.iloc[pick_i]
        else:
            selected_row = f.iloc[0]

    # --- When a single row is selected, show rates + BOQ ---
    if selected_row is not None:
        sel = selected_row
        st.subheader("📄 Selected Sub-Item")
        cL, cR = st.columns(2)
        with cL:
            st.markdown(f"**Description:** {sel.get('description','')}")
            st.markdown(f"**Chapter:** {sel.get('chapter','') or ''}")
            st.markdown(f"**Item No.:** {sel.get('item_no','') or ''}")
            st.markdown(f"**Page:** {int(sel.get('page_no', 0)) if pd.notna(sel.get('page_no', np.nan)) else ''}")
        with cR:
            st.markdown(f"**Class:** {sel.get('cls','') or ''}   **PSI:** {str(sel.get('psi','') or '')}   **Diameter:** {str(sel.get('mm','') or '')} mm")
            st.markdown(f"**Spec No.:** {sel.get('spec_no','') or ''}")
            st.markdown(f"**Remarks:** {sel.get('remarks','') or ''}")

        st.markdown("**Rates** (auto-parsed)")
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
        rate_choice = st.selectbox("Use rate", ["Composite (British)", "Composite (Metric)", "Labour (British)", "Labour (Metric)"])

        def to_float(x):
            try: return float(str(x).replace(",", ""))
            except: return np.nan

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

        if "boq" not in st.session_state: st.session_state.boq = []
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

    # --- BOQ table + download (unchanged) ---
    st.divider()
    st.subheader("📊 BOQ")
    boq_df = pd.DataFrame(st.session_state.get("boq", []))
    if boq_df.empty:
        st.info("BOQ is empty. Use the guided picker above.")
    else:
        st.dataframe(boq_df, use_container_width=True)
        total = boq_df["Amount"].sum()
        st.markdown(f"**Total:** {total:,.2f}")
        out = BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            boq_df.to_excel(writer, index=False, sheet_name="BOQ")
        st.download_button("⬇️ Download BOQ.xlsx",
                           data=out.getvalue(),
                           file_name="BOQ.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
