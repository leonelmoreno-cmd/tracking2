# pages/parameters.py
# -*- coding: utf-8 -*-

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
from fuzzywuzzy import fuzz, process  # si prefieres: from rapidfuzz import fuzz, process

# -----------------------------------
# Helpers
# -----------------------------------
def squish(s: pd.Series) -> pd.Series:
    """Trim + collapse inner whitespace."""
    return s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

def extract_asin(text: str) -> str | None:
    """Return first ASIN (B0 + 8 alfanum) found, else None."""
    if pd.isna(text):
        return None
    m = re.search(r"B0[A-Z0-9]{8}", str(text))
    return m.group(0) if m else None

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def calculate_base_metrics(data: pd.DataFrame, target_acos: float) -> pd.DataFrame:
    # Ensure numeric
    for c in ["Sales", "Orders", "Clicks"]:
        if c in data.columns:
            data[c] = to_num(data[c])
        else:
            data[c] = 0.0

    total_sales = data["Sales"].sum(skipna=True)
    total_orders = data["Orders"].sum(skipna=True)
    total_clicks = data["Clicks"].sum(skipna=True)

    aov = (total_sales / total_orders) if total_orders > 0 else 0.0
    target_cpa = target_acos * aov
    x = (total_clicks / total_orders) if total_orders > 0 else 0.0
    max_cpc = (target_cpa / x) if x > 0 else 0.0

    return pd.DataFrame({
        "Metric": ["Target ACoS", "AOV", "Target CPA", "Clicks per Order (x)", "Max CPC"],
        "Value":  [target_acos,     aov,    target_cpa,   x,                      max_cpc]
    })

def analyze_match_type(data: pd.DataFrame) -> pd.DataFrame:
    for c in ["Clicks", "Orders", "CPC", "ACOS"]:
        if c in data.columns:
            data[c] = to_num(data[c])
        else:
            data[c] = 0.0

    out_rows = []
    for mt in ["Broad", "Phrase", "Exact"]:
        mt_df = data.loc[data["Match Type"] == mt]
        if mt_df.empty:
            continue
        clicks = mt_df["Clicks"].sum(skipna=True)
        orders = mt_df["Orders"].sum(skipna=True)
        conv_rate = (orders / clicks) if clicks > 0 else 0.0
        avg_cpc = mt_df["CPC"].mean(skipna=True)
        max_cpc = mt_df["CPC"].max(skipna=True)
        avg_acos = mt_df["ACOS"].mean(skipna=True)
        max_acos = mt_df["ACOS"].max(skipna=True)

        out_rows.append({
            "Match Type": mt,
            "Clicks": clicks,
            "Orders": orders,
            "Conversion Rate": conv_rate,             # sin redondear
            "Average CPC": round(avg_cpc, 2) if pd.notna(avg_cpc) else np.nan,
            "Max CPC": round(max_cpc, 2) if pd.notna(max_cpc) else np.nan,
            "Average ACOS": round(avg_acos * 100, 1) if pd.notna(avg_acos) else np.nan,
            "Max ACOS": round(max_acos * 100, 1) if pd.notna(max_acos) else np.nan,
        })

    if not out_rows:
        return pd.DataFrame(columns=[
            "Match Type","Clicks","Orders","Conversion Rate",
            "Average CPC","Max CPC","Average ACOS","Max ACOS","% Clicks"
        ])

    df_out = pd.DataFrame(out_rows)
    total_clicks = df_out["Clicks"].sum(skipna=True)
    df_out["% Clicks"] = np.where(total_clicks > 0, np.round(df_out["Clicks"] / total_clicks * 100, 1), 0.0)
    return df_out

def analyze_placements(df_all: pd.DataFrame, df_asin: pd.DataFrame, target_acos: float, aov: float) -> pd.DataFrame:
    # Ensure numerics
    for c in ["Clicks", "Orders", "CPC"]:
        if c in df_all.columns:
            df_all[c] = to_num(df_all[c])
    if "Clicks" in df_asin.columns:
        df_asin["Clicks"] = to_num(df_asin["Clicks"])

    asin_list = df_asin["ASIN"].dropna().unique().tolist()
    asin_code = asin_list[0] if asin_list else None

    placements = ["Placement Top", "Placement Product Page", "Placement Rest Of Search"]
    placement_data = df_all[
        (df_all.get("Entity") == "Bidding Adjustment") &
        (df_all.get("Placement").isin(placements)) &
        (df_all.get("Campaign Name (Informational only)", "").astype(str).str.contains(asin_code if asin_code else "", na=False))
    ].copy()

    total_clicks_in_placements = placement_data["Clicks"].sum(skipna=True)
    out_rows = []

    for p in placements:
        p_df = placement_data.loc[placement_data["Placement"] == p]
        clicks = p_df["Clicks"].sum(skipna=True)
        orders = p_df["Orders"].sum(skipna=True)
        conv_rate = (orders / clicks) if clicks > 0 else 0.0

        if p_df.empty or p_df["CPC"].isna().all():
            max_cpc = np.nan
            avg_cpc = np.nan
        else:
            max_cpc = p_df["CPC"].max(skipna=True)
            avg_cpc = p_df["CPC"].mean(skipna=True)

        pct_clicks = (clicks / total_clicks_in_placements * 100) if total_clicks_in_placements > 0 else 0.0
        new_bid = target_acos * aov * conv_rate

        out_rows.append({
            "Placement": p,
            "Clicks": clicks,
            "Orders": orders,
            "Conversion Rate": conv_rate,                             # sin redondear
            "Max CPC": round(max_cpc, 2) if pd.notna(max_cpc) else np.nan,
            "% Clicks": round(pct_clicks, 1),
            "Average CPC": round(avg_cpc, 2) if pd.notna(avg_cpc) else np.nan,
            "New Bid (Max CPC)": round(new_bid, 2)
        })

    return pd.DataFrame(out_rows)


# -----------------------------------
# Streamlit Page
# -----------------------------------
def main():
    st.title("Parameters")
    st.caption("Upload the Sponsored Products Campaigns Excel, pick portfolio with fuzzy match, set Target ACoS, and export per-ASIN analysis.")

    # Inputs
    uploaded = st.file_uploader("Upload Excel (.xlsx) with a sheet named 'Sponsored Products Campaigns'", type=["xlsx"])
    user_portfolio = st.text_input("Portfolio (type to fuzzy match against 'Portfolio Name (Informational only)')", value="", placeholder="e.g., LGM Soil - JungleClick")
    default_target_acos = st.number_input("Default Target ACoS", min_value=0.0, max_value=1.0, value=0.15, step=0.01, format="%.2f")

    run = st.button("Run analysis", type="primary")

    st.divider()

    if not run:
        st.info("Upload the Excel, type a portfolio (optional), set Target ACoS, and click **Run analysis**.")
        return

    # Read Excel
    if uploaded is None:
        st.error("Please upload an Excel file first.")
        st.stop()

    try:
        xls = pd.ExcelFile(uploaded)
    except Exception as e:
        st.error(f"Could not open Excel: {e}")
        st.stop()

    sheet_names = xls.sheet_names
    st.write("Available sheets:", ", ".join(sheet_names))
    if "Sponsored Products Campaigns" not in sheet_names:
        st.error("Sheet 'Sponsored Products Campaigns' not found.")
        st.stop()

    try:
        df_raw = pd.read_excel(xls, sheet_name="Sponsored Products Campaigns")
    except Exception as e:
        st.error(f"Failed to read sheet: {e}")
        st.stop()

    st.success(f"Loaded {len(df_raw):,} rows.")

    # Initial cleaning
    df = df_raw.copy()
    df.columns = df.columns.map(lambda c: str(c).strip())
    if "Campaign Name (Informational only)" not in df.columns or "Portfolio Name (Informational only)" not in df.columns:
        st.error("Required columns not found: 'Campaign Name (Informational only)' and 'Portfolio Name (Informational only)'.")
        st.stop()

    df["Campaign Name (Informational only)"] = squish(df["Campaign Name (Informational only)"])
    df["Portfolio Name (Informational only)"] = squish(df["Portfolio Name (Informational only)"])

    st.write("Unique portfolios (sample):")
    st.write(pd.Series(df["Portfolio Name (Informational only)"].unique()).head(10))

    # Fuzzy portfolio filter (optional)
    if user_portfolio.strip():
        portfolios = df["Portfolio Name (Informational only)"].dropna().unique().tolist()
        best = process.extractOne(user_portfolio.strip(), portfolios, scorer=fuzz.token_sort_ratio)
        if best:
            chosen_portfolio = best[0]
            st.info(f"Portfolio selected (fuzzy): **{chosen_portfolio}**")
            df = df[df["Portfolio Name (Informational only)"] == chosen_portfolio]
        else:
            st.warning("No portfolio matched. Continuing without portfolio filter.")

    if df.empty:
        st.error("No rows after portfolio filtering. Please adjust your input.")
        st.stop()

    # Filter enabled states where columns exist
    enabled_states = {"enabled"}
    for col in ["State", "Campaign State (Informational only)", "Ad Group State (Informational only)"]:
        if col in df.columns:
            df = df[df[col].astype(str).str.lower().isin(enabled_states)]

    st.write(f"Rows after enabled-state filtering: {len(df):,}")

    # Keep only Keyword / Product Targeting
    if "Entity" in df.columns:
        df = df[df["Entity"].isin(["Keyword", "Product Targeting", "Bidding Adjustment"])]
    else:
        st.warning("Column 'Entity' not found; proceeding without entity filtering.")

    st.write(f"Rows after entity filtering: {len(df):,}")

    # Extract ASIN from Campaign Name
    df["ASIN"] = df["Campaign Name (Informational only)"].map(extract_asin)
    asin_detected = df["ASIN"].dropna().unique().tolist()
    if not asin_detected:
        st.warning("No ASIN detected in 'Campaign Name (Informational only)'. Check the pattern.")
    else:
        st.success(f"Detected ASINs: {', '.join(asin_detected)}")

    st.dataframe(df[["Campaign Name (Informational only)", "ASIN"]].head(10), use_container_width=True)

    # Build per-ASIN Excel in memory
    if not asin_detected:
        st.stop()

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for asin in asin_detected:
            asin_df = df[df["ASIN"] == asin].copy()

            # Base metrics
            base_df = calculate_base_metrics(asin_df.copy(), default_target_acos)
            aov_val = float(base_df.loc[base_df["Metric"] == "AOV", "Value"].values[0])

            # Match type analysis (only relevant rows)
            match_df = analyze_match_type(asin_df.copy())

            # Placement analysis uses full df_raw (for Bidding Adjustment rows) + asin subset
            placement_df = analyze_placements(df_raw.copy(), asin_df.copy(), default_target_acos, aov_val)

            # Write all three tables into a single sheet for this ASIN
            sheet = asin
            start_row = 0
            base_df.to_excel(writer, sheet_name=sheet, index=False, startrow=start_row)
            start_row += len(base_df) + 2
            match_df.to_excel(writer, sheet_name=sheet, index=False, startrow=start_row)
            start_row += len(match_df) + 2
            placement_df.to_excel(writer, sheet_name=sheet, index=False, startrow=start_row)

    output.seek(0)
    st.success("Analysis generated. You can download the Excel file below.")
    st.download_button(
        "Download per-ASIN analysis (Excel)",
        data=output.getvalue(),
        file_name="asin_analysis_output_python.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
