from __future__ import annotations
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def read_competitor_brand_map(download_url: str) -> pd.DataFrame:
    df = pd.read_csv(download_url)
    df["asin"] = df["asin"].astype(str).str.strip()
    if "brand" not in df.columns:
        df["brand"] = "Unknown"
    df["brand"] = df["brand"].fillna("Unknown").astype(str)
    return df[["asin", "brand"]].drop_duplicates()

def attach_brand(daily_df: pd.DataFrame, brand_map_df: pd.DataFrame) -> pd.DataFrame:
    out = daily_df.merge(brand_map_df, on="asin", how="left")
    out["brand"] = out["brand"].fillna("Unknown").astype(str)
    return out
