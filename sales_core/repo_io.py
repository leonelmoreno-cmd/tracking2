# sales_core/repo_io.py
from __future__ import annotations
import os
import io
import requests
import streamlit as st
import pandas as pd
from .config import (
    GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH,
    ASINS_DIR, SALES_DIR,
)

# ------------------------------------------------------------
# GitHub raw URL helpers
# ------------------------------------------------------------
def _raw_url(path: str) -> str:
    """Build a GitHub RAW URL for a repository file."""
    return f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{path}"

def asin_txt_raw_url(basket_name: str) -> str:
    """TXT with ASINs lives at Asins/JS/<basket_basename>.txt."""
    base = os.path.splitext(basket_name)[0]
    rel = f"{ASINS_DIR}/{base}.txt"
    return _raw_url(rel)

def weekly_csv_raw_url(basket_name: str) -> str:
    """Weekly CSV (produced by Actions) at sales_core/sales/<basket_basename>.csv."""
    base = os.path.splitext(basket_name)[0]
    rel = f"{SALES_DIR}/{base}.csv"
    return _raw_url(rel)

def weekly_csv_local_path(basket_name: str) -> str:
    """Local path fallback for weekly CSV."""
    base = os.path.splitext(basket_name)[0]
    return os.path.join(SALES_DIR, f"{base}.csv")


# ------------------------------------------------------------
# Read ASIN list for a basket
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_asins_for_basket(basket_name: str) -> list[tuple[str,str]]:
    url = asin_txt_raw_url(basket_name)
    r = requests.get(url, headers=headers, timeout=20)
    ...
    filtered = [ln for ln in lines if ln and not ln.startswith("#")]
    out = []
    for ln in filtered:
        parts = [p.strip() for p in ln.split(",", 1)]
        if len(parts) == 2:
            asin, brand = parts
        else:
            asin = parts[0]
            brand = "Unknown"
        out.append((asin, brand))
    return out


# ------------------------------------------------------------
# Write weekly CSV (used by pipeline in Actions)
# ------------------------------------------------------------
def write_weekly_csv_local(df: pd.DataFrame, basket_name: str) -> str:
    """
    Overwrite the weekly CSV for the given basket (no history).
    Used by the ETL pipeline (GitHub Actions).
    """
    os.makedirs(SALES_DIR, exist_ok=True)
    fp = weekly_csv_local_path(basket_name)
    df.to_csv(fp, index=False)
    return fp


# ------------------------------------------------------------
# Read weekly CSV (used by Streamlit UI)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_weekly_csv_remote(basket_name: str) -> pd.DataFrame:
    """
    Read the weekly CSV for a basket directly from GitHub RAW.
    This is recommended for the Streamlit UI so it always shows
    the latest file committed by the GitHub Action.
    """
    url = weekly_csv_raw_url(basket_name)
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        raise FileNotFoundError(f"Weekly CSV not found in repo ({r.status_code}): {url}")

    df = pd.read_csv(io.StringIO(r.text))
    if "week_end" in df.columns:
        df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
    return df

def read_weekly_csv_local(basket_name: str) -> pd.DataFrame:
    """
    Local fallback if you have the CSV on disk (e.g., running ETL locally).
    """
    fp = weekly_csv_local_path(basket_name)
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Weekly CSV not found locally: {fp}")
    df = pd.read_csv(fp)
    if "week_end" in df.columns:
        df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
    return df
