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
def read_asins_for_basket(basket_name: str) -> list[str]:
    """
    Read ASINs from the TXT in the repo (GitHub RAW).
    - Skips empty lines and lines starting with '#'
    - Deduplicates while preserving order
    """
    url = asin_txt_raw_url(basket_name)
    headers = {}
    token = st.secrets.get("GITHUB_TOKEN", None)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    r = requests.get(url, headers=headers, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Cannot read ASIN list ({r.status_code}): {url}")

    lines = [ln.strip() for ln in r.text.splitlines()]
    # filter comments/empties
    filtered = [ln for ln in lines if ln and not ln.startswith("#")]

    # deduplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for a in filtered:
        if a not in seen:
            seen.add(a)
            out.append(a)
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
