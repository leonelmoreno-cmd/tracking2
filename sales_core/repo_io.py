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
    BASKET_NAME_ALIASES,   # <-- NUEVO
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _raw_url(path: str) -> str:
    return f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{path}"

def _alias_base(basket_name: str) -> str:
    """Devuelve el 'base' sin extensión, aplicando alias si existe."""
    name = basket_name.strip()
    if name in BASKET_NAME_ALIASES:
        return BASKET_NAME_ALIASES[name]
    return os.path.splitext(name)[0]

# ------------------------------------------------------------
# GitHub raw URL builders
# ------------------------------------------------------------
def asin_txt_raw_url(basket_name: str) -> str:
    """
    TXT with ASINs at Asins/JS/<alias_base>.txt
    (ej. UI: 'competitors_history - IC.csv' -> alias 'asins_IC' -> Asins/JS/asins_IC.txt)
    """
    base = _alias_base(basket_name)
    rel = f"{ASINS_DIR}/{base}.txt"
    return _raw_url(rel)

def weekly_csv_raw_url(basket_name: str) -> str:
    """
    Weekly CSV at sales_core/sales/<alias_base>.csv
    (ej. 'competitors_history - IC.csv' -> 'asins_IC.csv')
    """
    base = _alias_base(basket_name)
    rel = f"{SALES_DIR}/{base}.csv"
    return _raw_url(rel)

def weekly_csv_local_path(basket_name: str) -> str:
    base = _alias_base(basket_name)
    return os.path.join(SALES_DIR, f"{base}.csv")

# ------------------------------------------------------------
# Read ASIN list for a basket
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_asins_for_basket(basket_name: str) -> list[tuple[str, str]]:
    url = asin_txt_raw_url(basket_name)
    headers = {}
    token = st.secrets.get("GITHUB_TOKEN", None)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(url, headers=headers, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Cannot read ASIN list ({r.status_code}): {url}")
    lines = [ln.strip() for ln in r.text.splitlines() if ln and not ln.startswith("#")]
    out, seen = [], set()
    for ln in lines:
        asin, brand = (p.strip() for p in (ln.split(",", 1) + ["Unknown"])[:2])
        if asin and asin not in seen:
            seen.add(asin)
            out.append((asin, brand or "Unknown"))
    return out

# ------------------------------------------------------------
# Write weekly CSV (used by pipeline in Actions)
# ------------------------------------------------------------
def write_weekly_csv_local(df: pd.DataFrame, basket_name: str) -> str:
    os.makedirs(SALES_DIR, exist_ok=True)
    fp = weekly_csv_local_path(basket_name)
    df.to_csv(fp, index=False)
    return fp

# ------------------------------------------------------------
# Read weekly CSV (used by Streamlit UI)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_weekly_csv_remote(basket_name: str) -> pd.DataFrame:
    url = weekly_csv_raw_url(basket_name)
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        raise FileNotFoundError(f"Weekly CSV not found in repo ({r.status_code}): {url}")
    df = pd.read_csv(io.StringIO(r.text))
    if "week_end" in df.columns:
        df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
    return df

def read_weekly_csv_local(basket_name: str) -> pd.DataFrame:
    fp = weekly_csv_local_path(basket_name)
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Weekly CSV not found locally: {fp}")
    df = pd.read_csv(fp)
    if "week_end" in df.columns:
        df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
    return df
