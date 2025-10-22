from __future__ import annotations
import os
import requests
import streamlit as st
import pandas as pd
from .config import (
    GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH,
    ASINS_DIR, SALES_DIR,
)

def _raw_url(path: str) -> str:
    return f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{path}"

def asin_txt_raw_url(basket_name: str) -> str:
    base = os.path.splitext(basket_name)[0]
    rel = f"{ASINS_DIR}/{base}.txt"
    return _raw_url(rel)

@st.cache_data(show_spinner=False)
def read_asins_for_basket(basket_name: str) -> list[str]:
    url = asin_txt_raw_url(basket_name)
    headers = {}
    token = st.secrets.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(url, headers=headers, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Cannot read ASIN list ({r.status_code}): {url}")
    raw = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
    seen, out = set(), []
    for a in raw:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out

def write_weekly_csv_local(df: pd.DataFrame, basket_name: str) -> str:
    os.makedirs(SALES_DIR, exist_ok=True)
    base = os.path.splitext(basket_name)[0]
    fp = os.path.join(SALES_DIR, f"{base}.csv")
    df.to_csv(fp, index=False)
    return fp
