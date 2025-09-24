import pandas as pd
import requests
import numpy as np
import streamlit as st
from typing import Dict, List

# -------------------------------
# Repo constants
# -------------------------------
GITHUB_OWNER = "leonelmoreno-cmd"
GITHUB_REPO = "tracking2"
GITHUB_PATH = "data"
GITHUB_BRANCH = "main"

# -------------------------------
# Mapeo principal → subcategoría
# -------------------------------
COMPETITOR_TO_SUBCATEGORY_MAP = {
    "competitors_history - BL.csv": "sub-categories2/sub_BL.csv",
    "competitors_history - GS.csv": "sub-categories2/sub_GS.csv",
    "competitors_history - IC.csv": "sub-categories2/sub_IC.csv",
    "competitors_history - LGM.csv": "sub-categories2/sub_LGM.csv",
    "competitors_history - QC.csv": "sub-categories2/sub_QC.csv",
    "competitors_history - RIO.csv": "sub-categories2/sub_RIO.csv",
    "competitors_history - UR.csv": "sub-categories2/sub_UR.csv",
    "synthethic3.csv": "sub-categories2/sub_SYN.csv"
}

# -------------------------------
# Page config
# -------------------------------
def set_page_config():
    st.set_page_config(
        page_title="Competitor Price Monitoring - JC",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
    )
    st.logo(
        "https://raw.githubusercontent.com/leonelmoreno-cmd/tracking2/main/assets/logo.png",
        size="large",
        link="https://github.com/leonelmoreno-cmd/tracking2"
    )

# -------------------------------
# Fetch CSV data
# -------------------------------
def fetch_data(url: str) -> pd.DataFrame:
    """Fetch CSV data from a URL and return it as a pandas DataFrame."""
    return pd.read_csv(url)

# -------------------------------
# Prepare data
# -------------------------------
def prepare_data(df: pd.DataFrame, basket_name: str = None) -> pd.DataFrame:
    """
    Prepare and clean data for analysis.
    Adds week number, discount label, price change,
    and merges brand info if available.
    Always returns a clean 'brand' column.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["asin", "date"])
    df["week_number"] = df["date"].dt.isocalendar().week
    df = df.sort_values(by=["asin", "week_number"])
    df["discount"] = df.apply(
        lambda row: "Discounted" if pd.notna(row.get("product_original_price")) else "No Discount",
        axis=1
    )
    df["price_change"] = df.groupby("asin")["product_price"].pct_change() * 100

    # --- 🔑 Merge brand info ---
    if basket_name and basket_name in COMPETITOR_TO_SUBCATEGORY_MAP:
        sub_file = COMPETITOR_TO_SUBCATEGORY_MAP[basket_name]
        sub_url = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{GITHUB_PATH}/{sub_file}"
        try:
            sub_df = pd.read_csv(sub_url, usecols=["asin", "brand"])
            # merge evitando sufijos feos como brand_x
            df = df.merge(sub_df, on="asin", how="left", suffixes=("", "_drop"))
            # eliminar columnas duplicadas con sufijo _drop
            df = df[[c for c in df.columns if not c.endswith("_drop")]]
        except Exception as e:
            print(f"⚠️ Error loading subcategory file {sub_file}: {e}")
            df["brand"] = "Unknown"
    else:
        df["brand"] = "Unknown"

    return df

# -------------------------------
# GitHub URL helper
# -------------------------------
def _raw_url_for(owner: str, repo: str, branch: str, path: str, fname: str) -> str:
    """Construct a raw GitHub URL for a given file."""
    subcategory_file = COMPETITOR_TO_SUBCATEGORY_MAP.get(fname)
    if subcategory_file:
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}/{subcategory_file}"
    else:
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}/{fname}"

    print(f"Generated URL: {url}")
    return url

# -------------------------------
# List CSV files in repo
# -------------------------------
@st.cache_data(show_spinner=False)
def list_repo_csvs(owner: str, repo: str, path: str, branch: str = "main") -> List[dict]:
    """Return a list of main CSV files (not sub-categories) from a GitHub repository."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    headers = {"Accept": "application/vnd.github+json"}
    token = st.secrets.get("GITHUB_TOKEN", None)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    items = resp.json()

    main_files = set(COMPETITOR_TO_SUBCATEGORY_MAP.keys())

    csvs = [
        {
            "name": it["name"],
            "download_url": it["download_url"],
            "path": it.get("path", "")
        }
        for it in items
        if it.get("type") == "file"
        and it.get("name", "").lower().endswith(".csv")
        and it["name"] in main_files
    ]
    csvs.sort(key=lambda x: x["name"])
    return csvs
