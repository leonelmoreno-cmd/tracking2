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
# Page config
# -------------------------------
def set_page_config():
    st.set_page_config(
        page_title="Competitor Price Monitoring - JC",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
    )
     st.logo(
        "assets/logo.png",
        size="large",
        link="https://github.com/leonelmoreno-cmd/tracking2",
        icon_image=None
    )
# -------------------------------
# Fetch CSV data
# -------------------------------
def fetch_data(url: str) -> pd.DataFrame:
    """
    Fetch CSV data from a URL and return it as a pandas DataFrame.
    """
    return pd.read_csv(url)

# -------------------------------
# Prepare data
# -------------------------------
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare and clean data for analysis.
    Adds week number, discount label, and price change.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["week_number"] = df["date"].dt.isocalendar().week
    df = df.sort_values(by=["asin", "week_number"])
    df["discount"] = df.apply(
        lambda row: "Discounted" if pd.notna(row.get("product_original_price")) else "No Discount",
        axis=1
    )
    df["price_change"] = df.groupby("asin")["product_price"].pct_change() * 100
    return df

# -------------------------------
# GitHub URL helper
# -------------------------------
def _raw_url_for(owner: str, repo: str, branch: str, path: str, fname: str) -> str:
    """
    Construct a raw GitHub URL for a given file.
    Maps certain competitor files to sub-categories.
    """
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
    """
    Return a list of main CSV files (not sub-categories) from a GitHub repository.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    headers = {"Accept": "application/vnd.github+json"}
    token = st.secrets.get("GITHUB_TOKEN", None)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    items = resp.json()

    main_files = {
        "competitors_history - BL.csv",
        "competitors_history - GS.csv",
        "competitors_history - IC.csv",
        "competitors_history - LGM.csv",
        "competitors_history - QC.csv",
        "competitors_history - RIO.csv",
        "competitors_history - UR.csv",
        "synthethic3.csv"
    }

    csvs = [
        {
            "name": it["name"],
            "download_url": it["download_url"],
            "path": it.get("path", "")
        }
        for it in items
        if it.get("type") == "file" and it.get("name", "").lower().endswith(".csv") and it["name"] in main_files
    ]
    csvs.sort(key=lambda x: x["name"])
    return csvs
