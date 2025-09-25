import re
import pandas as pd
import numpy as np
import streamlit as st
from itertools import chain
from typing import List, Tuple

# -------------------------------
# Helpers
# -------------------------------
ASIN_REGEX = re.compile(r"B0\w{8}", re.IGNORECASE)

DEFAULT_STOPWORDS = set("""
a an and are as at be by for from has have in is it of on or that the to was what when where which who with get
""".split())

def _clean_currency_series(s: pd.Series) -> pd.Series:
    # "$1,234.56" -> 1234.56
    return (
        s.astype(str)
         .str.replace(r"[\$,]", "", regex=True)
         .replace({"": np.nan, "nan": np.nan})
         .astype(float)
    )

def _coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _normalize_text(s: pd.Series) -> pd.Series:
    # Lowercase, remove punctuation/digits, condense spaces
    return (
        s.astype(str).str.lower()
         .str.replace(r"[^\w\s]", " ", regex=True)
         .str.replace(r"\d+", " ", regex=True)
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
    )

def extract_asins_from_campaigns(campaign_series: pd.Series) -> List[str]:
    asins = set(chain.from_iterable(campaign_series.dropna().apply(lambda x: ASIN_REGEX.findall(str(x)))))
    return sorted(asins)

def _row_ngrams(tokens: List[str], n_values: Tuple[int, ...]) -> List[str]:
    out = []
    L = len(tokens)
    for n in n_values:
        if n <= 0: 
            continue
        for i in range(max(0, L - n + 1)):
            ngram = " ".join(tokens[i:i+n])
            out.append(ngram)
    return out

# -------------------------------
# Core: build ngram table
# -------------------------------
@st.cache_data(show_spinner=False)
def build_ngram_table(
    df: pd.DataFrame,
    asin: str = None,
    n_values: Tuple[int, ...] = (1, 2, 3),
    stopwords: set = DEFAULT_STOPWORDS,
    min_char_len: int = 2,
    exclude_contains: List[str] = None,
) -> pd.DataFrame:
    """
    Build aggregated n-gram table with metrics and KPIs.
    If an ASIN is provided, filter rows where 'Campaign Name' contains that ASIN.
    """
    if exclude_contains is None:
        exclude_contains = []

    # --- 1) Filter by ASIN in 'Campaign Name'
    if "Campaign Name" not in df.columns:
        raise ValueError("The column 'Campaign Name' is required in the uploaded file.")

    if asin is None:
        df_asin = df.copy()
    else:
        df_asin = df[df["Campaign Name"].astype(str).str.contains(str(asin), na=False)]

    if df_asin.empty:
        return pd.DataFrame(columns=[
            "ngram","Impressions","Clicks","Orders","Spend","Sales",
            "CTR","CVR","CPC","RPC","ACOS","n"
        ])

    # --- 2) Normalize numeric columns
    if "Spend" in df_asin.columns:
        df_asin["Spend"] = _clean_currency_series(df_asin["Spend"])
    if "7 Day Total Sales" in df_asin.columns:
        df_asin["7 Day Total Sales"] = _clean_currency_series(df_asin["7 Day Total Sales"])

    for col in ["Impressions", "Clicks", "Orders"]:
        if col in df_asin.columns:
            df_asin[col] = _coerce_numeric(df_asin[col])

    # --- 3) Normalize 'Customer Search Term'
    if "Customer Search Term" not in df_asin.columns:
        raise ValueError("The column 'Customer Search Term' is required in the uploaded file.")

    df_asin["clean_search_term"] = _normalize_text(df_asin["Customer Search Term"])
    df_asin["tokens"] = df_asin["clean_search_term"].str.split()

    # --- 4) Generate ngrams
    allowed_n = tuple(sorted(set(int(n) for n in n_values if n in (1, 2, 3))))
    if not allowed_n:
        allowed_n = (1,)

    df_asin["ngram_list"] = df_asin["tokens"].apply(lambda toks: _row_ngrams(toks, allowed_n))
    exploded = df_asin.explode("ngram_list", ignore_index=True)
    exploded = exploded.rename(columns={"ngram_list": "ngram"})
    exploded["n"] = exploded["ngram"].str.count(" ") + 1

    # --- 5) Filter ngrams
    def _valid_ng(ng: str) -> bool:
        if not isinstance(ng, str) or len(ng) < min_char_len:
            return False
        words = ng.split()
        if any(w in stopwords for w in words):
            return False
        if any(ex.lower() in ng for ex in exclude_contains):
            return False
        return True

    exploded = exploded[exploded["ngram"].apply(_valid_ng)]
    if exploded.empty:
        return pd.DataFrame(columns=[
            "ngram","Impressions","Clicks","Orders","Spend","Sales",
            "CTR","CVR","CPC","RPC","ACOS","n"
        ])

    # --- 6) Aggregate metrics
    grouped = exploded.groupby(["ngram", "n"], as_index=False).agg(
        Impressions=("Impressions", "sum"),
        Clicks=("Clicks", "sum"),
        Orders=("Orders", "sum"),
        Spend=("Spend", "sum"),
        Sales=("7 Day Total Sales", "sum"),
    )

    # --- 7) KPIs
    grouped["CTR"] = np.where(grouped["Impressions"] > 0, grouped["Clicks"] / grouped["Impressions"] * 100, 0.0)
    grouped["CVR"] = np.where(grouped["Clicks"] > 0, grouped["Orders"] / grouped["Clicks"] * 100, 0.0)
    grouped["CPC"] = np.where(grouped["Clicks"] > 0, grouped["Spend"] / grouped["Clicks"], 0.0)
    grouped["RPC"] = np.where(grouped["Clicks"] > 0, grouped["Sales"] / grouped["Clicks"], 0.0)
    grouped["ACOS"] = np.where(grouped["Sales"] > 0, grouped["Spend"] / grouped["Sales"] * 100, np.nan)

    # Default sort
    grouped = grouped.sort_values(["Orders", "CVR", "RPC"], ascending=[False, False, False]).reset_index(drop=True)
    return grouped

# -------------------------------
# Filtering helper
# -------------------------------
def apply_metric_filters(
    df: pd.DataFrame,
    impressions_min: int,
    clicks_min: int,
    orders_min: int,
    cvr_range: Tuple[float, float],
    rpc_range: Tuple[float, float],
    cpc_range: Tuple[float, float],
    acos_range: Tuple[float, float],
    n_values: Tuple[int, ...],
) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out = out[out["Impressions"] >= impressions_min]
    out = out[out["Clicks"] >= clicks_min]
    out = out[out["Orders"] >= orders_min]
    out = out[(out["CVR"] >= cvr_range[0]) & (out["CVR"] <= cvr_range[1])]
    out = out[(out["RPC"] >= rpc_range[0]) & (out["RPC"] <= rpc_range[1])]
    out = out[(out["CPC"] >= cpc_range[0]) & (out["CPC"] <= cpc_range[1])]
    out = out[(out["ACOS"].fillna(0) >= acos_range[0]) & (out["ACOS"].fillna(0) <= acos_range[1])]
    out = out[out["n"].isin(n_values)]
    return out
