import re
import difflib
import pandas as pd
import numpy as np
import streamlit as st
from itertools import chain
from typing import List, Tuple, Dict, Optional

# -------------------------------
# Regex for ASIN extraction
# -------------------------------
ASIN_REGEX = re.compile(r"B0\w{8}", re.IGNORECASE)

# -------------------------------
# Defaults
# -------------------------------
DEFAULT_STOPWORDS = set("""
a an and are as at be by for from has have in is it of on or that the to was what when where which who with get
""".split())

# -------------------------------
# Column resolution helpers
# -------------------------------

def _norm(s: str) -> str:
    """Normalize a column name: lowercase, remove non-alnum, collapse spaces."""
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

# Canonical columns we want to operate with
CANONICAL_TARGETS: Dict[str, List[str]] = {
    "Campaign Name": [
        "campaign name", "campaign", "campaign_name"
    ],
    "Customer Search Term": [
        "customer search term", "search term", "search query", "customer term", "query"
    ],
    "Impressions": [
        "impressions", "impr"
    ],
    "Clicks": [
        "clicks", "click"
    ],
    "Orders": [
        "orders", "purchases", "conversions", "7 day total orders", "7-day total orders"
    ],
    "Spend": [
        "spend", "cost", "ad spend", "amount spent", "total spend"
    ],
    "Sales": [
        "7 day total sales", "7-day total sales", "7 day sales", "sales 7 day",
        "total sales (7 day)", "revenue", "sales"
    ],
}

def _build_normalized_map(columns: List[str]) -> Dict[str, str]:
    """
    Build a map {normalized_col_name: original_col_name}
    so we can find originals after normalization.
    """
    out = {}
    for c in columns:
        out[_norm(c)] = c
    return out

def _resolve_one(
    normalized_map: Dict[str, str],
    candidates: List[str],
    cutoff: float = 0.8
) -> Optional[str]:
    """
    Attempt to resolve a single canonical column by:
    1) exact normalized match,
    2) fuzzy match via difflib on normalized keys.
    Returns the original column name (or None if not found).
    """
    keys = list(normalized_map.keys())

    # Try exact normalized matches first
    for cand in candidates:
        cand_n = _norm(cand)
        if cand_n in normalized_map:
            return normalized_map[cand_n]

    # Fuzzy match against the first candidate's normalized form
    # (works well if CSV has small deviations)
    main_key = _norm(candidates[0])
    best = difflib.get_close_matches(main_key, keys, n=1, cutoff=cutoff)
    if best:
        return normalized_map[best[0]]

    return None

def resolve_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Resolve actual column names from a DataFrame to our canonical targets.
    Returns: {canonical_name: actual_name or None}
    """
    norm_map = _build_normalized_map(list(df.columns))
    resolved = {}
    for canon, cand_list in CANONICAL_TARGETS.items():
        resolved[canon] = _resolve_one(norm_map, cand_list)
    return resolved

def _ensure_column(df: pd.DataFrame, actual: Optional[str], canonical: str, default=0.0) -> None:
    """
    Ensure df[canonical] exists. If 'actual' is provided, copy its values to canonical;
    otherwise create canonical filled with default.
    """
    if actual and actual in df.columns:
        df[canonical] = df[actual]
    else:
        # If not present, create a fallback column with default values
        df[canonical] = default

# -------------------------------
# Basic cleaners
# -------------------------------
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
    # lower, remove punctuation/digits, condense spaces
    return (
        s.astype(str).str.lower()
         .str.replace(r"[^\w\s]", " ", regex=True)
         .str.replace(r"\d+", " ", regex=True)
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
    )

# -------------------------------
# Public helpers
# -------------------------------
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
# Core: build n-gram table
# -------------------------------
@st.cache_data(show_spinner=False)
def build_ngram_table(
    df: pd.DataFrame,
    asin: Optional[str],
    n_values: Tuple[int, ...] = (1, 2, 3),
    stopwords: set = DEFAULT_STOPWORDS,
    min_char_len: int = 2,
    exclude_contains: List[str] = None,
) -> pd.DataFrame:
    """
    Return an aggregated n-gram table with metrics + KPIs.
    Filters the DF by ASIN inside 'Campaign Name' (if asin provided) before tokenizing.
    Column names are resolved robustly (case/spacing/fuzzy).
    """
    if exclude_contains is None:
        exclude_contains = []

    df_local = df.copy()

    # --- Resolve/ensure canonical columns in df_local
    colmap = resolve_columns(df_local)
    _ensure_column(df_local, colmap["Campaign Name"], "Campaign Name", default="")
    _ensure_column(df_local, colmap["Customer Search Term"], "Customer Search Term", default="")
    _ensure_column(df_local, colmap["Impressions"], "Impressions", default=0.0)
    _ensure_column(df_local, colmap["Clicks"], "Clicks", default=0.0)
    _ensure_column(df_local, colmap["Orders"], "Orders", default=0.0)
    _ensure_column(df_local, colmap["Spend"], "Spend", default=0.0)
    _ensure_column(df_local, colmap["Sales"], "Sales", default=0.0)

    # --- Filter by ASIN (if provided)
    if asin:
        # literal match (escaped)
        pattern = re.escape(asin)
        mask = df_local["Campaign Name"].astype(str).str.contains(pattern, na=False, regex=True)
        df_asin = df_local.loc[mask].copy()
    else:
        df_asin = df_local.copy()

    if df_asin.empty:
        return pd.DataFrame(columns=[
            "ngram","Impressions","Clicks","Orders","Spend","Sales",
            "CTR","CVR","CPC","RPC","ACOS","n"
        ])

    # --- Numeric normalization
    df_asin["Spend"] = _clean_currency_series(df_asin["Spend"])
    df_asin["Sales"] = _clean_currency_series(df_asin["Sales"])

    for col in ["Impressions", "Clicks", "Orders"]:
        df_asin[col] = _coerce_numeric(df_asin[col])

    # --- Text normalization (Customer Search Term)
    if "Customer Search Term" not in df_asin.columns:
        raise ValueError("Column 'Customer Search Term' is required in the uploaded file.")
    df_asin["clean_search_term"] = _normalize_text(df_asin["Customer Search Term"])
    df_asin["tokens"] = df_asin["clean_search_term"].str.split()

    # --- Generate n-grams
    allowed_n = tuple(sorted(set(int(n) for n in n_values if n in (1, 2, 3))))
    if not allowed_n:
        allowed_n = (1,)

    df_asin["ngram_list"] = df_asin["tokens"].apply(lambda toks: _row_ngrams(toks, allowed_n))
    exploded = df_asin.explode("ngram_list", ignore_index=True)
    exploded = exploded.rename(columns={"ngram_list": "ngram"})
    exploded["n"] = exploded["ngram"].str.count(" ") + 1

    # --- Filter n-grams (stopwords, length, exclusions)
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

    # --- Aggregate metrics per n-gram
    grouped = exploded.groupby(["ngram", "n"], as_index=False).agg(
        Impressions=("Impressions", "sum"),
        Clicks=("Clicks", "sum"),
        Orders=("Orders", "sum"),
        Spend=("Spend", "sum"),
        Sales=("Sales", "sum"),
    )

    # --- KPIs
    grouped["CTR"] = np.where(grouped["Impressions"] > 0, grouped["Clicks"] / grouped["Impressions"] * 100, 0.0)
    grouped["CVR"] = np.where(grouped["Clicks"] > 0, grouped["Orders"] / grouped["Clicks"] * 100, 0.0)
    grouped["CPC"] = np.where(grouped["Clicks"] > 0, grouped["Spend"] / grouped["Clicks"], 0.0)
    grouped["RPC"] = np.where(grouped["Clicks"] > 0, grouped["Sales"] / grouped["Clicks"], 0.0)
    grouped["ACOS"] = np.where(grouped["Sales"] > 0, grouped["Spend"] / grouped["Sales"] * 100, np.nan)

    # Default useful ordering
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
