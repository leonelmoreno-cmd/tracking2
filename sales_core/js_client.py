from __future__ import annotations
import time
import typing as t
import requests
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from .config import JS_BASE_URL

@dataclass
class JSToken:
    key_name: str
    api_key: str

    @classmethod
    def from_secrets(cls) -> "JSToken":
        try:
            key_name = st.secrets["JS_KEY_NAME"]
            api_key  = st.secrets["JS_API_KEY"]
            if not key_name or not api_key:
                raise KeyError
            return cls(key_name, api_key)
        except KeyError:
            raise RuntimeError("Missing JS_KEY_NAME or JS_API_KEY in Streamlit secrets.")

def _headers(tok: JSToken) -> dict:
    return {
        "Authorization": f"{tok.key_name}:{tok.api_key}",
        "X-API-Type": "junglescout",
        "Accept": "application/vnd.junglescout.v1+json",
        "Content-Type": "application/vnd.api+json",
    }

def fetch_daily_for_asin(
    asin: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    token: JSToken,
    marketplace: str = "us",
    retries: int = 2,
    backoff: float = 1.2,
) -> pd.DataFrame:
    params = {
        "marketplace": marketplace,
        "asin": asin,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
    }
    for attempt in range(retries + 1):
        try:
            resp = requests.get(JS_BASE_URL, params=params, headers=_headers(token), timeout=60)
            resp.raise_for_status()
            js = resp.json()
            rows = js.get("data", [])
            if not rows:
                return pd.DataFrame(columns=["asin", "date", "estimated_units_sold", "last_known_price"])
            days = rows[0]["attributes"]["data"]
            recs = []
            for d in days:
                recs.append({
                    "asin": asin,
                    "date": pd.to_datetime(d.get("date"), errors="coerce"),
                    "estimated_units_sold": pd.to_numeric(d.get("estimated_units_sold"), errors="coerce"),
                    "last_known_price": pd.to_numeric(d.get("last_known_price"), errors="coerce"),
                })
            df = pd.DataFrame(recs).dropna(subset=["date"])
            df["estimated_units_sold"] = df["estimated_units_sold"].fillna(0).astype(int)
            df["last_known_price"] = df["last_known_price"].fillna(0.0).astype(float)
            return df
        except requests.RequestException as e:
            if attempt < retries:
                time.sleep(backoff ** attempt)
                continue
            raise RuntimeError(f"JS API error for {asin}: {e}") from e

def fetch_daily_for_asins(
    asins: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    token: JSToken,
) -> pd.DataFrame:
    frames = []
    for asin in asins:
        with st.status(f"Fetching {asin}...", expanded=False):
            frames.append(fetch_daily_for_asin(asin, start_date, end_date, token))
    if not frames:
        return pd.DataFrame(columns=["asin", "date", "estimated_units_sold", "last_known_price"])
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["asin", "date"])
