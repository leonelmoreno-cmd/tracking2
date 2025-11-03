# sales_core/js_client.py
from __future__ import annotations
import os
import time
import typing as t
import requests
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from .config import JS_BASE_URL


# =============================================================
# Jungle Scout Authentication Token
# =============================================================

@dataclass
class JSToken:
    key_name: str
    api_key: str

    @classmethod
    def from_secrets(cls) -> "JSToken":
        """
        Load Jungle Scout credentials either from:
        - Streamlit secrets (when running locally)
        - Environment variables (when running in GitHub Actions)
        """
        key_name = None
        api_key = None

        # 1️⃣ Try Streamlit secrets first (local development)
        try:
            key_name = st.secrets.get("JS_KEY_NAME")
            api_key = st.secrets.get("JS_API_KEY")
        except Exception:
            pass  # st.secrets may not exist outside Streamlit

        # 2️⃣ Fallback to environment variables (GitHub Actions)
        if not key_name:
            key_name = os.getenv("JS_KEY_NAME")
        if not api_key:
            api_key = os.getenv("JS_API_KEY")

        # 3️⃣ Validate
        if not key_name or not api_key:
            raise RuntimeError(
                "❌ Missing Jungle Scout credentials. "
                "Please set JS_KEY_NAME and JS_API_KEY in GitHub Secrets or environment variables."
            )

        return cls(key_name, api_key)


# =============================================================
# Request Headers Helper
# =============================================================

def _headers(tok: JSToken) -> dict:
    """Build the HTTP headers required by the Jungle Scout API."""
    return {
        "Authorization": f"{tok.key_name}:{tok.api_key}",
        "X-API-Type": "junglescout",
        "Accept": "application/vnd.junglescout.v1+json",
        "Content-Type": "application/vnd.api+json",
    }


# =============================================================
# Fetch daily data for a single ASIN
# =============================================================

def fetch_daily_for_asin(
    asin: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    token: JSToken,
    marketplace: str = "us",
    retries: int = 2,
    backoff: float = 1.2,
) -> pd.DataFrame:
    """
    Request daily sales estimates for one ASIN between two dates.
    Returns a DataFrame with columns: asin, date, estimated_units_sold, last_known_price
    """

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

            # If the API returns no data, return empty DataFrame
            if not rows:
                return pd.DataFrame(columns=["asin", "date", "estimated_units_sold", "last_known_price"])

            # Parse daily records
            days = rows[0].get("attributes", {}).get("data", [])
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
            # Retry on network/API failure
            if attempt < retries:
                wait = backoff ** attempt
                print(f"⚠️ Retry {attempt + 1}/{retries} for ASIN {asin} after {wait:.1f}s due to error: {e}")
                time.sleep(wait)
                continue
            raise RuntimeError(f"JS API error for {asin}: {e}") from e


# =============================================================
# Fetch daily data for multiple ASINs
# =============================================================
# sales_core/js_client.py
def fetch_daily_for_asins(asins, start_date, end_date, token):
    frames = []
    failures = []  # ← nuevo

    for asin in asins:
        try:
            with st.status(f"📦 Fetching {asin}...", expanded=False):
                df_asin = fetch_daily_for_asin(asin, start_date, end_date, token)

                # Si el API devuelve vacío (asin sin datos / no disponible)
                if df_asin.empty:
                    failures.append((asin, "no_data_or_unavailable"))

                frames.append(df_asin)

        except RuntimeError as e:
            failures.append((asin, str(e)))  # guardas el error y sigues
            continue

    if not frames:
        return (
            pd.DataFrame(columns=["asin","date","estimated_units_sold","last_known_price"]),
            failures
        )

    out = pd.concat(frames, ignore_index=True).sort_values(["asin","date"]).reset_index(drop=True)
    return out, failures  # ← ahora regresa datos + fallos
