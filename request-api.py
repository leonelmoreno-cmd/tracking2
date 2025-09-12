import os
import re
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
import pandas as pd


# -------------------------
# Configuration
# -------------------------
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY") or os.environ.get("AMAZON_API_KEY")
if not RAPIDAPI_KEY:
    print("ERROR: Missing RAPIDAPI_KEY (or AMAZON_API_KEY) environment variable.", file=sys.stderr)
    sys.exit(1)

ASIN_FILE = os.environ.get("ASIN_FILE")  # obligatorio en workflow
COUNTRY = os.environ.get("COUNTRY", "US")
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "data/competitors_history.csv"))
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

API_URL = "https://real-time-amazon-data.p.rapidapi.com/product-details"
API_HOST = "real-time-amazon-data.p.rapidapi.com"


# -------------------------
# Helpers
# -------------------------
_money_pat = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")

def parse_money(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value)
    m = _money_pat.search(s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None

def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", ""))
    except (ValueError, TypeError):
        return None

def to_int_from_text(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip().lower().replace(",", "")
    try:
        return int(float(s))
    except ValueError:
        return None

def safe_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("true", "1"):
        return True
    if str(value).lower() in ("false", "0"):
        return False
    return None

def extract_brand(item: Dict[str, Any]) -> Optional[str]:
    info = item.get("product_information") or {}
    details = item.get("product_details") or {}
    for d in (info, details):
        for key in ("Brand Name", "Brand"):
            if d.get(key):
                return str(d[key]).strip()
    return None


# -------------------------
# Fetch
# -------------------------
def fetch_products(asins_csv: str, country: str = "US") -> Dict[str, Any]:
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": API_HOST,
    }
    params = {"asin": asins_csv, "country": country}
    resp = requests.get(API_URL, headers=headers, params=params, timeout=60)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"HTTP error: {e}\nResponse: {resp.text[:500]}", file=sys.stderr)
        raise
    try:
        return resp.json()
    except json.JSONDecodeError:
        print("ERROR: Response is not valid JSON.", file=sys.stderr)
        print(resp.text[:1000], file=sys.stderr)
        raise


# -------------------------
# Transform
# -------------------------
def normalize_data_list(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = payload.get("data")
    if data is None:
        return []
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        return [data]
    return []

def build_rows(payload: Dict[str, Any], requested_asins: List[str]) -> List[Dict[str, Any]]:
    data_list = normalize_data_list(payload)
    by_asin = {str(item.get("asin")).strip(): item for item in data_list if item and item.get("asin")}

    today_dt = datetime.now(timezone.utc)
    today = today_dt.date().isoformat()
    week_num = today_dt.isocalendar().week

    rows: List[Dict[str, Any]] = []
    
    for asin in requested_asins:
        raw = by_asin.get(asin)
        
        # Inicializar valores por defecto
        best_seller_rank = ""
        sub_category_name = ""
        rank = None
        unit_price = "NA"

        if raw and isinstance(raw, dict):
            # Extraer Best Sellers Rank
            best_seller_rank = raw.get("product_details", {}).get("Best Sellers Rank", "")
            if best_seller_rank:
                parts = best_seller_rank.split('#')
                if len(parts) > 1:
                    sub_category_name = parts[-1].split(' in ')[-1].strip()
                    rank = parts[-2].strip()
                    try:
                        rank = int(rank.replace(",", ""))
                    except ValueError:
                        rank = None
                        print(f"DEBUG: Failed to convert rank to integer for ASIN {asin}: {rank}", file=sys.stderr)
            else:
                print(f"DEBUG: Invalid Best Sellers Rank format for ASIN {asin}: {best_seller_rank}", file=sys.stderr)

            # Obtener unit_price
            unit_price = raw.get("unit_price", "NA")
            if unit_price == "NA":
                print(f"DEBUG: unit_price not found for ASIN: {asin}", file=sys.stderr)

        rows.append({
            "asin": asin,
            "product_title": raw.get("product_title") if raw else None,
            "product_price": parse_money(raw.get("product_price")) if raw else None,
            "product_original_price": parse_money(raw.get("product_original_price")) if raw else None,
            "product_star_rating": to_float(raw.get("product_star_rating")) if raw else None,
            "product_num_ratings": to_int_from_text(raw.get("product_num_ratings")) if raw else None,
            "is_amazon_choice": safe_bool(raw.get("is_amazon_choice")) if raw else None,
            "is_best_seller": safe_bool(raw.get("is_best_seller")) if raw else None,
            "sales_volume": raw.get("sales_volume") if raw else None,
            "discount": None,
            "brand": extract_brand(raw) if raw else None,
            "product_url": raw.get("product_url") if raw else None,
            "date": today,
            "week": week_num,
            "unit_price": unit_price,
            "sub_category_name": sub_category_name,
            "rank": rank,
        })

    return rows


# -------------------------
# Main
# -------------------------
def main() -> None:
    if not ASIN_FILE or not Path(ASIN_FILE).exists():
        print("ERROR: ASIN_FILE not provided or does not exist.", file=sys.stderr)
        sys.exit(1)

    requested_asins = [l.strip() for l in Path(ASIN_FILE).read_text().splitlines() if l.strip()]
    if not requested_asins:
        print(f"ERROR: No ASINs found in {ASIN_FILE}.", file=sys.stderr)
        sys.exit(1)

    payload = fetch_products(",".join(requested_asins), country=COUNTRY)

    status = payload.get("status")
    if status != "OK":
        print(f"ERROR: API status != OK (status={status})", file=sys.stderr)
        sys.exit(1)

    rows = build_rows(payload, requested_asins)
    df = pd.DataFrame(rows)

    # Sufijo dinámico (Asins/asins_UR.txt -> UR)
    suffix = Path(ASIN_FILE).stem.split("_")[-1]
    output_with_suffix = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.stem} - {suffix}{OUTPUT_PATH.suffix}")

    write_header = not output_with_suffix.exists()

    # Tipificar columnas numéricas; discount es porcentaje (0–100)
    numeric_cols = [
        "product_price",
        "product_original_price",
        "product_star_rating",
        "product_num_ratings",
        "discount",
        "week",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Booleans con NA
    for col in ["is_amazon_choice", "is_best_seller"]:
        if col in df.columns:
            df[col] = df[col].astype("boolean")

    # sales_volume permanece como string original
    df.to_csv(output_with_suffix, mode="a", index=False, header=write_header)

    print("Wrote rows:", len(df))
    print("Output file:", str(output_with_suffix))
    print(df.head(min(5, len(df))).to_string(index=False))


if __name__ == "__main__":
    main()
