# scripts/fetch_amazon_data.py

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
# Read API key from env (set this in GitHub Actions with: env RAPIDAPI_KEY: ${{ secrets.AMAZON_API_KEY }})
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY") or os.environ.get("AMAZON_API_KEY")
if not RAPIDAPI_KEY:
    print("ERROR: Missing RAPIDAPI_KEY (or AMAZON_API_KEY) environment variable.", file=sys.stderr)
    sys.exit(1)

# Comma-separated ASINs. You can also pass them via env ASIN_LIST or as a CLI arg.
DEFAULT_ASIN_LIST = "B00730QW70,B08C3PPYSV,B000QRAXSG,B07XRSC5WZ,B09F7Q9VSS,B0BMNZBHG1,B006AU57W0,B0D9PPG39W,B0DMVR83MW,B0148W0WOE"
ASIN_LIST = os.environ.get("ASIN_LIST") or DEFAULT_ASIN_LIST
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
    """
    Extract a numeric price from strings like '$34.99', '34.99', 'USD 1,299.00', or None.
    Returns None if not parseable.
    """
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
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None

def to_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (ValueError, TypeError):
        return None

def safe_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value in ("true", "True", "1", 1):
        return True
    if value in ("false", "False", "0", 0):
        return False
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
def build_rows(payload: Dict[str, Any], requested_asins: List[str]) -> List[Dict[str, Any]]:
    """
    Returns one row per product with the requested fields:
    asin, product_title, product_price, product_original_price, product_star_rating,
    product_num_ratings, is_amazon_choice, sales_volume, discount (%) (conditional), date
    """
    data = payload.get("data", [])
    # Index by asin for quick lookup (API returns a list)
    by_asin = {str(item.get("asin")).strip(): item for item in data if item and item.get("asin")}

    today = datetime.now(timezone.utc).date().isoformat()
    rows: List[Dict[str, Any]] = []

    for asin in requested_asins:
        raw = by_asin.get(asin)
        if not raw:
            # Create a stub row if the ASIN was requested but not returned
            rows.append({
                "asin": asin,
                "product_title": None,
                "product_price": None,
                "product_original_price": None,
                "product_star_rating": None,
                "product_num_ratings": None,
                "is_amazon_choice": None,
                "sales_volume": None,
                "discount": None,
                "date": today,
            })
            continue

        price = parse_money(raw.get("product_price"))
        orig = parse_money(raw.get("product_original_price"))
        rating = to_float(raw.get("product_star_rating"))
        num_ratings = to_int(raw.get("product_num_ratings"))
        is_choice = safe_bool(raw.get("is_amazon_choice"))
        sales_vol = raw.get("sales_volume")

        # Discount rule in percentage
        discount: Optional[float] = None
        if price is not None and orig is not None and orig > 0 and price < orig:
            discount = round((1 - (price / orig)) * 100, 2)

        rows.append({
            "asin": asin,
            "product_title": raw.get("product_title"),
            "product_price": price,
            "product_original_price": orig,
            "product_star_rating": rating,
            "product_num_ratings": num_ratings,
            "is_amazon_choice": is_choice,
            "sales_volume": sales_vol,
            "discount": discount,
            "date": today,
        })

    return rows


# -------------------------
# Main
# -------------------------
def main() -> None:
    requested_asins = [s.strip() for s in ASIN_LIST.split(",") if s.strip()]
    if not requested_asins:
        print("ERROR: No ASINs provided.", file=sys.stderr)
        sys.exit(1)

    payload = fetch_products(",".join(requested_asins), country=COUNTRY)

    status = payload.get("status")
    if status != "OK":
        print(f"ERROR: API status != OK (status={status})", file=sys.stderr)
        print(f"Payload head: {str(payload)[:500]}", file=sys.stderr)
        sys.exit(1)

    rows = build_rows(payload, requested_asins)
    df = pd.DataFrame(rows, columns=[
        "asin",
        "product_title",
        "product_price",
        "product_original_price",
        "product_star_rating",
        "product_num_ratings",
        "is_amazon_choice",
        "sales_volume",
        "discount",
        "date",
    ])

    # Append to CSV history (create header if file doesn't exist)
    write_header = not OUTPUT_PATH.exists()
    df.to_csv(OUTPUT_PATH, mode="a", index=False, header=write_header)

    # Optional: print a small preview to logs
    print("Wrote rows:", len(df))
    print(df.head(min(5, len(df))).to_string(index=False))


if __name__ == "__main__":
    main()
