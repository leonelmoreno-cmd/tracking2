import os
import re
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple  # <- añadimos Tuple

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

# --- NUEVOS helpers para Best Sellers Rank (desde el código funcional) ---
def parse_best_sellers_rank_text(bsr_text: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """
    Devuelve (sub_category_name, rank) a partir del texto 'Best Sellers Rank'.
    Soporta variantes con varios '#' y paréntesis como '(See Top 100 in ...)'.
    """
    if not bsr_text or str(bsr_text).strip().lower() in {"not available", "n/a"}:
        return None, None

    parts = [p.strip() for p in str(bsr_text).split('#') if p.strip()]
    # Recorremos de derecha a izquierda para tomar la categoría más específica
    for part in reversed(parts):
        if " in " in part:
            left, right = part.split(" in ", 1)
            # rank = primer número en 'left' (acepta comas)
            m = re.search(r"\d[\d,]*", left)
            rank = int(m.group(0).replace(",", "")) if m else None
            # subcat = texto antes de cualquier paréntesis extra
            subcat = right.split(" (", 1)[0].strip()
            return (subcat or None), rank
    # Fallback si no hay " in "
    return (parts[-1] if parts else None), None

def get_best_sellers_rank_fields(item: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    info = (item or {}).get("product_information") or {}
    details = (item or {}).get("product_details") or {}
    bsr_text = info.get("Best Sellers Rank") or details.get("Best Sellers Rank")
    return parse_best_sellers_rank_text(bsr_text)


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

        # --- Control cuando falta el item del ASIN (desde el código funcional) ---
        # Si no hay datos para el ASIN, fila con defaults y continuar
        if not raw:
            rows.append({
                "asin": asin,
                "product_title": None,
                "product_price": None,
                "product_original_price": None,
                "product_star_rating": None,
                "product_num_ratings": None,
                "is_amazon_choice": None,
                "is_best_seller": None,
                "sales_volume": None,   # se deja tal cual venga
                "discount": None,
                "brand": None,
                "product_url": None,
                "product_photo": None,  # <-- NUEVO
                "date": today,
                "week": week_num,
                "unit_price": "N/A",
                "sub_category_name": "Not Available",
                "rank": "Not Available",
            })
            continue

        # --- Extracción robusta de BSR y Unit Price (helpers dedicados) ---
        sub_category_name, rank = get_best_sellers_rank_fields(raw)
        unit_price = raw.get("unit_price", "N/A")

        if not sub_category_name:
            sub_category_name = "Not Available"
        if rank is None:
            rank = "Not Available"

        # Resto de campos
        price = parse_money(raw.get("product_price"))
        orig = parse_money(raw.get("product_original_price"))
        rating = to_float(raw.get("product_star_rating"))
        num_ratings = to_int_from_text(raw.get("product_num_ratings"))
        is_choice = safe_bool(raw.get("is_amazon_choice"))
        is_best = safe_bool(raw.get("is_best_seller"))
        sales_vol = raw.get("sales_volume")  # mantener texto original del API
        brand = extract_brand(raw)
        product_url = raw.get("product_url")
        photo_url = raw.get("product_photo") or None  # <-- NUEVO

        discount = None
        if price is not None and orig is not None and orig > 0 and price < orig:
            discount = round((1 - (price / orig)) * 100, 2)  # 0–100

        rows.append({
            "asin": asin,
            "product_title": raw.get("product_title"),
            "product_price": price,
            "product_original_price": orig,
            "product_star_rating": rating,
            "product_num_ratings": num_ratings,
            "is_amazon_choice": is_choice,
            "is_best_seller": is_best,
            "sales_volume": sales_vol,
            "discount": discount,
            "brand": brand,
            "product_url": product_url,
            "product_photo": photo_url,  # <-- NUEVO
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
        # --- Conversión opcional de rank (comentada, como en el código funcional) ---
        # "rank",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Si quieres convertir rank numéricamente, descomenta esta parte:
    # if "rank" in df.columns:
    #     df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

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
