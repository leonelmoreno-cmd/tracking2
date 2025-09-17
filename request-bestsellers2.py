# request-bestsellers.py
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
# Config
# -------------------------
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY") or os.environ.get("AMAZON_API_KEY")
if not RAPIDAPI_KEY:
    print("ERROR: Missing RAPIDAPI_KEY (or AMAZON_API_KEY).", file=sys.stderr)
    sys.exit(1)

COUNTRY = os.environ.get("COUNTRY", "US")
LANG = os.environ.get("LANGUAGE", "en_US")
SUBCATS_DIR = Path(os.environ.get("SUBCATS_DIR", "sub-categories"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "data/sub-categories"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

API_HOST = "real-time-amazon-data.p.rapidapi.com"
BEST_URL = "https://real-time-amazon-data.p.rapidapi.com/best-sellers"  # endpoint de Best Sellers

# -------------------------
# Helpers
# -------------------------
_money_pat = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")

def parse_money(v: Any) -> Optional[float]:
    if v is None: return None
    if isinstance(v, (int, float)): return float(v)
    s = str(v); m = _money_pat.search(s)
    if not m: return None
    try: return float(m.group(0).replace(",", ""))
    except ValueError: return None

def to_float(v: Any) -> Optional[float]:
    if v is None: return None
    try: return float(str(v).replace(",", ""))
    except (ValueError, TypeError): return None

def to_int(v: Any) -> Optional[int]:
    if v is None: return None
    try: return int(str(v).replace(",", ""))
    except (ValueError, TypeError): return None

def pct_change(old: Optional[float], new: Optional[float]) -> Optional[float]:
    """(new - old) / old * 100, con guardas contra None/0."""
    if old is None or new is None:
        return None
    try:
        old_f = float(old)
        new_f = float(new)
        if old_f == 0:
            return None
        return (new_f - old_f) / abs(old_f) * 100.0
    except Exception:
        return None

def read_category_from_txt(p: Path) -> Optional[str]:
    """
    Lee la única línea del TXT y extrae 'beauty/11060671' a partir de:
    '/best-sellers - beauty/11060671'
    """
    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    if " - " in raw:
        raw = raw.split(" - ", 1)[1]
    return raw.strip().lstrip("/")

def fetch_best_sellers(category: str, page: int = 1, country: str = "US", language: str = "en_US") -> Dict[str, Any]:
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": API_HOST,
    }
    params = {
        "category": category,
        "country": country,
        "language": language,
        "page": page
    }
    resp = requests.get(BEST_URL, headers=headers, params=params, timeout=60)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"[{category}] HTTP error: {e}\nResponse: {resp.text[:500]}", file=sys.stderr)
        raise
    try:
        return resp.json()
    except json.JSONDecodeError:
        print(f"[{category}] ERROR: Response is not valid JSON.", file=sys.stderr)
        print(resp.text[:1000], file=sys.stderr)
        raise

def normalize_best(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = (payload or {}).get("data") or {}
    arr = data.get("best_sellers") or []
    if not isinstance(arr, list):
        return []
    out = []
    for it in arr:
        out.append({
            "rank": to_int(it.get("rank")),
            "asin": str(it.get("asin") or "").strip() or None,
            "product_title": it.get("product_title"),
            "product_price": parse_money(it.get("product_price")),
            "product_star_rating": to_float(it.get("product_star_rating")),
            "product_num_ratings": to_int(it.get("product_num_ratings")),
            "product_url": it.get("product_url"),
            "product_photo": it.get("product_photo"),
            "rank_change_label": it.get("rank_change_label"),
        })
    return out

def load_prev_latest(hist_csv: Path) -> pd.DataFrame:
    """Carga el CSV previo de la subcategoría y devuelve la última fila por ASIN (snapshot previo)."""
    if not hist_csv.exists():
        return pd.DataFrame(columns=[
            "asin","rank","product_price","product_star_rating","product_num_ratings","date"
        ])
    prev = pd.read_csv(hist_csv)
    if "date" in prev.columns:
        prev["date"] = pd.to_datetime(prev["date"], errors="coerce")
        prev = prev.sort_values("date").groupby("asin", as_index=False).tail(1)
    return prev

# -------------------------
# Main
# -------------------------
def main() -> None:
    txts = list(SUBCATS_DIR.rglob("*.txt"))
    if not txts:
        print(f"WARNING: No TXT files found in {SUBCATS_DIR}/", file=sys.stderr)
        return

    now_utc = datetime.now(timezone.utc)
    today = now_utc.date().isoformat()
    week_num = now_utc.isocalendar().week  # ISO week para consistencia con Streamlit

    for txt in txts:
        category = read_category_from_txt(txt)
        if not category:
            print(f"[{txt}] WARNING: empty or invalid category line.", file=sys.stderr)
            continue

        # CSV de salida por TXT (un archivo por sub-categoría)
        stem = txt.stem  # p.ej. 'beauty_11060671'
        out_csv = OUTPUT_DIR / f"{stem}.csv"

        # 1) Descarga (con paginación sencilla)
        page = 1
        rows_all: List[Dict[str, Any]] = []
        while True:
            payload = fetch_best_sellers(category=category, page=page, country=COUNTRY, language=LANG)
            if payload.get("status") != "OK":
                print(f"[{category}] ERROR: API status != OK (status={payload.get('status')})", file=sys.stderr)
                break
            chunk = normalize_best(payload)
            if not chunk:
                break
            rows_all.extend(chunk)
            # Heurística de corte: la mayoría devuelve 50 o 100 por página
            if len(chunk) < 50:
                break
            page += 1

        if not rows_all:
            print(f"[{category}] No results.", file=sys.stderr)
            continue

        # 2) DataFrame nuevo snapshot
        df_new = pd.DataFrame(rows_all)
        df_new.insert(0, "category", category)
        df_new["date"] = today
        df_new["week"] = int(week_num)

        # 3) Cargar snapshot previo y calcular variaciones porcentuales
        prev_latest = load_prev_latest(out_csv)

        # Merge para % change
        merged = df_new.merge(
            prev_latest[["asin","rank","product_price","product_star_rating","product_num_ratings"]],
            on="asin", how="left", suffixes=("", "_prev")
        )

        merged["pct_rank"] = merged.apply(lambda r: pct_change(r.get("rank_prev"), r.get("rank")), axis=1)
        merged["pct_price"] = merged.apply(lambda r: pct_change(r.get("product_price_prev"), r.get("product_price")), axis=1)
        merged["pct_rating"] = merged.apply(lambda r: pct_change(r.get("product_star_rating_prev"), r.get("product_star_rating")), axis=1)
        merged["pct_num_ratings"] = merged.apply(lambda r: pct_change(r.get("product_num_ratings_prev"), r.get("product_num_ratings")), axis=1)

        # Flags de estatus de aparición
        prev_asins = set(prev_latest["asin"].dropna().astype(str).tolist())
        new_asins = set(merged["asin"].dropna().astype(str).tolist())
        merged["is_new"] = merged["asin"].astype(str).apply(lambda a: a not in prev_asins)
        merged["is_removed"] = False  # las salidas se manejarán como filas aparte
        merged["status"] = merged["is_new"].map(lambda x: "new" if x else "active")

        # 4) ASINes salientes: estaban antes y no están ahora → agregamos fila “removed”
        removed_asins = prev_asins - new_asins
        removed_rows: List[Dict[str, Any]] = []
        if removed_asins:
            prev_map = prev_latest.set_index("asin").to_dict(orient="index")
            for a in removed_asins:
                prev_row = prev_map.get(a, {})
                removed_rows.append({
                    "category": category,
                    "date": today,
                    "week": int(week_num),
                    "rank": prev_row.get("rank"),
                    "asin": a,
                    "product_title": None,
                    "product_price": prev_row.get("product_price"),
                    "product_star_rating": prev_row.get("product_star_rating"),
                    "product_num_ratings": prev_row.get("product_num_ratings"),
                    "product_url": None,
                    "product_photo": None,
                    "rank_change_label": None,
                    # prev values for ref (no se pisan)
                    "rank_prev": prev_row.get("rank"),
                    "product_price_prev": prev_row.get("product_price"),
                    "product_star_rating_prev": prev_row.get("product_star_rating"),
                    "product_num_ratings_prev": prev_row.get("product_num_ratings"),
                    # no hay % change (no existe valor nuevo)
                    "pct_rank": None,
                    "pct_price": None,
                    "pct_rating": None,
                    "pct_num_ratings": None,
                    # flags
                    "is_new": False,
                    "is_removed": True,
                    "status": "removed",
                })
        df_removed = pd.DataFrame(removed_rows) if removed_rows else pd.DataFrame(columns=list(merged.columns)+[
            "rank_prev","product_price_prev","product_star_rating_prev","product_num_ratings_prev"
        ])

        # 5) Limpieza de columnas *_prev (no necesitamos guardarlas)
        keep_cols = [
            "category","date","week",
            "rank","asin","product_title","product_price",
            "product_star_rating","product_num_ratings",
            "product_url","product_photo","rank_change_label",
            "pct_rank","pct_price","pct_rating","pct_num_ratings",
            "is_new","is_removed","status"
        ]
        merged_final = merged[keep_cols].copy()
        if not df_removed.empty:
            df_removed_final = df_removed[keep_cols].copy()
            merged_final = pd.concat([merged_final, df_removed_final], ignore_index=True)

        # 6) Append con header si es primera vez
        write_header = not out_csv.exists()
        merged_final.to_csv(out_csv, mode="a", index=False, header=write_header)

        print(f"[{category}] Wrote rows: {len(merged_final)} -> {out_csv}")

if __name__ == "__main__":
    main()
