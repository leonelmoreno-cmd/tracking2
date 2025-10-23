from __future__ import annotations
import argparse
import pandas as pd
import streamlit as st
from .config import DEFAULT_BASKET
from .week_utils import four_full_weeks_window
from .js_client import JSToken, fetch_daily_for_asins
from .repo_io import read_asins_for_basket, write_weekly_csv_local
# from .brand_map import read_competitor_brand_map, attach_brand  # <-- ya no necesitamos
from .aggregate import to_weekly_fri_thu

def run_pipeline(basket_name: str, competitor_download_url: str | None = None) -> pd.DataFrame:
    # 1) ASINs + Brand
    asin_brand_list = read_asins_for_basket(basket_name)  # ahora retorna List[tuple(asin, brand)]
    if not asin_brand_list:
        raise RuntimeError(f"ASIN list is empty for basket {basket_name}")

    # 2) Prepara estructuras para fetch + mapping
    asins = [a for (a, _) in asin_brand_list]
    brand_map = {a: b for (a, b) in asin_brand_list}

    # 3) Dates (last 4 full weeks)
    start_date, end_date = four_full_weeks_window()

    # 4) Jungle Scout token
    token = JSToken.from_secrets()

    # 5) Fetch daily data
    daily = fetch_daily_for_asins(asins, start_date, end_date, token)
    if daily.empty:
        return pd.DataFrame()

    # 6) Inyecta la marca desde nuestro mapping
    daily["brand"] = daily["asin"].map(brand_map).fillna("Unknown").astype(str)

    # 7) Weekly aggregation
    weekly = to_weekly_fri_thu(daily)

    # 8) Save local CSV (Actions will commit)
    write_weekly_csv_local(weekly, basket_name)

    return weekly

# Optional CLI for GitHub Actions (headless)
def main_cli():
    p = argparse.ArgumentParser(description="Weekly sales ETL (Fri→Thu).")
    p.add_argument("--basket", default=DEFAULT_BASKET, help="Basket TXT name (e.g., synthethic3.txt)")
    # ya no usamos competitor-url para marca externa
    args = p.parse_args()
    df = run_pipeline(args.basket, None)
    print(f"Wrote weekly CSV for {args.basket}. Rows: {len(df)}")

if __name__ == "__main__":
    main_cli()
