from __future__ import annotations
import argparse
import pandas as pd
import streamlit as st
from .config import DEFAULT_BASKET
from .week_utils import four_full_weeks_window
from .js_client import JSToken, fetch_daily_for_asins
from .repo_io import read_asins_for_basket, write_weekly_csv_local
from .brand_map import read_competitor_brand_map, attach_brand
from .aggregate import to_weekly_fri_thu

def run_pipeline(basket_name: str, competitor_download_url: str | None = None) -> pd.DataFrame:
    # 1) ASINs
    asins = read_asins_for_basket(basket_name)
    if not asins:
        raise RuntimeError(f"ASIN list is empty for basket {basket_name}")

    # 2) Dates (last 4 full weeks)
    start_date, end_date = four_full_weeks_window()

    # 3) Jungle Scout token
    token = JSToken.from_secrets()

    # 4) Fetch daily
    daily = fetch_daily_for_asins(asins, start_date, end_date, token)
    if daily.empty:
        return pd.DataFrame()

    # 5) Brand map
    if competitor_download_url:
        try:
            brand_map = read_competitor_brand_map(competitor_download_url)
            daily = attach_brand(daily, brand_map)
        except Exception:
            daily["brand"] = "Unknown"
    else:
        daily["brand"] = "Unknown"

    # 6) Weekly aggregation
    weekly = to_weekly_fri_thu(daily)

    # 7) Save local CSV (Actions will commit)
    write_weekly_csv_local(weekly, basket_name)
    return weekly

# Optional CLI for GitHub Actions (headless)
def main_cli():
    p = argparse.ArgumentParser(description="Weekly sales ETL (Fri→Thu).")
    p.add_argument("--basket", default=DEFAULT_BASKET, help="Basket CSV name (e.g., synthethic3.csv)")
    p.add_argument("--competitor-url", default=None, help="download_url to competitor CSV (brand map)")
    args = p.parse_args()
    # Streamlit's st.secrets is unavailable in plain CLI; read env if needed.
    df = run_pipeline(args.basket, args.competitor_url)
    print(f"Wrote weekly CSV for {args.basket}. Rows: {len(df)}")

if __name__ == "__main__":
    main_cli()
