# sales_core/pipeline.py
from __future__ import annotations
import argparse
import pandas as pd
import streamlit as st

from .config import DEFAULT_BASKET
from .week_utils import four_full_weeks_window
from .js_client import JSToken, fetch_daily_for_asins
from .repo_io import read_asins_for_basket, write_weekly_csv_local
from .aggregate import to_weekly_fri_thu

# sales_core/pipeline.py
def run_pipeline(basket_name: str) -> pd.DataFrame:
    asin_brand_list = read_asins_for_basket(basket_name)
    if not asin_brand_list:
        raise RuntimeError(f"ASIN list is empty for basket {basket_name}")

    asins = [a for (a, _) in asin_brand_list]
    brand_map = {a: b for (a, b) in asin_brand_list}

    start_date, end_date = four_full_weeks_window()
    token = JSToken.from_secrets()

    # ↓↓↓ antes: daily = fetch_daily_for_asins(...)
    daily, failures = fetch_daily_for_asins(asins, start_date, end_date, token)

    # Avisar sin romper
    if failures:
        for asin, reason in failures:
            print(f"⚠️ ASIN {asin} sin datos o con error: {reason}")

    if daily.empty:
        print("ℹ️ No se generaron datos (todos fallaron o no hubo datos).")
        return pd.DataFrame()

    daily["brand"] = daily["asin"].map(brand_map).fillna("Unknown").astype(str)
    weekly = to_weekly_fri_thu(daily)
    write_weekly_csv_local(weekly, basket_name)

    ok_count = daily["asin"].nunique()
    fail_count = len(failures)
    print(f"✅ ASINs OK: {ok_count} | ❌ ASINs fallidos: {fail_count}")
    return weekly

# CLI para GitHub Actions (headless)
def main_cli():
    p = argparse.ArgumentParser(description="Weekly sales ETL (Fri→Thu).")
    p.add_argument("--basket", default=DEFAULT_BASKET, help="Basket TXT name (e.g., asins_IC.txt)")
    args = p.parse_args()

    df = run_pipeline(args.basket)
    print(f"Wrote weekly CSV for {args.basket}. Rows: {len(df)}")


if __name__ == "__main__":
    main_cli()
