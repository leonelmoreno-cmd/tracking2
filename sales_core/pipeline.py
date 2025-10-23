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


def run_pipeline(basket_name: str) -> pd.DataFrame:
    # 1) Leer ASINs + Brand desde el TXT (formato: ASIN,Brand)
    asin_brand_list = read_asins_for_basket(basket_name)  # List[tuple[str, str]]
    if not asin_brand_list:
        raise RuntimeError(f"ASIN list is empty for basket {basket_name}")

    # 2) Estructuras para fetch y mapping
    asins = [a for (a, _) in asin_brand_list]
    brand_map = {a: b for (a, b) in asin_brand_list}

    # 3) Ventana de fechas (últimas 4 semanas completas)
    start_date, end_date = four_full_weeks_window()

    # 4) Token Jungle Scout
    token = JSToken.from_secrets()

    # 5) Fetch diario
    daily = fetch_daily_for_asins(asins, start_date, end_date, token)
    if daily.empty:
        return pd.DataFrame()

    # 6) Inyectar marca desde el TXT
    daily["brand"] = daily["asin"].map(brand_map).fillna("Unknown").astype(str)

    # 7) Agregación semanal (Vie→Jue)
    weekly = to_weekly_fri_thu(daily)

    # 8) Guardar CSV local (el workflow lo commitea)
    write_weekly_csv_local(weekly, basket_name)
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
