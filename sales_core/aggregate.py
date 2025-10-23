from __future__ import annotations
import numpy as np
import pandas as pd
from .week_utils import week_end_for_date, week_label_iso

def to_weekly_fri_thu(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: daily (asin, date, estimated_units_sold, last_known_price, brand)
    Output (weekly):
      asin, brand, week_end, week_number, units_sold, sales_amount, avg_price
    """
    if daily_df.empty:
        return pd.DataFrame(columns=[
            "asin", "brand", "week_end", "week_number", "units_sold", "sales_amount", "avg_price"
        ])
    w = daily_df.copy()
    w["brand"] = w["brand"].fillna("Unknown").astype(str)
    w["date"] = pd.to_datetime(w["date"], errors="coerce")
    w = w.dropna(subset=["date"])
    w["sales_amount_day"] = w["estimated_units_sold"] * w["last_known_price"]
    w["week_end"] = w["date"].apply(week_end_for_date)
    w["week_number"] = w["week_end"].apply(week_label_iso).astype(int)

    gb = w.groupby(["asin", "brand", "week_end", "week_number"], as_index=False).agg(
        units_sold=("estimated_units_sold", "sum"),
        sales_amount=("sales_amount_day", "sum"),
        avg_price=("last_known_price", "mean"),
    )
    return gb.sort_values(["brand", "asin", "week_end"])

def compute_highlights_for_week(weekly_df: pd.DataFrame, selected_week_end: pd.Timestamp) -> dict:
    cur = weekly_df.loc[weekly_df["week_end"] == selected_week_end].copy()
    if cur.empty:
        return {}
    cur_b = cur.groupby("brand", as_index=False).agg(
        total_sales=("sales_amount", "sum"),
        total_units=("units_sold", "sum")
    )
    top_sales = cur_b.loc[cur_b["total_sales"].idxmax()]
    low_sales = cur_b.loc[cur_b["total_sales"].idxmin()]
    avg_sales = float(cur_b["total_sales"].mean())

    top_units = cur_b.loc[cur_b["total_units"].idxmax()]
    low_units = cur_b.loc[cur_b["total_units"].idxmin()]
    avg_units = float(cur_b["total_units"].mean())

    prev_we = selected_week_end - pd.Timedelta(days=7)
    prev_b = (
        weekly_df.loc[weekly_df["week_end"] == prev_we, ["brand", "sales_amount"]]
        .groupby("brand", as_index=False)["sales_amount"].sum()
        .rename(columns={"sales_amount": "prev_sales"})
    )
    cur_b = cur_b.merge(prev_b, on="brand", how="left")
    cur_b["prev_sales"] = cur_b["prev_sales"].fillna(0.0)

    def pct_change(row):
        prev = row["prev_sales"]
        curv = row["total_sales"]
        if prev == 0:
            if curv == 0:
                return 0.0
            return float("inf")
        return (curv - prev) / prev * 100.0

    cur_b["pct_var"] = cur_b.apply(pct_change, axis=1)

    high_var_brand = cur_b.loc[cur_b["pct_var"].idxmax()]
    low_var_brand  = cur_b.loc[cur_b["pct_var"].idxmin()]
    avg_var = float(cur_b["pct_var"].replace([np.inf, -np.inf], np.nan).fillna(0.0).mean())

    return {
        "sales": {
            "highest": (high_var_brand["brand"], float(top_sales["total_sales"])),
            "lowest": (low_sales["brand"], float(low_sales["total_sales"])),
            "average": avg_sales,
        },
        "variation": {
            "highest": (high_var_brand["brand"], float(high_var_brand["pct_var"])),
            "lowest": (low_var_brand["brand"], float(low_var_brand["pct_var"])),
            "average": avg_var,
        },
        "units": {
            "most": (top_units["brand"], int(top_units["total_units"])),
            "least": (low_units["brand"], int(low_units["total_units"])),
            "average": float(avg_units),
        },
    }
