# pages/ad_product_type.py
import pandas as pd
import streamlit as st
import altair as alt


def _load_and_clean_ad_product_type_csv(file) -> pd.DataFrame:
    """
    Loads the CSV and cleans it.

    Critical rule from you:
    - Drop rows that don't have ASINs (we cannot assume where they belong).
    """
    df = pd.read_csv(file)

    # Normalize column names (defensive)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"asin", "ad_product_type", "total_spend_dollars", "total_sales_dollars", "roas"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    # Strip whitespace + turn empty strings into NA
    df["asin"] = df["asin"].astype(str).str.strip()
    df["ad_product_type"] = df["ad_product_type"].astype(str).str.strip()

    # Rows like asin="" or asin="nan" should be treated as missing
    df.loc[df["asin"].isin(["", "nan", "none", "null"]), "asin"] = pd.NA
    df.loc[df["ad_product_type"].isin(["", "nan", "none", "null"]), "ad_product_type"] = pd.NA

    # ✅ Your requirement: omit rows without ASINs (don't infer / forward-fill)
    df = df.dropna(subset=["asin"])

    # Also drop rows without ad_product_type (can't chart/group them reliably)
    df = df.dropna(subset=["ad_product_type"])

    # Convert numeric columns
    num_cols = ["total_spend_dollars", "total_sales_dollars", "roas"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # If spend/sales are missing, treat as 0 (common for sparse exports)
    df["total_spend_dollars"] = df["total_spend_dollars"].fillna(0.0)
    df["total_sales_dollars"] = df["total_sales_dollars"].fillna(0.0)

    # ROAS: if missing, recompute when possible
    # (Still safe because it uses the same row’s spend/sales; no inference across rows)
    roas_missing = df["roas"].isna()
    df.loc[roas_missing, "roas"] = df.loc[roas_missing].apply(
        lambda r: (r["total_sales_dollars"] / r["total_spend_dollars"]) if r["total_spend_dollars"] else 0.0,
        axis=1,
    )

    # Aggregate in case file contains duplicate ASIN+type rows
    df = (
        df.groupby(["asin", "ad_product_type"], as_index=False)[
            ["total_spend_dollars", "total_sales_dollars"]
        ]
        .sum()
    )
    df["roas"] = df.apply(
        lambda r: (r["total_sales_dollars"] / r["total_spend_dollars"]) if r["total_spend_dollars"] else 0.0,
        axis=1,
    )

    return df


def _stacked_bar(df: pd.DataFrame, value_col: str, title: str) -> alt.Chart:
    # Sort ASINs by total value (best practice: meaningful ordering)
    asin_order = (
        df.groupby("asin", as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=False)["asin"]
        .tolist()
    )

    base = alt.Chart(df).encode(
        x=alt.X(
            "asin:N",
            sort=asin_order,
            title="ASIN",
            axis=alt.Axis(labelAngle=0),
        ),
        y=alt.Y(
            f"{value_col}:Q",
            title=title,
            axis=alt.Axis(format="$,.0f"),
        ),
        color=alt.Color(
            "ad_product_type:N",
            title="Ad product type",
            legend=alt.Legend(orient="right"),
        ),
        tooltip=[
            alt.Tooltip("asin:N", title="ASIN"),
            alt.Tooltip("ad_product_type:N", title="Ad type"),
            alt.Tooltip("total_sales_dollars:Q", title="Sales", format="$,.2f"),
            alt.Tooltip("total_spend_dollars:Q", title="Spend", format="$,.2f"),
            alt.Tooltip("roas:Q", title="ROAS", format=".2f"),
        ],
    )

    chart = (
        base.mark_bar()
        .properties(height=380, title=title)
        .configure_title(fontSize=16, anchor="start")
        .configure_axis(labelFontSize=12, titleFontSize=12)
        .configure_legend(labelFontSize=12, titleFontSize=12)
    )
    return chart


def main():
    st.title("Ad product type")

    st.write(
        "This page shows Spend/Sales/ROAS by **ASIN** and **ad_product_type**. "
        "Rows without ASINs are **dropped** (we don’t infer ownership)."
    )

    with st.expander("Data source", expanded=True):
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        st.caption("Expected columns: asin, ad_product_type, total_spend_dollars, total_sales_dollars, roas")

        # Optional: support a default local file path
        use_local = st.toggle("Use local file path instead of upload", value=False)
        local_path = st.text_input("Local CSV path", value="data/ad_product_type.csv", disabled=not use_local)

    if not uploaded and not use_local:
        st.info("Upload a CSV to get started (or toggle local file path).")
        return

    try:
        source = local_path if use_local else uploaded
        df = _load_and_clean_ad_product_type_csv(source)
    except Exception as e:
        st.error(f"Could not load/clean CSV: {e}")
        return

    # Build the table in your desired format
    table = df.rename(
        columns={
            "ad_product_type": "ad type",
            "total_spend_dollars": "Spend",
            "total_sales_dollars": "Sales",
            "roas": "ROAS",
        }
    )[["asin", "ad type", "Spend", "Sales", "ROAS"]].copy()

    # Table ordering: sponsored_products first, then sponsored_brands, then others; within each by Spend desc
    type_order = {"sponsored_products": 0, "sponsored_brands": 1}
    table["_type_rank"] = table["ad type"].map(type_order).fillna(9).astype(int)
    table = table.sort_values(by=["_type_rank", "Spend"], ascending=[True, False]).drop(columns=["_type_rank"])

    st.subheader("Table")
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Spend": st.column_config.NumberColumn("Spend", format="$%,.2f"),
            "Sales": st.column_config.NumberColumn("Sales", format="$%,.2f"),
            "ROAS": st.column_config.NumberColumn("ROAS", format="%.2f"),
        },
    )

    st.subheader("Charts")

    # Sales stacked
    sales_chart = _stacked_bar(df, "total_sales_dollars", "Total Sales ($)")
    st.altair_chart(sales_chart, use_container_width=True)

    # Spend stacked
    spend_chart = _stacked_bar(df, "total_spend_dollars", "Total Spend ($)")
    st.altair_chart(spend_chart, use_container_width=True)
