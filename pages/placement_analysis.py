import pandas as pd
import streamlit as st
import altair as alt


# -----------------------------
# Cleaning & normalization
# -----------------------------
def load_and_clean_placements(file) -> pd.DataFrame:
    df = pd.read_csv(file)

    # Normalize columns
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"placement_type", "asin", "spend", "clicks", "sales", "orders"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Clean ASIN
    df["asin"] = df["asin"].astype(str).str.strip()
    df.loc[df["asin"].isin(["", "nan", "none", "null"]), "asin"] = pd.NA
    df = df.dropna(subset=["asin"])

    # Remove unwanted placements
    excluded = {
        "off amazon",
        "offsite",
        "amazon onsite",
        "homepage on-amazon",
    }
    df["placement_type"] = df["placement_type"].astype(str).str.strip()
    df = df[~df["placement_type"].str.lower().isin(excluded)]

    # Normalize placement names
    placement_map = {
        "Detail Page on-Amazon": "Product Pages",
        "Other on-Amazon": "Rest of Search",
        "Top of Search on-Amazon": "Top of Search",
    }
    df["placement"] = df["placement_type"].map(placement_map)
    df = df.dropna(subset=["placement"])

    # Numeric columns
    num_cols = ["spend", "clicks", "sales", "orders"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Aggregate safely
    df = (
        df.groupby(["placement", "asin"], as_index=False)[num_cols]
        .sum()
    )

    # CVR
    df["cvr"] = df.apply(
        lambda r: (r["orders"] / r["clicks"]) if r["clicks"] > 0 else 0,
        axis=1,
    )
    df["cvr"] = df["cvr"] * 100
    return df

def placement_asin_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "placement": "Placement",
            "asin": "ASIN",
            "spend": "Spend",
            "clicks": "Clicks",
            "sales": "Sales",
            "orders": "Orders",
            "cvr": "CVR",
        }
    )[["Placement", "ASIN", "Spend", "Clicks", "Sales", "Orders", "CVR"]]


def placement_cvr_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("placement", as_index=False)
        .agg({"orders": "sum", "clicks": "sum"})
    )
    summary["CVR"] = summary.apply(
        lambda r: (r["orders"] / r["clicks"]) if r["clicks"] > 0 else 0,
        axis=1,
    )
    summary["CVR"] = summary["CVR"] * 100
    return summary.rename(
        columns={"placement": "Placement"}
    )[["Placement", "CVR"]]

def stacked_percent_chart(df: pd.DataFrame, value_col: str, title: str) -> alt.Chart:
    # Compute percentages per ASIN
    pct_df = df.copy()
    pct_df["total"] = pct_df.groupby("asin")[value_col].transform("sum")
    pct_df["pct"] = pct_df.apply(
        lambda r: (r[value_col] / r["total"]) if r["total"] > 0 else 0,
        axis=1,
    )

    asin_order = (
        df.groupby("asin", as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=False)["asin"]
        .tolist()
    )

    return (
        alt.Chart(pct_df)
        .mark_bar()
        .encode(
            x=alt.X(
                    "asin:N",
                    sort=asin_order,
                    title="ASIN",
                    axis=alt.Axis(
                    labelAngle=0,
                    labelLimit=120,
                    labelFontSize=11,
                    ),
            ),
            y=alt.Y(
                "pct:Q",
                axis=alt.Axis(format="%"),
                title=title,
            ),
            color=alt.Color(
                "placement:N",
                title="Placement",
                legend=alt.Legend(orient="right"),
            ),
            tooltip=[
                alt.Tooltip("asin:N", title="ASIN"),
                alt.Tooltip("placement:N", title="Placement"),
                alt.Tooltip("pct:Q", title="Share", format=".1%"),
                alt.Tooltip(f"{value_col}:Q", title=value_col.capitalize(), format="$,.2f"),
            ],
        )
        .properties(height=380, title=title)
        .configure_title(anchor="start", fontSize=16)
        .configure_axis(labelFontSize=12, titleFontSize=12)
        .configure_legend(labelFontSize=12, titleFontSize=12)
    )

def main():
    st.title("Placement Analysis")

    st.write(
        "Analysis of **Spend, Sales and CVR by placement**. "
        "Off-Amazon, Offsite, Homepage and Amazon Onsite placements are excluded."
    )

    uploaded = st.file_uploader("Upload placement CSV", type=["csv"])
    if not uploaded:
        st.info("Upload a CSV to get started.")
        return

    try:
        df = load_and_clean_placements(uploaded)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return

    # ---------------- Tables ----------------
    st.subheader("Placement × ASIN performance")
    table = placement_asin_table(df)

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Placement CVR summary")
    cvr_summary = placement_cvr_summary(df)

    st.dataframe(
        cvr_summary,
        use_container_width=True,
        hide_index=True,
    )

    # ---------------- Charts ----------------
    st.subheader("Spend distribution by placement (%)")
    st.altair_chart(
        stacked_percent_chart(df, "spend", "Spend by Placement by ASIN (%)"),
        use_container_width=True,
    )

    st.subheader("Sales distribution by placement (%)")
    st.altair_chart(
        stacked_percent_chart(df, "sales", "Sales by Placement by ASIN (%)"),
        use_container_width=True,
    )

