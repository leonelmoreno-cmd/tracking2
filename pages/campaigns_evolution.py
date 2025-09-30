# pages/campaigns_evolution.py

import streamlit as st
import pandas as pd
from components import campaigns_evolution_components as comp


def main():
    st.title("Campaigns Evolution")
    st.write("Upload Weekly 1, Weekly 2, and Weekly 3 files to analyze status evolution.")

    # File uploaders
    w1 = st.file_uploader("Upload Weekly 1", type=["csv", "xlsx", "xls"], key="w1")
    w2 = st.file_uploader("Upload Weekly 2", type=["csv", "xlsx", "xls"], key="w2")
    w3 = st.file_uploader("Upload Weekly 3", type=["csv", "xlsx", "xls"], key="w3")

    if w1 and w2 and w3:
        # Load dataframes
        dfs = [
            comp.load_weekly_file(w1),
            comp.load_weekly_file(w2),
            comp.load_weekly_file(w3)
        ]

        # Previews
        with st.expander("Weekly Previews"):
            for i, df in enumerate(dfs, start=1):
                st.subheader(f"Weekly {i}")
                st.write(df["status"].value_counts())
                st.dataframe(df.head())

        # Unify campaigns across weeks
        dfs = comp.unify_campaign_names(dfs)

        # Build transitions
        nodes, sources, targets, values = comp.build_transitions(dfs)

        # Sankey visualization
        fig = comp.create_sankey(nodes, sources, targets, values)
        st.plotly_chart(fig, use_container_width=True)

        # Evolution table (W1 → W2 → W3)
        evolution_df = comp.build_evolution_table(dfs)
        st.subheader("Full Evolution (W1 → W2 → W3)")
        st.dataframe(evolution_df)

        # Filters: campaigns in W3 with Purple or White
        filtered = evolution_df[evolution_df["W3_status"].isin(["Purple", "White"])]
        st.subheader("Filtered Campaigns (W3 Purple/White)")
        st.dataframe(filtered)

        # PDF Export
        if st.button("Export PDF"):
            pdf_path = comp.export_pdf(fig, filtered)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Download PDF", 
                    f, 
                    file_name="campaigns_evolution.pdf", 
                    mime="application/pdf"
                )
