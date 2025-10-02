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
        dfs = [
            comp.load_weekly_file(w1),
            comp.load_weekly_file(w2),
            comp.load_weekly_file(w3)
        ]

        # Unify campaign names across weeks
        dfs = comp.unify_campaign_names(dfs)

        # Build evolution tables
        full_evolution, filtered_w3 = comp.build_evolution_table(dfs)

        # Display full evolution table
        st.subheader("Full Evolution of Campaigns")
        st.dataframe(full_evolution)

        # Display filtered W3 table
        st.subheader("Filtered Campaigns (W3: Purple/White)")
        st.dataframe(filtered_w3)

        # Sankey diagram for each status
        nodes, sources, targets, values, node_colors, transitions_df = comp.build_transitions(dfs)
        
        # Generate Sankey diagrams for each status
        for status in ["Purple", "White", "Green", "Red", "Orange"]:
            st.subheader(f"Sankey Diagram for {status}")
            fig = comp.create_sankey(nodes, sources, targets, values, node_colors, status_filter=status)
            st.plotly_chart(fig, use_container_width=True)

        # PDF Export
        if st.button("Export PDF"):
            pdf_path = comp.export_pdf(fig, filtered_w3)
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f, file_name="campaigns_evolution.pdf", mime="application/pdf")
