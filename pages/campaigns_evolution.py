# pages/campaigns_evolution.py
# -*- coding: utf-8 -*-
"""
Campaigns Evolution Analysis (Single-File Page)

This Streamlit page lets users upload three weekly files (W1, W2, W3),
match campaign names with fuzzy matching (90%), compute status evolution,
render an interactive Sankey diagram (Plotly), and filter campaigns that end
in Purple/White on W3. It also exports a PDF (fpdf2) with a static Sankey image
(Kaleido) + the final filtered table.

Author: Share It Studio
"""

from __future__ import annotations

import io
import logging
import re
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- Optional fuzzy libraries (prefer rapidfuzz if available, else fuzzywuzzy) ---
_FUZZ_BACKEND = None
try:
    from rapidfuzz import fuzz, process  # type: ignore
    _FUZZ_BACKEND = "rapidfuzz"
except Exception:  # pragma: no cover
    try:
        # FuzzyWuzzy is slower and deprecated in favor of RapidFuzz, but kept per requirement.
        from fuzzywuzzy import fuzz, process  # type: ignore
        _FUZZ_BACKEND = "fuzzywuzzy"
    except Exception:
        _FUZZ_BACKEND = None  # We'll handle gracefully in UI.

# --- PDF / Image export deps ---
try:
    from fpdf import FPDF  # type: ignore
except Exception:  # pragma: no cover
    FPDF = None  # Handled in UI

try:
    import plotly.io as pio  # type: ignore
    _KALEIDO_AVAILABLE = True
except Exception:  # pragma: no cover
    pio = None
    _KALEIDO_AVAILABLE = False


# =============================================================================
# Constants & Config
# =============================================================================

PAGE_TITLE = "Campaigns Evolution Analysis"
SHEET_NAME = "Sponsored Products Campaigns"
COL_CAMPAIGN = "Campaign Name (Informational only)"
COL_STATUS = "Status"

VALID_STATUSES = ["white", "green", "orange", "red", "purple", "new_reactivated", "not_present"]
BASE_STATUSES = ["white", "green", "orange", "red", "purple", "new_reactivated"]  # excluding not_present by default

DEFAULT_FUZZ_THRESHOLD = 90

STATUS_COLOR_MAP = {
    "white": "#BFC7D5",
    "green": "#34A853",
    "orange": "#FBBC05",
    "red": "#EA4335",
    "purple": "#A142F4",
    "new_reactivated": "#0AA1DD",
    "not_present": "#9E9E9E",
}

WEEK_LABELS = ["W1", "W2", "W3"]


# =============================================================================
# Logging setup
# =============================================================================

def _setup_logging() -> None:
    """Configure logging once per session."""
    if "log_configured" not in st.session_state:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )
        st.session_state["log_configured"] = True


# =============================================================================
# Data Models
# =============================================================================

@dataclass(frozen=True)
class WeeklyData:
    week_label: str
    df: pd.DataFrame  # Must contain COL_CAMPAIGN and COL_STATUS (after normalization)


@dataclass
class MatchResult:
    observed_name: str
    canonical_name: str
    score: float


# =============================================================================
# Utilities
# =============================================================================

def normalize_whitespace(text: str) -> str:
    """Collapse/strip whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_campaign_name(name: str) -> str:
    """Normalize campaign names for fuzzy matching."""
    s = normalize_whitespace(name or "")
    s = s.lower()
    # Optional: remove trivial suffixes like "(v2)", "- copy", trailing hyphens/spaces.
    s = re.sub(r"\((v\d+|copy)\)$", "", s).strip(" -_")
    return s


def normalize_status(raw: Optional[str]) -> str:
    """Normalize status to one of the allowed values."""
    if raw is None:
        return "white"
    s = str(raw).strip()
    if s == "":
        return "white"
    s_low = s.lower()
    # Accepted labels
    if s_low in {"white", "green", "orange", "red", "purple", "new_reactivated", "new/reactivated"}:
        return "new_reactivated" if "reactivated" in s_low else s_low
    # Try mapping common variants
    synonyms = {
        "blank": "white",
        "": "white",
    }
    mapped = synonyms.get(s_low)
    if mapped:
        return mapped
    logging.warning("Unknown status '%s' mapped to 'white'.", s)
    return "white"


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


@st.cache_data(show_spinner=False)
def load_weekly_from_upload(file, week_label: str) -> WeeklyData:
    """
    Load a weekly file (xlsx/csv). For Excel, read the specific sheet.
    Normalize essential columns.
    """
    if file is None:
        raise ValueError("No file provided.")

    filename = getattr(file, "name", "uploaded")
    is_excel = filename.lower().endswith((".xlsx", ".xls"))
    try:
        if is_excel:
            # Read sheet explicitly
            df = pd.read_excel(file, sheet_name=SHEET_NAME, engine=None)
        else:
            # Fallback: CSV
            df = pd.read_csv(file)
    except ValueError as exc:
        # Possibly missing sheet
        raise ValueError(
            f"Failed reading '{filename}'. Ensure it contains sheet '{SHEET_NAME}' "
            f"(for Excel) or upload a CSV with the expected columns."
        ) from exc
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Failed reading '{filename}': {exc}") from exc

    # Column normalization
    cols_map = {
        COL_CAMPAIGN: COL_CAMPAIGN,
        COL_STATUS: COL_STATUS,
    }
    # Try to tolerate casing / minor variations
    lower_cols = {c.lower(): c for c in df.columns}
    for required in [COL_CAMPAIGN, COL_STATUS]:
        if required not in df.columns:
            # try to find case-insensitive match
            candidate = lower_cols.get(required.lower())
            if candidate:
                cols_map[candidate] = required

    # Rename if needed
    df = df.rename(columns=cols_map)

    ensure_columns(df, [COL_CAMPAIGN, COL_STATUS])

    # Keep only needed columns to stay lean
    df = df[[COL_CAMPAIGN, COL_STATUS]].copy()

    # Normalize fields
    df[COL_CAMPAIGN] = df[COL_CAMPAIGN].astype(str).map(normalize_whitespace)
    df[COL_STATUS] = df[COL_STATUS].map(normalize_status)

    # Drop empty campaign names (if any)
    df = df[df[COL_CAMPAIGN].str.len() > 0].reset_index(drop=True)

    logging.info("Loaded %s with %d rows.", week_label, len(df))
    return WeeklyData(week_label=week_label, df=df)


# =============================================================================
# Fuzzy Matching
# =============================================================================

class CampaignMatcher:
    """
    Build a canonical catalog of campaign names across weeks using fuzzy matching.
    Tie-breaker: prefer highest score; if tie, prefer the most recent week (W3 > W2 > W1).
    """

    def __init__(self, threshold: int = DEFAULT_FUZZ_THRESHOLD):
        if _FUZZ_BACKEND is None:
            raise RuntimeError(
                "No fuzzy matching backend available. Install 'rapidfuzz' (recommended) or 'fuzzywuzzy'."
            )
        self.threshold = int(threshold)
        self._canonical: List[str] = []           # canonical display names
        self._canonical_norm: List[str] = []      # normalized canonical names
        self._norm_to_display: Dict[str, str] = {}
        self._matches: List[MatchResult] = []     # diagnostics

    @property
    def diagnostics(self) -> pd.DataFrame:
        return pd.DataFrame([m.__dict__ for m in self._matches])

    def _best_match(self, norm_name: str) -> Tuple[Optional[int], float]:
        """Return (index, score) of best canonical match above threshold, else (None, 0)."""
        if not self._canonical_norm:
            return (None, 0.0)
        # Use process from the selected backend
        choices = self._canonical_norm
        # Use token_sort_ratio to be resilient to word order / suffixes.
        result = process.extractOne(
            norm_name,
            choices,
            scorer=fuzz.token_sort_ratio,  # works in both backends
            score_cutoff=self.threshold,
        )
        if result is None:
            return (None, 0.0)
        # result -> (matched_string, score, index)
        # rapidfuzz: (choice, score, index)
        # fuzzywuzzy: (choice, score) (no index) -> need to find index
        if len(result) == 3:
            _, score, index = result
        else:
            matched_str, score = result  # type: ignore
            index = choices.index(matched_str)  # O(n), fine for small lists
        return (int(index), float(score))

    def _add_canonical(self, display_name: str) -> int:
        norm = normalize_campaign_name(display_name)
        self._canonical.append(display_name)
        self._canonical_norm.append(norm)
        self._norm_to_display[norm] = display_name
        return len(self._canonical) - 1

    def fit_transform_weeks(self, weeks: List[WeeklyData]) -> Dict[str, Dict[str, str]]:
        """
        Build canonical mapping and return a dict:
        {
          "W1": {observed_name -> canonical_name},
          "W2": {observed_name -> canonical_name},
          "W3": {observed_name -> canonical_name},
        }
        """
        mapping: Dict[str, Dict[str, str]] = {w.week_label: {} for w in weeks}

        # Process in chronological order (W1 -> W3)
        for w in weeks:
            for observed in w.df[COL_CAMPAIGN].astype(str):
                norm_obs = normalize_campaign_name(observed)
                idx, score = self._best_match(norm_obs)
                if idx is None:
                    # New canonical entry
                    idx = self._add_canonical(observed)
                    canonical = self._canonical[idx]
                    self._matches.append(MatchResult(observed_name=observed, canonical_name=canonical, score=100.0))
                else:
                    canonical = self._canonical[idx]
                    self._matches.append(MatchResult(observed_name=observed, canonical_name=canonical, score=score))
                mapping[w.week_label][observed] = canonical

        # Optional post-pass to prefer most recent display name when multiple variants map to same canonical
        # (We keep the original canonical display chosen at first appearance for simplicity.)

        return mapping


# =============================================================================
# Transformations (Statuses & Transitions)
# =============================================================================

def build_long_panel(
    weeks: List[WeeklyData],
    mapping: Dict[str, Dict[str, str]],
    include_not_present: bool = False,
) -> pd.DataFrame:
    """
    Construct a long panel with columns:
      canonical_name, week, status, present
    Apply 'new_reactivated' when a campaign appears for the first time in W2/W3.
    Optionally include rows for 'not_present'.
    """
    # Collect all canonical names across weeks
    all_canonical = set()
    for w in weeks:
        all_canonical.update(mapping[w.week_label].values())
    all_canonical = sorted(all_canonical)

    # Build per-week lookup: canonical -> status (from df)
    per_week_status: Dict[str, Dict[str, str]] = {w.week_label: {} for w in weeks}
    for w in weeks:
        for _, row in w.df.iterrows():
            obs = row[COL_CAMPAIGN]
            status = row[COL_STATUS]
            canonical = mapping[w.week_label][obs]
            per_week_status[w.week_label][canonical] = normalize_status(status)

    # Determine first appearance week for each canonical
    first_week_idx: Dict[str, int] = {}
    for name in all_canonical:
        idx = 999
        for i, wl in enumerate(WEEK_LABELS):
            if name in per_week_status.get(wl, {}):
                idx = i
                break
        first_week_idx[name] = idx

    records: List[Dict[str, object]] = []
    for name in all_canonical:
        for i, wl in enumerate(WEEK_LABELS):
            present = name in per_week_status.get(wl, {})
            if present:
                # If first appearance is this week and it's not W1, override with new_reactivated
                if first_week_idx[name] == i and i > 0:
                    status = "new_reactivated"
                else:
                    status = per_week_status[wl][name]
            else:
                if include_not_present:
                    status = "not_present"
                else:
                    # Skip not-present rows when not including
                    continue
            records.append(
                {
                    "canonical_name": name,
                    "week": wl,
                    "status": status,
                    "present": bool(present),
                }
            )

    panel = pd.DataFrame.from_records(records)
    return panel


def compute_sankey_links(panel: pd.DataFrame) -> Tuple[List[str], List[int], List[int], List[float], List[str]]:
    """
    From the long panel, compute Sankey nodes and links between W1->W2 and W2->W3.
    Returns: (labels, sources, targets, values, colors)
    """
    # Build node labels (unique combinations of week-status) in a stable order
    # We preserve the BASE_STATUSES order for readability; include 'not_present' only if present.
    statuses_in_data = sorted(panel["status"].unique().tolist(), key=lambda s: (s not in BASE_STATUSES, BASE_STATUSES.index(s) if s in BASE_STATUSES else 99))
    weeks = WEEK_LABELS

    node_labels: List[str] = []
    node_colors: List[str] = []
    node_index: Dict[Tuple[str, str], int] = {}

    for wl in weeks:
        for st_ in statuses_in_data:
            node = (wl, st_)
            node_labels.append(f"{wl}-{st_.replace('_', ' ').title()}")
            node_colors.append(STATUS_COLOR_MAP.get(st_, "#CCCCCC"))
            node_index[node] = len(node_labels) - 1

    # Compute transitions between consecutive weeks
    links_counter: Dict[Tuple[int, int], int] = {}

    def status_of(name: str, wl: str) -> Optional[str]:
        rows = panel[(panel["canonical_name"] == name) & (panel["week"] == wl)]
        if rows.empty:
            return None
        return str(rows["status"].iloc[0])

    for name in panel["canonical_name"].unique():
        for (w_from, w_to) in [("W1", "W2"), ("W2", "W3")]:
            s_from = status_of(name, w_from)
            s_to = status_of(name, w_to)
            if s_from is None or s_to is None:
                continue  # skip incomplete paths when not including 'not_present'
            src_idx = node_index[(w_from, s_from)]
            tgt_idx = node_index[(w_to, s_to)]
            links_counter[(src_idx, tgt_idx)] = links_counter.get((src_idx, tgt_idx), 0) + 1

    sources, targets, values = [], [], []
    for (src, tgt), val in links_counter.items():
        sources.append(src)
        targets.append(tgt)
        values.append(float(val))

    return node_labels, sources, targets, values, node_colors


def final_filtered_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Return campaigns with W3 status in {Purple, White}. Also mark 'always Purple/White' across W1..W3."""
    # Pivot to wide: one row per campaign, columns W1/W2/W3 statuses (missing -> None)
    wide = panel.pivot_table(index="canonical_name", columns="week", values="status", aggfunc="first")
    # Ensure all weeks exist
    for wl in WEEK_LABELS:
        if wl not in wide.columns:
            wide[wl] = np.nan
    wide = wide[WEEK_LABELS].reset_index()

    def in_target(status: Optional[str]) -> bool:
        if status is None or (isinstance(status, float) and np.isnan(status)):
            return False
        return status in {"purple", "white"}

    wide["in_W3_target"] = wide["W3"].apply(in_target)
    wide["always_purple_white"] = wide[["W1", "W2", "W3"]].apply(
        lambda row: all(in_target(v) for v in row.values.tolist()), axis=1
    )

    filtered = wide[wide["in_W3_target"]].drop(columns=["in_W3_target"])
    # Rename columns for display
    filtered = filtered.rename(
        columns={
            "canonical_name": "Campaign",
            "W1": "W1 Status",
            "W2": "W2 Status",
            "W3": "W3 Status",
            "always_purple_white": "Always Purple/White (W1-W3)",
        }
    ).reset_index(drop=True)

    # Title-case statuses for readability
    for c in ["W1 Status", "W2 Status", "W3 Status"]:
        if c in filtered.columns:
            filtered[c] = filtered[c].fillna("").map(lambda s: s.replace("_", " ").title() if isinstance(s, str) else s)

    return filtered


# =============================================================================
# Visualization
# =============================================================================

def render_sankey(labels: List[str], sources: List[int], targets: List[int], values: List[float], node_colors: List[str]) -> go.Figure:
    """Build a Plotly Sankey figure."""
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=18,
                    thickness=16,
                    line=dict(width=0.5, color="rgba(0,0,0,0.35)"),
                    label=labels,
                    color=node_colors,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    hovertemplate="From %{source.label}<br>To %{target.label}<br>Count: %{value}<extra></extra>",
                ),
                arrangement="snap",
            )
        ]
    )
    fig.update_layout(
        title_text="Campaign Status Evolution (W1 → W3)",
        font=dict(size=12),
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x",
    )
    return fig


# =============================================================================
# PDF Export
# =============================================================================

def generate_pdf_report(fig: go.Figure, table_df: pd.DataFrame) -> bytes:
    """
    Generate a PDF with:
      - Static Sankey image (exported via Kaleido)
      - Final filtered table
    Returns PDF bytes.
    """
    if FPDF is None:
        raise RuntimeError("fpdf2 is not installed. Please install 'fpdf2' to export PDF.")
    if not _KALEIDO_AVAILABLE:
        raise RuntimeError("Kaleido is not available for Plotly static image export. Install 'kaleido'.")

    # Export figure to PNG in-memory
    img_bytes = fig.to_image(format="png", scale=2)  # requires kaleido
    img_stream = io.BytesIO(img_bytes)

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Campaigns Evolution - Overview", ln=1, align="L")

    # Add image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(img_stream.getvalue())
        tmp.flush()
        # Fit image width to page (keeping aspect ratio)
        page_w = pdf.w - 2 * pdf.l_margin
        pdf.image(tmp.name, x=pdf.l_margin, y=None, w=page_w)

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Final Filtered Campaigns (W3 in Purple/White)", ln=1)

    # Table header
    pdf.set_font("Helvetica", "B", 10)
    headers = list(table_df.columns)
    col_widths = _compute_col_widths_for_pdf(pdf, headers, table_df)

    for h, w in zip(headers, col_widths):
        pdf.cell(w, 8, h, border=1, align="L")
    pdf.ln(8)

    # Table rows
    pdf.set_font("Helvetica", "", 9)
    for _, row in table_df.iterrows():
        for h, w in zip(headers, col_widths):
            txt = str(row[h]) if not pd.isna(row[h]) else ""
            # Truncate long text for simplicity; fpdf2 multicell could be used for wrapping.
            if len(txt) > 60:
                txt = txt[:57] + "..."
            pdf.cell(w, 7, txt, border=1, align="L")
        pdf.ln(7)

    return pdf.output(dest="S").encode("latin-1")


def _compute_col_widths_for_pdf(pdf: "FPDF", headers: List[str], table_df: pd.DataFrame) -> List[float]:
    """Compute simple proportional column widths based on header + sample rows."""
    # Basic heuristic: allocate more width to 'Campaign'
    base_w = (pdf.w - 2 * pdf.l_margin)
    widths: Dict[str, float] = {h: 1.0 for h in headers}
    if "Campaign" in headers:
        widths["Campaign"] = 2.5  # weighted wider
    total_weight = sum(widths.values())
    return [base_w * (widths[h] / total_weight) for h in headers]


# =============================================================================
# UI Helpers
# =============================================================================

def preview_weekly(w: WeeklyData) -> None:
    """Show per-week preview: counts by status and campaign list."""
    st.subheader(f"{w.week_label} Preview")
    with st.expander(f"View campaigns ({w.week_label})", expanded=False):
        st.dataframe(w.df.rename(columns={COL_CAMPAIGN: "Campaign", COL_STATUS: "Status"}), use_container_width=True)

    counts = (
        w.df.assign(Status=w.df[COL_STATUS].map(lambda s: s.replace("_", " ").title()))
        .groupby("Status")
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )
    c1, c2 = st.columns((1, 1))
    with c1:
        st.markdown("**Status distribution**")
        st.dataframe(counts, use_container_width=True)
    with c2:
        st.markdown("**Quick bar chart**")
        st.bar_chart(counts.set_index("Status"))


def diagnostics_section(matcher: Optional[CampaignMatcher]) -> None:
    """Optional diagnostics to inspect fuzzy matches."""
    if matcher is None:
        return
    with st.expander("Diagnostics (Fuzzy Matching)", expanded=False):
        st.caption(f"Backend: **{_FUZZ_BACKEND}**, threshold: **{matcher.threshold}**")
        diag = matcher.diagnostics.sort_values(["canonical_name", "score"], ascending=[True, False])
        st.dataframe(diag, use_container_width=True)


# =============================================================================
# Main Page
# =============================================================================

def main() -> None:
    """Streamlit entrypoint for the Campaigns Evolution page."""
    _setup_logging()
    st.title(PAGE_TITLE)
    st.write(
        "Upload **Weekly 1**, **Weekly 2**, and **Weekly 3** files from the "
        f"sheet **'{SHEET_NAME}'** with columns **'{COL_CAMPAIGN}'** and **'{COL_STATUS}'**. "
        "We will fuzzy-match campaigns (90%), compute evolution, render an interactive Sankey, "
        "and filter campaigns that end in **Purple** or **White** in W3."
    )

    # Sidebar controls
    with st.sidebar:
        st.header("Analysis Options")
        fuzz_threshold = st.slider("Fuzzy matching threshold", 80, 100, DEFAULT_FUZZ_THRESHOLD, step=1)
        include_not_present = st.checkbox("Include 'not_present' nodes in Sankey", value=False)
        show_diagnostics = st.checkbox("Show diagnostics", value=False)
        st.caption(
            f"Fuzzy backend detected: **{_FUZZ_BACKEND or 'None'}**. "
            "Install `rapidfuzz` (recommended) or `fuzzywuzzy`."
        )

    # Uploaders
    col1, col2, col3 = st.columns(3)
    with col1:
        up_w1 = st.file_uploader("Upload Weekly 1 (oldest)", type=["xlsx", "xls", "csv"], key="w1")
    with col2:
        up_w2 = st.file_uploader("Upload Weekly 2", type=["xlsx", "xls", "csv"], key="w2")
    with col3:
        up_w3 = st.file_uploader("Upload Weekly 3 (latest)", type=["xlsx", "xls", "csv"], key="w3")

    # Load each weekly and preview
    weekly_objects: List[WeeklyData] = []
    try:
        if up_w1:
            w1 = load_weekly_from_upload(up_w1, "W1")
            preview_weekly(w1)
            weekly_objects.append(w1)
        if up_w2:
            w2 = load_weekly_from_upload(up_w2, "W2")
            preview_weekly(w2)
            weekly_objects.append(w2)
        if up_w3:
            w3 = load_weekly_from_upload(up_w3, "W3")
            preview_weekly(w3)
            weekly_objects.append(w3)
    except Exception as exc:
        st.error(f"Error while loading files: {exc}")
        return

    # Proceed only if all three are present
    if len(weekly_objects) != 3:
        st.info("Please upload all three weeklies to run the evolution analysis.")
        return

    # Sort in chronological order just in case
    weekly_objects = sorted(weekly_objects, key=lambda w: WEEK_LABELS.index(w.week_label))

    # Fuzzy matching
    matcher: Optional[CampaignMatcher] = None
    try:
        matcher = CampaignMatcher(threshold=fuzz_threshold)
        mapping = matcher.fit_transform_weeks(weekly_objects)
    except Exception as exc:
        st.error(f"Fuzzy matching is unavailable: {exc}")
        return

    if show_diagnostics:
        diagnostics_section(matcher)

    # Build long panel (apply new_reactivated logic, optionally include not_present)
    panel = build_long_panel(weekly_objects, mapping, include_not_present=include_not_present)

    # Compute Sankey links
    labels, sources, targets, values, node_colors = compute_sankey_links(panel)
    if not values:
        st.warning("No transitions to show. Check the inputs or enable 'Include not_present nodes'.")
        return

    # Render Sankey
    st.subheader("Interactive Sankey (W1 → W3)")
    sankey_fig = render_sankey(labels, sources, targets, values, node_colors)
    st.plotly_chart(sankey_fig, use_container_width=True, theme="streamlit")

    # Final filtered table (W3 in Purple/White)
    st.subheader("Final Filter: Campaigns in Purple/White on W3")
    filtered = final_filtered_table(panel)
    show_only_always = st.checkbox("Show only campaigns that were always Purple/White across W1–W3", value=False)
    if show_only_always and "Always Purple/White (W1-W3)" in filtered.columns:
        filtered_display = filtered[filtered["Always Purple/White (W1-W3)"] == True]  # noqa: E712
    else:
        filtered_display = filtered.copy()

    st.dataframe(filtered_display, use_container_width=True)

    # Download PDF report (Sankey static + final filtered table)
    st.subheader("Export")
    try:
        pdf_bytes = generate_pdf_report(sankey_fig, filtered_display)
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="campaigns_evolution_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as exc:
        st.warning(f"PDF export not available: {exc}")
        st.caption("To enable PDF: install `fpdf2` and `kaleido`.")


# Run directly (for local testing)
if __name__ == "__main__":  # pragma: no cover
    import streamlit.web.bootstrap as bootstrap

    # When run as a script, mount as a minimal Streamlit app
    def _standalone():
        st.set_page_config(page_title=PAGE_TITLE, page_icon="🔄", layout="wide")
        main()

    bootstrap.run(_standalone, "", [], {})

