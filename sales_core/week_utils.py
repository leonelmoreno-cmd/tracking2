from __future__ import annotations
import pandas as pd
import streamlit as st
from .config import TIMEZONE

@st.cache_data(show_spinner=False)
def now_ny() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TIMEZONE)

def last_finished_thursday(ref: pd.Timestamp | None = None) -> pd.Timestamp:
    """
    Most recent *finished* Thursday (00:00 local).
    If today is Thursday, we use the *previous* Thursday instead
    (to avoid including incomplete data for the current day).
    """
    if ref is None:
        ref = now_ny()
    ref = ref.tz_convert(TIMEZONE).normalize()
    # Mon=0..Thu=3..Sun=6
    w = ref.weekday()
    delta_to_thu = (w - 3)  # days since Thu

    if delta_to_thu == 0:
        # Today is Thursday → go back one full week
        return ref - pd.Timedelta(days=7)
    elif delta_to_thu > 0:
        # Friday to Sunday → go back to this week's Thursday
        return ref - pd.Timedelta(days=delta_to_thu)
    else:
        # Monday to Wednesday → go back to last week's Thursday
        return ref - pd.Timedelta(days=(7 + delta_to_thu))

def week_end_for_date(d: pd.Timestamp) -> pd.Timestamp:
    """Return the Thursday ending the Fri→Thu week for date d (naive date)."""
    d = pd.to_datetime(d).tz_localize(None)
    delta = (3 - d.weekday()) % 7
    return (d + pd.Timedelta(days=delta)).normalize()

def week_label_iso(week_end: pd.Timestamp) -> int:
    return int(week_end.isocalendar().week)

def four_full_weeks_window(today_ny: pd.Timestamp | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Inclusive window covering last 4 full Fri→Thu weeks,
    ending at the last finished Thursday.
    """
    lt = last_finished_thursday(today_ny)
    start = lt - pd.Timedelta(days=27)
    return start, lt
