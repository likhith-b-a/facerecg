"""Streamlit page: filterable entry log table + metrics."""
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db

import streamlit as st

st.set_page_config(page_title="Entry Logs", page_icon="\U0001F4CB", layout="wide")
st.title("Entry Logs")

col1, col2, col3 = st.columns(3)
today = datetime.date.today()
start_date = col1.date_input("From", value=today - datetime.timedelta(days=7))
end_date = col2.date_input("To", value=today)
decisions = col3.multiselect(
    "Decision", ["AUTHORIZED", "DENIED", "UNKNOWN"],
    default=["AUTHORIZED", "DENIED", "UNKNOWN"],
)
name_filter = st.text_input("Filter by name contains")

start_ts = f"{start_date.isoformat()} 00:00:00"
end_ts = f"{end_date.isoformat()} 23:59:59"

logs_df = db.query_entry_logs(
    start=start_ts, end=end_ts,
    decision=decisions if decisions else None,
    name_contains=name_filter or None,
    limit=500,
)

m1, m2, m3 = st.columns(3)
m1.metric("Authorized", int((logs_df["decision"] == "AUTHORIZED").sum()) if not logs_df.empty else 0)
m2.metric("Denied", int((logs_df["decision"] == "DENIED").sum()) if not logs_df.empty else 0)
m3.metric("Unknown", int((logs_df["decision"] == "UNKNOWN").sum()) if not logs_df.empty else 0)

st.dataframe(logs_df, use_container_width=True)

if not logs_df.empty:
    selected = st.selectbox("Preview snapshot for row id", [None] + logs_df["id"].tolist())
    if selected:
        snap = logs_df.loc[logs_df["id"] == selected, "snapshot_path"].iloc[0]
        if snap and os.path.exists(snap):
            st.image(snap, width=300)
        else:
            st.caption("No snapshot saved for this row.")
