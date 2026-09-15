"""Streamlit entrypoint. Run with: streamlit run app/Home.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db
import face_engine

import streamlit as st

st.set_page_config(page_title="Staff Entry Recognition", page_icon="\U0001FA7A", layout="wide")

db.init_db()


@st.cache_resource(show_spinner="Loading face detection + recognition models (one-time)...")
def _warm_up_models():
    face_engine.warm_up()
    return True


_warm_up_models()

st.title("Masked-Face Staff Entry Recognition - POC")
st.markdown(
    """
Use the sidebar to:
- **Register Staff** - enroll a new staff member via webcam burst capture (mask + hair cap on).
- **Manage Authorization** - authorize/deauthorize or remove staff.
- **Entry Logs** - review entry attempts recorded by `entry_monitor.py`.

Run the live entry camera separately:
```
python app/entry_monitor.py
```
"""
)

staff_df = db.list_staff()
col1, col2, col3 = st.columns(3)
col1.metric("Registered staff", len(staff_df))
col2.metric("Authorized", int(staff_df["authorized"].sum()) if not staff_df.empty else 0)
col3.metric("Deauthorized", int((staff_df["authorized"] == 0).sum()) if not staff_df.empty else 0)
