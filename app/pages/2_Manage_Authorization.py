"""Streamlit page: list staff, toggle authorization, delete."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db

import streamlit as st

st.set_page_config(page_title="Manage Authorization", page_icon="✅")
st.title("Manage Authorization")

staff_df = db.list_staff()

if staff_df.empty:
    st.info("No staff registered yet. Use the Register Staff page first.")
else:
    header = st.columns([3, 2, 2, 2, 1])
    header[0].markdown("**Name**")
    header[1].markdown("**Role**")
    header[2].markdown("**Employee code**")
    header[3].markdown("**Authorized**")
    header[4].markdown("**Delete**")

    for _, row in staff_df.iterrows():
        c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 1])
        c1.write(row["name"])
        c2.write(row["role"] or "-")
        c3.write(row["employee_code"] or "-")
        new_val = c4.toggle("", value=bool(row["authorized"]), key=f"auth_{row['id']}")
        if new_val != bool(row["authorized"]):
            db.set_authorized(row["id"], new_val)
            st.rerun()
        if c5.button("Delete", key=f"del_{row['id']}"):
            db.delete_staff(row["id"])
            st.rerun()
