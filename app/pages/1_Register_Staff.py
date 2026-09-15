"""Streamlit page: register a new staff member via webcam burst capture."""
import glob
import os
import subprocess
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import db
import face_engine

import cv2
import streamlit as st

st.set_page_config(page_title="Register Staff", page_icon="\U0001F4DD")
st.title("Register Staff")

if "capture_dir" not in st.session_state:
    st.session_state.capture_dir = None

with st.form("register_form"):
    name = st.text_input("Full name")
    role = st.selectbox("Role", ["Nurse", "Doctor", "Technician", "Admin", "Other"])
    employee_code = st.text_input("Employee code (optional, must be unique)")
    authorized = st.checkbox("Authorized for entry", value=True)
    submitted = st.form_submit_button("Start webcam burst capture")

if submitted:
    if not name.strip():
        st.error("Name is required.")
    else:
        capture_dir = os.path.join(config.ENROLL_SNAPSHOTS_DIR, f"tmp_{uuid.uuid4().hex[:8]}")
        os.makedirs(capture_dir, exist_ok=True)
        script = os.path.join(config.APP_DIR, "capture_utils.py")
        with st.spinner(
            f"Opening webcam window - capturing {config.BURST_CAPTURE_COUNT} photos. "
            "Face the camera with mask + hair cap on. Press 'q' in the window to abort."
        ):
            result = subprocess.run(
                [sys.executable, script, capture_dir,
                 "--count", str(config.BURST_CAPTURE_COUNT),
                 "--interval", str(config.BURST_CAPTURE_INTERVAL_SEC),
                 "--camera-index", str(config.CAMERA_INDEX)],
            )
        if result.returncode != 0:
            st.error("Capture aborted or no photos saved. Try again.")
        else:
            st.session_state.capture_dir = capture_dir
            st.session_state.pending_name = name.strip()
            st.session_state.pending_role = role
            st.session_state.pending_code = employee_code.strip() or None
            st.session_state.pending_authorized = authorized

if st.session_state.capture_dir:
    photos = sorted(glob.glob(os.path.join(st.session_state.capture_dir, "*.jpg")))
    st.subheader(f"Captured {len(photos)} photo(s)")
    cols = st.columns(min(len(photos), 4) or 1)
    for i, p in enumerate(photos):
        cols[i % len(cols)].image(p, width=150)

    col_a, col_b = st.columns(2)
    if col_a.button("Confirm & Save"):
        images = [cv2.imread(p) for p in photos]
        progress_bar = st.progress(0, text="Starting...")
        status_text = st.empty()

        def _on_progress(i, total, status):
            label = {"ok": "detected", "no_face": "no face found", "empty_crop": "bad crop"}[status]
            progress_bar.progress(i / total, text=f"Processing photo {i}/{total} ({label})")
            status_text.write(f"Photo {i}: {label}")

        try:
            staff_id = face_engine.enroll_person(
                st.session_state.pending_name,
                st.session_state.pending_role,
                images,
                employee_code=st.session_state.pending_code,
                authorized=st.session_state.pending_authorized,
                progress_callback=_on_progress,
            )
            progress_bar.progress(1.0, text="Done")
            st.success(f"Registered '{st.session_state.pending_name}' as staff id {staff_id}.")
            st.session_state.capture_dir = None
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Enrollment failed: {e}")

    if col_b.button("Discard & Retake"):
        st.session_state.capture_dir = None
        st.rerun()
