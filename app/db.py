"""SQLite storage layer: staff, embeddings, entry_logs.

Streamlit pages and entry_monitor.py are separate processes hitting the same
DB file, so WAL mode is enabled on every connection.
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def get_connection():
    os.makedirs(config.DATA_STORE_DIR, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(config.DATA_STORE_DIR, exist_ok=True)
    conn = get_connection()
    with open(config.SCHEMA_PATH, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()


def insert_staff(name, role, employee_code=None, authorized=True):
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO staff (name, role, employee_code, authorized) VALUES (?, ?, ?, ?)",
        (name, role, employee_code, 1 if authorized else 0),
    )
    conn.commit()
    staff_id = cur.lastrowid
    conn.close()
    return staff_id


def set_authorized(staff_id, authorized):
    conn = get_connection()
    conn.execute(
        "UPDATE staff SET authorized = ? WHERE id = ?",
        (1 if authorized else 0, staff_id),
    )
    conn.commit()
    conn.close()


def list_staff(include_inactive=True):
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM staff ORDER BY created_at DESC", conn)
    conn.close()
    if not include_inactive:
        df = df[df["authorized"] == 1]
    return df


def delete_staff(staff_id):
    conn = get_connection()
    conn.execute("DELETE FROM staff WHERE id = ?", (staff_id,))
    conn.commit()
    conn.close()


def insert_embedding(staff_id, embedding, source_image_path=None):
    conn = get_connection()
    vec = np.asarray(embedding, dtype=np.float32)
    conn.execute(
        "INSERT INTO embeddings (staff_id, embedding, dim, source_image_path) "
        "VALUES (?, ?, ?, ?)",
        (staff_id, vec.tobytes(), vec.shape[0], source_image_path),
    )
    conn.commit()
    conn.close()


def get_all_embeddings():
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT e.id, e.staff_id, e.embedding, e.dim, s.name, s.authorized "
        "FROM embeddings e JOIN staff s ON s.id = e.staff_id",
        conn,
    )
    conn.close()
    return df


def insert_entry_log(staff_id, matched_name, similarity, decision, snapshot_path=None):
    conn = get_connection()
    conn.execute(
        "INSERT INTO entry_logs (staff_id, matched_name, similarity, decision, snapshot_path) "
        "VALUES (?, ?, ?, ?, ?)",
        (staff_id, matched_name, similarity, decision, snapshot_path),
    )
    conn.commit()
    conn.close()


def query_entry_logs(start=None, end=None, decision=None, name_contains=None, limit=500):
    conn = get_connection()
    clauses = []
    params = []

    if start:
        clauses.append("ts >= ?")
        params.append(start)
    if end:
        clauses.append("ts <= ?")
        params.append(end)
    if decision:
        if isinstance(decision, (list, tuple, set)):
            decision = list(decision)
            placeholders = ",".join("?" for _ in decision)
            clauses.append(f"decision IN ({placeholders})")
            params.extend(decision)
        else:
            clauses.append("decision = ?")
            params.append(decision)
    if name_contains:
        clauses.append("matched_name LIKE ?")
        params.append(f"%{name_contains}%")

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM entry_logs {where} ORDER BY ts DESC LIMIT ?"
    params.append(limit)

    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df
