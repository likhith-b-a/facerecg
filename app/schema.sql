CREATE TABLE IF NOT EXISTS staff (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT NOT NULL,
    role           TEXT,
    employee_code  TEXT UNIQUE,
    authorized     INTEGER NOT NULL DEFAULT 1,
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS embeddings (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    staff_id           INTEGER NOT NULL REFERENCES staff(id) ON DELETE CASCADE,
    embedding          BLOB NOT NULL,   -- float32 ndarray.tobytes(), 512-d
    dim                INTEGER NOT NULL DEFAULT 512,
    source_image_path  TEXT,
    created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_embeddings_staff_id ON embeddings(staff_id);

CREATE TABLE IF NOT EXISTS entry_logs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT NOT NULL DEFAULT (datetime('now')),
    staff_id       INTEGER REFERENCES staff(id),  -- NULL if unknown
    matched_name   TEXT,                           -- denormalized, survives staff deletion
    similarity     REAL,
    decision       TEXT NOT NULL,                  -- AUTHORIZED | DENIED | UNKNOWN
    snapshot_path  TEXT
);
CREATE INDEX IF NOT EXISTS idx_entry_logs_ts ON entry_logs(ts);
