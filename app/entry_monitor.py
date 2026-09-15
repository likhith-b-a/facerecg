"""Standalone OpenCV live entry-camera recognition loop.

Usage: python app/entry_monitor.py [--camera-index 0] [--threshold 0.62]

For each detected face: crop -> embed -> match against enrolled staff via
cosine similarity. Draws green/red/orange boxes for AUTHORIZED/DENIED/
UNKNOWN and writes a debounced entry_logs row per decision.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import db
import face_engine

import cv2

COLORS = {
    "AUTHORIZED": (0, 200, 0),
    "DENIED": (0, 0, 255),
    "UNKNOWN": (0, 165, 255),
}


def ensure_snapshot_dirs():
    for decision in ("AUTHORIZED", "DENIED", "UNKNOWN"):
        os.makedirs(os.path.join(config.ENTRY_SNAPSHOTS_DIR, decision), exist_ok=True)


def run(camera_index, threshold, frame_skip, cooldown):
    db.init_db()
    face_engine.load_all_embeddings()
    ensure_snapshot_dirs()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {camera_index}", file=sys.stderr)
        return 1

    # Known limitation (POC scope): a single global "UNKNOWN" bucket means two
    # different unrecognized people in quick succession may only log the first.
    last_logged = {}  # key: staff_id or "UNKNOWN" -> last logged unix ts
    last_faces = []
    frame_count = 0

    print("Entry monitor running. Press q or ESC to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            if frame_count % frame_skip == 0 or not last_faces:
                small = cv2.resize(frame, None,
                                    fx=config.DETECTION_DOWNSCALE, fy=config.DETECTION_DOWNSCALE)
                detections = face_engine.detect_faces(small)
                scale = 1.0 / config.DETECTION_DOWNSCALE

                results = []
                for det in detections:
                    x1, y1, x2, y2 = [int(v * scale) for v in det["box"]]
                    x1, y1 = max(x1, 0), max(y1, 0)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    emb = face_engine.get_embedding(crop)
                    m = face_engine.match(emb, threshold=threshold)
                    results.append((x1, y1, x2, y2, m))
                last_faces = results

            now = time.time()
            for x1, y1, x2, y2, m in last_faces:
                color = COLORS[m.decision]
                label = f"{m.name or 'UNKNOWN'} - {m.decision} ({m.similarity:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                key = m.staff_id if m.staff_id is not None else "UNKNOWN"
                last_ts = last_logged.get(key, 0)
                if now - last_ts >= cooldown:
                    last_logged[key] = now
                    snapshot_path = os.path.join(
                        config.ENTRY_SNAPSHOTS_DIR, m.decision, f"{int(now)}_{key}.jpg",
                    )
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        cv2.imwrite(snapshot_path, crop)
                    else:
                        snapshot_path = None
                    db.insert_entry_log(m.staff_id, m.name, m.similarity, m.decision, snapshot_path)

            cv2.imshow("Entry Monitor - press q to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--threshold", type=float, default=config.MATCH_THRESHOLD)
    parser.add_argument("--frame-skip", type=int, default=config.FRAME_SKIP)
    parser.add_argument("--cooldown", type=float, default=config.LOG_COOLDOWN_SECONDS)
    args = parser.parse_args()

    sys.exit(run(args.camera_index, args.threshold, args.frame_skip, args.cooldown))
