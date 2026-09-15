"""Standalone OpenCV live entry-camera recognition loop.

Usage: python app/entry_monitor.py [--camera-index 0] [--threshold 0.62]

For each detected face: crop -> embed -> match against enrolled staff via
cosine similarity. Draws green/red/orange boxes for AUTHORIZED/DENIED/
UNKNOWN and writes a debounced entry_logs row per decision.

Detection runs on a background thread so the preview keeps rendering at
camera FPS instead of freezing for each (slow, CPU-bound) detect+embed
pass. The main loop always draws the most recently finished result.
"""
import argparse
import os
import sys
import threading
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


class _DetectionWorker:
    """Runs detect+embed+match on a background thread against whatever
    frame was captured most recently, so the main loop never blocks on it.
    """

    def __init__(self, threshold):
        self.threshold = threshold
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._latest_frame_id = 0
        self._results_lock = threading.Lock()
        self.last_faces = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def submit_frame(self, frame):
        with self._frame_lock:
            self._latest_frame = frame
            self._latest_frame_id += 1

    def get_results(self):
        with self._results_lock:
            return self.last_faces

    def _loop(self):
        processed_id = 0
        while not self._stop.is_set():
            with self._frame_lock:
                frame, frame_id = self._latest_frame, self._latest_frame_id
            if frame is None or frame_id == processed_id:
                time.sleep(0.01)
                continue
            processed_id = frame_id

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
                m = face_engine.match(emb, threshold=self.threshold)
                results.append((x1, y1, x2, y2, m))

            with self._results_lock:
                self.last_faces = results


def run(camera_index, threshold, cooldown):
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
    worker = _DetectionWorker(threshold)
    worker.start()

    print("Entry monitor running. Press q or ESC to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            worker.submit_frame(frame)
            last_faces = worker.get_results()

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
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--threshold", type=float, default=config.MATCH_THRESHOLD)
    parser.add_argument("--cooldown", type=float, default=config.LOG_COOLDOWN_SECONDS)
    args = parser.parse_args()

    sys.exit(run(args.camera_index, args.threshold, args.cooldown))
