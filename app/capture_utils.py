"""Standalone OpenCV burst-capture window.

Launched as a subprocess from the Streamlit registration page, since
st.camera_input is single-shot per rerun and too clunky for an 8-photo
burst with a live preview + countdown.

Usage: python capture_utils.py <out_dir> [--count 8] [--interval 1.2] [--camera-index 0]
"""
import argparse
import os
import sys
import time

import cv2


def run_burst_capture(out_dir, count, interval_sec, camera_index):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {camera_index}", file=sys.stderr)
        return 1

    saved = 0
    next_capture_at = time.time() + 2.0  # initial warm-up delay so first frame isn't a blink
    window = "Registration Capture - press q to abort"

    try:
        while saved < count:
            ok, frame = cap.read()
            if not ok:
                continue

            remaining = next_capture_at - time.time()
            display = frame.copy()
            status = f"Next photo in {remaining:0.1f}s" if remaining > 0 else "Capturing..."
            cv2.putText(display, f"{status}  ({saved}/{count})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(display, "Face camera, mask+cap on. Press q to abort.", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.imshow(window, display)

            if remaining <= 0:
                path = os.path.join(out_dir, f"capture_{saved:02d}.jpg")
                cv2.imwrite(path, frame)
                saved += 1
                next_capture_at = time.time() + interval_sec

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Saved {saved} photo(s) to {out_dir}")
    return 0 if saved > 0 else 1


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config

    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--count", type=int, default=config.BURST_CAPTURE_COUNT)
    parser.add_argument("--interval", type=float, default=config.BURST_CAPTURE_INTERVAL_SEC)
    parser.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX)
    args = parser.parse_args()

    sys.exit(run_burst_capture(args.out_dir, args.count, args.interval, args.camera_index))
