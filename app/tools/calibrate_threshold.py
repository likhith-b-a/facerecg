"""Genuine vs impostor cosine-similarity calibration.

Usage: python app/tools/calibrate_threshold.py

Computes pairwise similarity between all stored embeddings, splits into
same-person (genuine) and different-person (impostor) pairs, and prints
stats to help pick a real MATCH_THRESHOLD for masked/capped faces.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import face_engine

import numpy as np


def main():
    matrix, staff_ids = face_engine.load_all_embeddings()
    n = matrix.shape[0]
    if n < 2:
        print("Need at least 2 stored embeddings to calibrate. Register more staff first.")
        return 1

    sims = matrix @ matrix.T  # (N,N) cosine similarity, embeddings are L2-normalized
    genuine, impostor = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if staff_ids[i] == staff_ids[j]:
                genuine.append(sims[i, j])
            else:
                impostor.append(sims[i, j])

    def report(label, values):
        if not values:
            print(f"{label}: no pairs")
            return
        arr = np.array(values)
        print(f"{label}: n={len(arr)} min={arr.min():.3f} mean={arr.mean():.3f} "
              f"max={arr.max():.3f} std={arr.std():.3f}")

    report("Genuine (same person)", genuine)
    report("Impostor (different person)", impostor)

    if genuine and impostor:
        suggested = (float(np.mean(genuine)) + float(np.mean(impostor))) / 2
        print(f"\nSuggested MATCH_THRESHOLD midpoint (rough starting point): {suggested:.3f}")
        print("Set app/config.py MATCH_THRESHOLD accordingly, then re-verify with entry_monitor.py.")
    else:
        print("\nNeed both genuine and impostor pairs (>=2 staff, each with multiple embeddings) "
              "for a useful suggestion.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
