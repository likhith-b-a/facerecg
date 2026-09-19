# Masked-Face Staff Entry Recognition (POC)

Staff entry-control system that recognises people even when masked or
wearing a cap. Grew out of earlier partial-face-recognition research
(see [Research Background](#research-background)); this repo is the
working proof-of-concept implementation.

## What it does

- Enroll staff via webcam burst capture (mask + cap on).
- Live entry camera detects faces, matches against enrolled staff, and
  logs each attempt as `AUTHORIZED` / `DENIED` / `UNKNOWN`.
- Streamlit UI to register staff, toggle authorization, and review logs.

## Architecture

```
webcam burst (capture_utils.py)
        |
        v
  detect_faces()  -- OpenCV YuNet (ONNX), fast CPU detection
        |
        v
  get_embedding() -- facenet-pytorch InceptionResnetV1 (vggface2), 512-d, L2-normalized
        |
        v
  match()         -- cosine similarity vs stored embeddings (open-set, per-person max)
        |
        v
  SQLite (WAL)    -- staff / embeddings / entry_logs
```

Two independent processes share the same SQLite file (WAL mode handles
the concurrent access):

- `streamlit run app/Home.py` — admin UI (register / authorize / view logs).
- `python app/entry_monitor.py` — standalone OpenCV live camera loop for
  the actual entry point.

## Modules

| File | Purpose |
|---|---|
| `app/face_engine.py` | Detection, embedding, enrollment, open-set matching. Core recognition logic. |
| `app/db.py` | SQLite access layer (staff, embeddings, entry_logs), WAL mode. |
| `app/schema.sql` | Table definitions. |
| `app/config.py` | All tunables (thresholds, paths, camera settings). |
| `app/capture_utils.py` | Standalone OpenCV window for webcam burst capture, launched as a subprocess from the registration page. |
| `app/entry_monitor.py` | Live camera loop: detect → embed → match → draw box → debounced log + snapshot. Detection runs on a background thread so the video preview doesn't freeze. |
| `app/Home.py` | Streamlit entrypoint, model warm-up, dashboard metrics. |
| `app/pages/1_Register_Staff.py` | Burst-capture enrollment flow. |
| `app/pages/2_Manage_Authorization.py` | List staff, toggle authorized flag, delete. |
| `app/pages/3_Entry_Logs.py` | Filterable log table (date/decision/name) + snapshot preview. |
| `app/tools/calibrate_threshold.py` | Computes genuine vs impostor cosine-similarity stats from stored embeddings to help pick a real `MATCH_THRESHOLD`. |

## How recognition works

1. **Detect**: OpenCV YuNet (ONNX, weights checked into
   `app/models/`) finds face boxes + confidence scores. Score
   threshold is lowered to `0.6` (from a typical `0.9`) because
   occluded (mask/cap) faces score lower confidence than full faces.
2. **Embed**: face crop is resized to 160x160, normalized
   (`(pixel - 127.5) / 128.0`), and passed through FaceNet
   (`InceptionResnetV1`, pretrained on VGGFace2) to get a 512-d
   embedding, then L2-normalized.
3. **Match**: cosine similarity between the query embedding and every
   stored embedding. For each enrolled person, the *max* similarity
   across their stored embeddings is taken (more forgiving of
   pose/lighting than averaging into one centroid). If the best score
   clears `MATCH_THRESHOLD`, decision is `AUTHORIZED` or `DENIED`
   depending on that person's `authorized` flag; otherwise `UNKNOWN`.

This is **open-set** matching — new staff enroll without retraining any
model, unlike a closed-set classifier trained on a fixed set of
identities.

## Setup

```
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

`facenet-pytorch` auto-downloads its pretrained weights on first run
(needs internet once; see `models/README.md` for offline copy steps).
YuNet detector weights are checked into `app/models/`, no download needed.

## Running

```
streamlit run app/Home.py       # admin UI: register staff, manage auth, view logs
python app/entry_monitor.py     # live entry camera (separate process)
```

Optional flags for the entry monitor:

```
python app/entry_monitor.py --camera-index 0 --threshold 0.62
```

To calibrate the match threshold from real enrolled staff data:

```
python app/tools/calibrate_threshold.py
```

## Known limitations (POC scope)

- Single global `UNKNOWN` bucket in `entry_monitor.py` — two different
  unrecognized people in quick succession may only log the first.
- `MATCH_THRESHOLD` in `config.py` is a placeholder; calibrate with
  `tools/calibrate_threshold.py` once enough staff are enrolled.

## Research background

Earlier work on Partial Face Recognition (PFR) — recognising individuals
from partial/occluded facial regions for masked-face and attendance use
cases — compared VGGFace, ResNet50, AlexNet, FaceNet, and a custom CNN
on a custom dataset. **FaceNet performed best**: 97.7% training
accuracy, 85.5% validation accuracy. This POC carries FaceNet forward
as the embedding model, replacing the original closed-set classifier
with open-set cosine-similarity matching so staff can be added without
retraining.

### Dataset

| Metric | Value |
|---|---|
| Original images | 423 |
| After augmentation | 2,202 |
| Augmentation techniques | Rotation range, shear range, zoom range, horizontal flip, brightness range |

### Model comparison

Only FaceNet's final accuracy survived into the written record (below).
The other models' accuracy/precision/recall from the original training
runs are **not recorded in this repo** — do not cite numbers for them
without checking the original notebooks/thesis. Architecture facts
(public, model-inherent, not run-specific) are included for context:

| Model | Architecture | Typical input | Notes | Train acc. | Val acc. |
|---|---|---|---|---|---|
| VGGFace | VGG16-based CNN | 224x224 | Classic deep CNN, closed-set softmax | not recorded | not recorded |
| ResNet50 | 50-layer residual CNN | 224x224 | Residual connections ease deeper training | not recorded | not recorded |
| AlexNet | 8-layer CNN (2012) | 227x227 | Shallower, older baseline | not recorded | not recorded |
| Custom CNN | Project-specific, architecture undocumented here | — | Built for this dataset | not recorded | not recorded |
| **FaceNet** | Inception-ResNet-v1, triplet-loss embeddings | 160x160 | 512-d embedding, best performer | **97.7%** | **85.5%** |

### Current POC embedding config

| Parameter | Value | Source |
|---|---|---|
| Embedding dimension | 512 | `app/config.py` |
| Detector confidence threshold | 0.6 | `app/config.py` (lowered from typical 0.9 for occlusion) |
| Match threshold (placeholder) | 0.62 | `app/config.py`, needs calibration |
| Min images to enroll | 3 | `app/config.py` |
| Burst capture count | 8 photos @ 1.2s interval | `app/config.py` |

Real genuine-vs-impostor similarity stats (min/mean/max/std) are not
yet available — run `python app/tools/calibrate_threshold.py` once
enough staff are enrolled to get live numbers for the current system.
