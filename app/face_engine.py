"""Face detection + embedding + open-set matching.

Detection: OpenCV YuNet (cv2.FaceDetectorYN) -- ONNX model, runs 5-10x
faster than RetinaFace on CPU since it skips the TF graph entirely.
Traded off: less robust to heavy occlusion (mask/cap) than RetinaFace,
which is why DETECTOR_SCORE_THRESHOLD stays low. Embedding:
facenet-pytorch InceptionResnetV1 pretrained on vggface2, 512-d.
Matching is open-set cosine similarity (max similarity across a
person's stored embeddings vs a threshold) so new staff enroll without
retraining -- not the old repo's closed 14-class classifier.
"""
import os
import sys
from collections import namedtuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import db

MatchResult = namedtuple("MatchResult", "staff_id name similarity authorized decision")

_embedder = None
_detector = None
_detector_size = None  # (w, h) the cached _detector was created/resized for
_embedding_cache = None  # (matrix (N,512), staff_ids, names, authorized)


def _get_embedder():
    global _embedder
    if _embedder is None:
        from facenet_pytorch import InceptionResnetV1
        _embedder = InceptionResnetV1(pretrained="vggface2").eval()
    return _embedder


def _get_detector(width, height):
    """YuNet needs setInputSize matched to the actual frame; cache the
    detector and only re-set size when the frame dims change (video
    frames are constant size, so this is a no-op after the first call)."""
    global _detector, _detector_size
    import cv2

    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(
            config.YUNET_MODEL_PATH, "", (width, height),
            score_threshold=config.DETECTOR_SCORE_THRESHOLD,
            nms_threshold=config.YUNET_NMS_THRESHOLD,
        )
        _detector_size = (width, height)
    elif _detector_size != (width, height):
        _detector.setInputSize((width, height))
        _detector_size = (width, height)
    return _detector


def warm_up():
    """Force-load the detector + embedder models. Call once at app startup
    so the cost is paid before a user starts registering, not during it."""
    dummy = np.zeros((160, 160, 3), dtype=np.uint8)
    detect_faces(dummy)  # triggers ONNX graph load, result ignored
    _get_embedder()  # triggers torch weight load


def detect_faces(frame):
    """frame: BGR uint8 np.ndarray (as from cv2.imread/VideoCapture).
    Returns list of {"box": (x1,y1,x2,y2), "score": float, "landmarks": dict|None}.
    """
    height, width = frame.shape[:2]
    detector = _get_detector(width, height)
    _, results = detector.detect(frame)

    faces = []
    if results is None:
        return faces

    for row in results:
        x, y, w, h = row[0:4]
        score = float(row[14])
        if score < config.DETECTOR_SCORE_THRESHOLD:
            continue
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        landmarks = {
            "right_eye": tuple(row[4:6].astype(int)),
            "left_eye": tuple(row[6:8].astype(int)),
            "nose": tuple(row[8:10].astype(int)),
            "mouth_right": tuple(row[10:12].astype(int)),
            "mouth_left": tuple(row[12:14].astype(int)),
        }
        faces.append({
            "box": (x1, y1, x2, y2),
            "score": score,
            "landmarks": landmarks,
        })
    return faces


def get_embedding(face_crop):
    """face_crop: BGR uint8 np.ndarray face region.
    Returns an L2-normalized float32 (512,) embedding.
    """
    import cv2
    import torch

    embedder = _get_embedder()
    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (160, 160))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float()
    tensor = (tensor - 127.5) / 128.0  # facenet-pytorch's expected input normalization
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        emb = embedder(tensor).squeeze(0).numpy().astype(np.float32)

    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def enroll_person(name, role, images, employee_code=None, authorized=True, progress_callback=None):
    """images: list of BGR uint8 np.ndarray frames from burst capture.
    Detects the largest face per image, embeds it, and stores staff +
    embedding rows. Raises ValueError if fewer than config.MIN_ENROLL_IMAGES
    faces were usable.

    progress_callback, if given, is called after each image as
    progress_callback(index, total, status) where status is one of
    "ok", "no_face", or "empty_crop".
    """
    embeddings = []
    total = len(images)
    for i, img in enumerate(images):
        status = "ok"
        faces = detect_faces(img)
        if not faces:
            status = "no_face"
        else:
            best = max(faces, key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))
            x1, y1, x2, y2 = best["box"]
            x1, y1 = max(x1, 0), max(y1, 0)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                status = "empty_crop"
            else:
                embeddings.append(get_embedding(crop))
        if progress_callback:
            progress_callback(i + 1, total, status)

    if len(embeddings) < config.MIN_ENROLL_IMAGES:
        raise ValueError(
            f"Only {len(embeddings)} usable face(s) detected out of {len(images)} photos "
            f"(need at least {config.MIN_ENROLL_IMAGES}). Retry with better lighting/angle."
        )

    staff_id = db.insert_staff(name, role, employee_code=employee_code, authorized=authorized)
    for emb in embeddings:
        db.insert_embedding(staff_id, emb)

    invalidate_cache()
    return staff_id


def invalidate_cache():
    global _embedding_cache
    _embedding_cache = None


def load_all_embeddings():
    """Returns (matrix (N,512) float32, staff_ids list). Cached in memory
    until invalidate_cache() is called (enroll_person() does this)."""
    global _embedding_cache
    if _embedding_cache is not None:
        return _embedding_cache[0], _embedding_cache[1]

    df = db.get_all_embeddings()
    if df.empty:
        matrix = np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32)
        staff_ids, names, authorized = [], [], []
    else:
        vectors = [np.frombuffer(b, dtype=np.float32) for b in df["embedding"]]
        matrix = np.vstack(vectors).astype(np.float32)
        staff_ids = df["staff_id"].tolist()
        names = df["name"].tolist()
        authorized = df["authorized"].tolist()

    _embedding_cache = (matrix, staff_ids, names, authorized)
    return matrix, staff_ids


def match(embedding, threshold=None):
    """Compares embedding (512,) against all stored embeddings via cosine
    similarity. Per-person score = max similarity across that person's
    stored embeddings (more forgiving of pose/lighting than a centroid).
    """
    if threshold is None:
        threshold = config.MATCH_THRESHOLD

    load_all_embeddings()
    matrix, staff_ids, names, authorized = _embedding_cache

    if matrix.shape[0] == 0:
        return MatchResult(None, None, 0.0, False, "UNKNOWN")

    sims = matrix @ embedding  # both L2-normalized -> dot product == cosine similarity

    best_per_staff = {}
    for sid, name, auth, sim in zip(staff_ids, names, authorized, sims):
        if sid not in best_per_staff or sim > best_per_staff[sid][0]:
            best_per_staff[sid] = (sim, name, bool(auth))

    best_staff_id, (best_sim, best_name, best_auth) = max(
        best_per_staff.items(), key=lambda kv: kv[1][0]
    )

    if best_sim < threshold:
        return MatchResult(None, None, float(best_sim), False, "UNKNOWN")

    decision = "AUTHORIZED" if best_auth else "DENIED"
    return MatchResult(best_staff_id, best_name, float(best_sim), best_auth, decision)
