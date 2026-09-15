# Model weights

No weight files are checked into this repo.

- `facenet-pytorch` auto-downloads its `InceptionResnetV1(pretrained='vggface2')`
  weights to `~/.cache/torch/checkpoints` the first time `app/face_engine.py`
  runs.
- `retina-face` auto-downloads its own detector weights to `~/.deepface`
  the first time `RetinaFace.detect_faces()` is called.

Both need internet access once. If this laptop is ever fully offline, copy
`~/.cache/torch/checkpoints` and `~/.deepface` from a machine that already
ran the app (e.g. via USB) into the same paths here.
