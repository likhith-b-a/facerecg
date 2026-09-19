# Model weights

No weight files are checked into this repo.

- `facenet-pytorch` auto-downloads its `InceptionResnetV1(pretrained='vggface2')`
  weights to `~/.cache/torch/checkpoints` the first time `app/face_engine.py`
  runs.
- Face detection uses OpenCV YuNet, weights checked into
  `app/models/face_detection_yunet_2023mar.onnx` -- no download needed.

`facenet-pytorch` needs internet access once. If this laptop is ever fully
offline, copy `~/.cache/torch/checkpoints` from a machine that already ran
the app (e.g. via USB) into the same path here.
