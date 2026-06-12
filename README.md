# Age Predictor

Detect every face in an image and estimate each person's age — from a photo
upload **or** your live camera. Runs a fast two-stage ONNX vision pipeline
behind a FastAPI server with a clean, responsive web UI that works on phone and
desktop.

> 📷 **Live demo:** _video coming soon_ — see [Demo video](#demo-video) for how
> it's recorded.

```
image ──► face detection ──► per-face crop + align ──► age model ──► age per face
```

## Features

- 🧑‍🤝‍🧑 **Multi-face** — detects and ages every face in the frame, not just one.
- 🖼️ **Upload** — tap-to-pick or drag-and-drop. Responsive layout for mobile + desktop.
- 🎥 **Live camera** — point your webcam/phone camera and get ages in real time.
- 🎯 **Visual results** — boxes drawn on a `<canvas>` with an age label per face,
  plus a chip list showing age + detection confidence.
- ⚡ **GPU-accelerated** — ONNX Runtime uses CUDA when available, falls back to
  CPU automatically.
- 🪶 **Light footprint** — face detector is quantized (w8a8); no torch or
  qai_hub_models needed at runtime.
- 🛡️ **Clear errors** — bad image, oversized upload, blocked camera, and
  inference failures each surface a readable message.

## Quick start

### Run with Docker (recommended, GPU)

The image is GPU-enabled: CUDA 12.6 + cuDNN 9 base + `onnxruntime-gpu`. Models
are **not** baked into the image — they are volume-mounted read-only from
`./models` (see `docker-compose.yml`).

**Host requirement:** NVIDIA driver + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker compose up --build
# open http://localhost:8000
```

Confirm GPU is used — startup log should print:

```
[age-predictor] face providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

No GPU/toolkit on the host? ONNX Runtime falls back to CPU automatically (the
CUDA provider just won't load). Env vars (set in `docker-compose.yml`):
`MODELS_DIR`, `FACE_SCORE_THRESHOLD`, `FACE_NMS_IOU`, `AGE_MARGIN_RATE`,
`MAX_UPLOAD_BYTES`.

### Run locally (conda env `vision_worker`)

```bash
./run.sh                 # serves on 0.0.0.0:8000
```

> `run.sh` sets `PYTHONNOUSERSITE=1`. Without it, `~/.local/.../onnxruntime`
> shadows the env's real `onnxruntime` as an empty namespace package and
> `import onnxruntime` fails.

## Web UI notes

- **Camera** needs a **secure context** (HTTPS or `localhost`). Over plain HTTP
  on a LAN IP the browser blocks the camera — the UI detects this and shows a
  clear message instead of failing silently. Permission-denied / no-camera /
  camera-in-use errors are each surfaced.
- Loading and error states are shown inline.

## Tech stack

| Layer        | Choice |
|--------------|--------|
| Web UI       | Vanilla HTML / CSS / JS (`app/static/`), `<canvas>` overlay, `getUserMedia` for camera |
| API server   | FastAPI 0.118 + Uvicorn |
| Inference    | ONNX Runtime (`onnxruntime-gpu` 1.23) |
| Image ops    | OpenCV (headless) + NumPy |
| Models       | ONNX — quantized face detector + float32 age classifier |
| Packaging    | Docker, CUDA 12.6 / cuDNN 9 base image, models mounted as a volume |
| Hardware     | NVIDIA GPU (CUDA) with automatic CPU fallback |

## Technical details

### Pipeline

- **Face detection** — `models/face_det_lite-onnx-w8a8/face_det_lite.onnx`
  (quantized w8a8: `uint8` in, `uint8` out). Runtime quantizes the grayscale
  input and dequantizes the `heatmap`/`bbox`/`landmark` outputs using the
  scales/zero-points in `metadata.json`. The CenterNet-style decode + NMS is a
  pure-numpy reimplementation of `qai_hub_models...face_det_lite.utils.detect`,
  so neither **torch** nor **qai_hub_models** is needed at runtime.
- **Age** — `models/age.onnx` (`float32`, NHWC `[1,224,224,3]`, BGR /255).
  Output is a softmax over ages 0..100; predicted age = `Σ p_i · i`.
  Each crop is eye-aligned + letterboxed (matches the benchmark notebook).

### API

| Method | Path           | Body                         | Returns |
|--------|----------------|------------------------------|---------|
| GET    | `/health`      | —                            | `{"status":"ok"}` |
| POST   | `/api/predict` | `multipart/form-data` `file` | faces + ages |
| GET    | `/`            | —                            | web UI |

`POST /api/predict` response:

```json
{
  "count": 2,
  "width": 1920,
  "height": 1080,
  "faces": [
    {"box": [799.5, 329.9, 1085.6, 661.6], "score": 0.846, "age": 36.6}
  ]
}
```

`box` is `[x1,y1,x2,y2]` in original-image pixels. Errors return
`{"detail": "..."}` with 400 (bad/empty/undecodable image), 413 (too large),
or 500 (inference failure).

## Demo video

To showcase live age detection, record a short screen capture of the web UI:

1. Run the app (`docker compose up` or `./run.sh`) and open
   `http://localhost:8000`.
2. Start a screen recording — `OBS Studio` (free, cross-platform) or your OS
   recorder (macOS: <kbd>⌘⇧5</kbd>; Linux: `wf-recorder` / GNOME screencast;
   Windows: Xbox Game Bar <kbd>Win+G</kbd>).
3. Switch to the **Camera** tab, let it detect faces live, move around so the
   boxes + age labels track in real time.
4. Trim to ~15–30s, export as MP4.
5. Add it to the repo: either commit a short clip under `docs/demo.mp4` and
   reference it, or (better for GitHub) drag the MP4 into a GitHub issue/PR
   comment to get a hosted URL and paste that at the top of this README.

> GitHub renders an uploaded MP4 inline. A repo-committed file needs a link or
> an animated GIF (use `ffmpeg` to convert: `ffmpeg -i demo.mp4 -vf fps=12 demo.gif`).
