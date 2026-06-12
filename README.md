# Age & Gender Predictor

Detect every face in an image and estimate each person's **age and gender** —
from a photo upload **or** your live camera. Runs a fast three-stage ONNX vision
pipeline behind a FastAPI server with a clean, responsive web UI that works on
phone and desktop.

```
image --> face detection --> per-face crop + align --+--> age model    --+--> age + gender
                                                     +--> gender model --+
```

## Demo

> 📷 **Live demo:** _video coming soon._

A short screen capture of the web UI showing live age + gender detection.

**How it's recorded:**

1. Run the app (`docker compose up` or `./run.sh`) and open
   `http://localhost:8000`.
2. Start a screen recording — `OBS Studio` (free, cross-platform) or your OS
   recorder (macOS: <kbd>⌘⇧5</kbd>; Linux: `wf-recorder` / GNOME screencast;
   Windows: Xbox Game Bar <kbd>Win+G</kbd>).
3. Switch to the **Camera** tab, let it detect faces live, move around so the
   boxes + age/gender labels track in real time.
4. Trim to ~15–30s, export as MP4.
5. Add it to the repo: either commit a short clip under `docs/demo.mp4` and
   reference it, or (better for GitHub) drag the MP4 into a GitHub issue/PR
   comment to get a hosted URL and paste it at the top of this README.

> GitHub renders an uploaded MP4 inline. A repo-committed file needs a link or
> an animated GIF (convert with `ffmpeg -i demo.mp4 -vf fps=12 demo.gif`).

## Models

| Stage          | Model                      | Backbone                    | Input             | Params |
|----------------|----------------------------|-----------------------------|-------------------|--------|
| Face detection | Lightweight-Face-Detection (Qualcomm) | MobileNetV3-Small | 640×480 grayscale | ~0.9M |
| Age            | DeepFace age               | VGGFace (VGG-16)            | 224×224 RGB       | ~135M |
| Gender         | DeepFace gender            | VGGFace (VGG-16)            | 224×224 RGB       | ~134M |

All three run as **ONNX** under ONNX Runtime. The face detector is quantized to
**8-bit** for a tiny footprint; age and gender are full-precision and share the
same detected face crop. Age predicts over 0–100; gender predicts Female/Male.

## Features

- 🧑‍🤝‍🧑 **Multi-face** — ages and genders every face in the frame, not just one.
- 🖼️ **Upload** — tap-to-pick or drag-and-drop. Responsive layout for mobile + desktop.
- 🎥 **Live camera** — point your webcam/phone camera and get age + gender in real time.
- ⚖️ **Debiased gender** — tunable threshold to counter the dataset's male skew.
- 🎯 **Visual results** — boxes drawn on a `<canvas>` with an age + gender label
  per face, plus a chip list showing the per-face scores.
- ⚡ **GPU-accelerated** — ONNX Runtime uses CUDA when available, falls back to
  CPU automatically.
- 🪶 **Light footprint** — quantized face detector; runs without torch.
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
`GENDER_FEMALE_THRESHOLD`, `MAX_UPLOAD_BYTES`.

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
| Models       | ONNX — quantized face detector + float32 age + gender classifiers |
| Packaging    | Docker, CUDA 12.6 / cuDNN 9 base image, models mounted as a volume |
| Hardware     | NVIDIA GPU (CUDA) with automatic CPU fallback |

## API

| Method | Path           | Body                         | Returns |
|--------|----------------|------------------------------|---------|
| GET    | `/health`      | —                            | `{"status":"ok"}` |
| POST   | `/api/predict` | `multipart/form-data` `file` | faces + age + gender |
| GET    | `/`            | —                            | web UI |

`POST /api/predict` response:

```json
{
  "count": 2,
  "width": 1920,
  "height": 1080,
  "faces": [
    {"box": [799.5, 329.9, 1085.6, 661.6], "score": 0.846, "age": 36.6, "gender": "Male", "gender_score": 0.91}
  ]
}
```

`box` is `[x1,y1,x2,y2]` in original-image pixels. Errors return
`{"detail": "..."}` with 400 (bad/empty/undecodable image), 413 (too large),
or 500 (inference failure).

## Known challenges

The age + gender models were trained on **IMDB-WIKI**, whose biases carry over:

- **Age** — skewed toward middle-aged faces; weaker on the elderly (tends to
  under-estimate older people).
- **Gender** — skewed toward **Male**; women are sometimes misclassified. A
  tunable decision threshold partly counters this at inference time.

## Future work

- **Fine-tune** the age + gender models on better-balanced data (more elderly,
  more female samples) to reduce the IMDB-WIKI bias.
- **Quantize** the age + gender models (like the face detector) for smaller
  footprint and faster inference.
