"""FastAPI service: POST an image (multipart/form-data) -> detected faces + ages."""

from __future__ import annotations

import base64
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.inference import get_pipeline
from app.utils import calibrate_age, enhance_lowlight

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))  # 15 MB
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="Age Predictor", version="1.0.0")


@app.on_event("startup")
def _warmup() -> None:
    # Build sessions + run one dummy inference so first request is fast.
    pipe = get_pipeline()
    print("[age-predictor] face providers:", pipe.face_sess.get_providers())
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        pipe.predict(dummy)
    except Exception:
        pass


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large.")

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    # Low-light enhancement for age/gender crops; detection stays on the
    # original to keep the detector in its training distribution.
    enhanced = enhance_lowlight(img)

    try:
        faces = get_pipeline().predict(enhanced, detect_image=img)
    except Exception as exc:  # inference failure -> 500 with message
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    calibrate_age(faces)

    # Return the enhanced image so the web UI visualizes what the models saw.
    ok, buf = cv2.imencode(".jpg", enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    image_data_url = (
        "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")
        if ok
        else None
    )

    return JSONResponse(
        {
            "count": len(faces),
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "image": image_data_url,
            "faces": faces,
        }
    )


# Web UI (mounted last so /api and /health take precedence).
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
