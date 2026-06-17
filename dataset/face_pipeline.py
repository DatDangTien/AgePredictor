"""
FaceAlignPipeline: face detection + alignment using face_det_lite ONNX.

Replicates the model_benchmark.ipynb pipeline:
  1. Letterbox-resize to 640×480, run face_det_lite ONNX
  2. Rotate full frame so the eye line is horizontal (warpAffine)
  3. Expand bbox by margin, crop
  4. Letterbox-resize crop to target_size
  5. Return PIL RGB image (unnormalized) — let the dataset transform normalize

Usage:
    pipeline = FaceAlignPipeline(
        model_path='models/face_det_lite-onnx-w8a8/face_det_lite.onnx',
        target_size=(112, 112),
        margin=0.3,
    )
    # use as face_align_fn in AllAgeFacesDataset
    dataset = AllAgeFacesDataset(..., face_align_fn=pipeline)
"""

import json
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


# ── detector constants ────────────────────────────────────────────────────────
_DET_H, _DET_W = 480, 640
_DET_STRIDE = 8
_DET_FMAP_H = _DET_H // _DET_STRIDE   # 60
_DET_FMAP_W = _DET_W // _DET_STRIDE   # 80


# ── low-level decode helpers ──────────────────────────────────────────────────
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _maxpool3x3(hm: np.ndarray) -> np.ndarray:
    padded = np.pad(hm, 1, mode="constant", constant_values=-np.inf)
    h, w = hm.shape
    out = np.full_like(hm, -np.inf)
    for dy in range(3):
        for dx in range(3):
            out = np.maximum(out, padded[dy:dy + h, dx:dx + w])
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(objs, iou_thr: float):
    if not objs or len(objs) <= 1:
        return objs
    objs = sorted(objs, key=lambda o: o[1], reverse=True)
    flags = [False] * len(objs)
    keep = []
    for i, obj in enumerate(objs):
        if flags[i]:
            continue
        keep.append(obj)
        for j in range(i + 1, len(objs)):
            if not flags[j] and _iou(np.array(obj[0]), np.array(objs[j][0])) > iou_thr:
                flags[j] = True
    return keep


def _decode(hm, box, landmark, threshold, nms_iou, stride):
    """hm/box/landmark: (1,C,H,W) float32 after dequant."""
    hm = _sigmoid(hm)[0, 0]
    hmp = _maxpool3x3(hm)
    peak = ((hm == hmp).astype(np.float32) * hm).ravel()

    h, w = hm.shape
    k = min(peak.size, 2000)
    top = np.argpartition(peak, -k)[-k:]
    top = top[np.argsort(peak[top])[::-1]]

    objs = []
    for idx in top:
        score = float(peak[idx])
        if score < threshold:
            break
        cy, cx = divmod(int(idx), w)
        x, y, r, b = box[0, :, cy, cx]
        xyrb = ((np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride).tolist()
        x5y5 = landmark[0, :, cy, cx]
        x5y5 = (x5y5 + ([cx] * 5 + [cy] * 5)) * stride
        lm = list(zip(x5y5[:5].tolist(), x5y5[5:].tolist()))
        objs.append((xyrb, score, lm))

    return _nms(objs, nms_iou)


class FaceAlignPipeline:
    """
    Drop-in face_align_fn for AllAgeFacesDataset.

    Accepts a PIL RGB image, returns a PIL RGB image at `target_size`,
    or None if no face is detected.
    """

    def __init__(
        self,
        model_path: str,
        target_size: Tuple[int, int] = (112, 112),
        margin: float = 0.3,
        score_threshold: float = 0.55,
        nms_iou: float = 0.3,
    ):
        self.target_size = target_size   # (W, H) for cv2.resize / PIL
        self.margin = margin
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Load per-output quant params from metadata.json
        meta_path = os.path.join(os.path.dirname(model_path), "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        spec = meta["model_files"][os.path.basename(model_path)]
        self._quant = {
            name: (
                o["quantization_parameters"]["scale"],
                o["quantization_parameters"]["zero_point"],
            )
            for name, o in spec["outputs"].items()
            if "quantization_parameters" in o
        }

    # ── internal helpers ──────────────────────────────────────────────────────

    def _dequant(self, arr: np.ndarray, name: str) -> np.ndarray:
        if arr.dtype != np.uint8:
            return arr.astype(np.float32)
        scale, zp = self._quant[name]
        return (arr.astype(np.float32) - zp) * scale

    def _detect(self, image_bgr: np.ndarray):
        """Run face_det_lite on a BGR frame. Returns list of (xyxy, score, lm5)."""
        orig_h, orig_w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        scale = min(_DET_W / orig_w, _DET_H / orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        pad_x = (_DET_W - new_w) // 2
        pad_y = (_DET_H - new_h) // 2
        canvas = np.zeros((_DET_H, _DET_W), dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = cv2.resize(
            gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

        nchw = np.ascontiguousarray(canvas[None, None, ...])
        out = self.session.run(self.output_names, {self.input_name: nchw})

        hm = self._dequant(out[0], "heatmap").reshape(1, 1, _DET_FMAP_H, _DET_FMAP_W)
        bx = self._dequant(out[1], "bbox").reshape(1, 4, _DET_FMAP_H, _DET_FMAP_W)
        lm = self._dequant(out[2], "landmark").reshape(1, 10, _DET_FMAP_H, _DET_FMAP_W)

        dets = _decode(hm, bx, lm, self.score_threshold, self.nms_iou, _DET_STRIDE)

        # Undo letterbox
        results = []
        for xyxy, score, landmark in dets:
            x1 = max(0.0, (float(xyxy[0]) - pad_x) / scale)
            y1 = max(0.0, (float(xyxy[1]) - pad_y) / scale)
            x2 = min(orig_w - 1.0, (float(xyxy[2]) - pad_x) / scale)
            y2 = min(orig_h - 1.0, (float(xyxy[3]) - pad_y) / scale)
            if x2 <= x1 or y2 <= y1:
                continue
            lm_orig = [
                (max(0.0, (lx - pad_x) / scale), max(0.0, (ly - pad_y) / scale))
                for lx, ly in landmark
            ]
            results.append(([x1, y1, x2, y2], score, lm_orig))

        return results

    def _crop(self, image_bgr: np.ndarray, xyxy, landmark) -> Optional[np.ndarray]:
        """Rotate → margin-expand → crop → letterbox. Returns BGR uint8 at target_size."""
        h, w = image_bgr.shape[:2]

        # 1. Align: rotate whole frame so the eye line is horizontal
        if landmark and len(landmark) >= 2:
            (lx, ly), (rx, ry) = landmark[0], landmark[1]
            angle = float(np.degrees(np.arctan2(ry - ly, rx - lx)))
            eye_c = ((lx + rx) / 2.0, (ly + ry) / 2.0)
            M = cv2.getRotationMatrix2D(eye_c, angle, 1.0)
            image_bgr = cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_LINEAR)

        # 2. Expand bbox by margin, clamp
        x1, y1, x2, y2 = xyxy
        bw, bh = x2 - x1, y2 - y1
        x1 -= bw * self.margin;  x2 += bw * self.margin
        y1 -= bh * self.margin;  y2 += bh * self.margin
        x1 = int(max(0, min(x1, w - 1)));  x2 = int(max(0, min(x2, w - 1)))
        y1 = int(max(0, min(y1, h - 1)));  y2 = int(max(0, min(y2, h - 1)))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = image_bgr[y1:y2, x1:x2]

        # 3. Letterbox to target_size (aspect-preserving + zero-pad)
        tw, th = self.target_size
        ch, cw = crop.shape[:2]
        factor = min(tw / cw, th / ch)
        dsize = (max(1, int(cw * factor)), max(1, int(ch * factor)))
        img = cv2.resize(crop, dsize)
        d0, d1 = th - img.shape[0], tw - img.shape[1]
        img = np.pad(
            img,
            ((d0 // 2, d0 - d0 // 2), (d1 // 2, d1 - d1 // 2), (0, 0)),
            "constant",
        )
        if img.shape[:2] != (th, tw):
            img = cv2.resize(img, (tw, th))
        return img

    # ── public interface ──────────────────────────────────────────────────────

    def __call__(self, pil_rgb_image: Image.Image) -> Optional[Image.Image]:
        """
        Args:
            pil_rgb_image: PIL RGB image (any size).
        Returns:
            PIL RGB image at self.target_size, or None if no face detected.
        """
        img_bgr = cv2.cvtColor(np.array(pil_rgb_image), cv2.COLOR_RGB2BGR)
        dets = self._detect(img_bgr)
        if not dets:
            return None

        xyxy, score, landmark = dets[0]   # highest-confidence face
        crop_bgr = self._crop(img_bgr, xyxy, landmark)
        if crop_bgr is None:
            return None

        return Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
