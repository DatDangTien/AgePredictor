"""Age detection pipeline (onnxruntime).

Two stages:
  1. face_det_lite  -> detect every face (bbox + 5 landmarks)
  2. age            -> per-face age regression over a softmax of 0..100

The face-detection post-processing (`decode`/`nms`) is vendored here as a
pure-numpy reimplementation of qai_hub_models.models.face_det_lite.utils.detect
so the runtime needs neither torch nor qai_hub_models.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

# Max ONNX IR version onnxruntime accepts; newer model files are downgraded
# in-memory at load (the face model ships as IR 12, ort supports <= 11).
_MAX_IR_VERSION = 10

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
FACE_DET_MODEL_PATH = os.path.join(
    MODELS_DIR, "face_det_lite-onnx-w8a8", "face_det_lite.onnx"
)
AGE_MODEL_PATH = os.path.join(MODELS_DIR, "age.onnx")
GENDER_MODEL_PATH = os.path.join(MODELS_DIR, "gender.onnx")

# gender model output order
GENDER_LABELS = ["Female", "Male"]
# Predict Female when p(Female) >= this. 0.5 == plain argmax. Lower it (e.g.
# 0.35) to counter the IMDB-WIKI male skew (model calls women "Male"). Fit on
# AgeDB via the gender-threshold sweep cell in model_benchmark.ipynb.
GENDER_FEMALE_THRESHOLD = float(os.environ.get("GENDER_FEMALE_THRESHOLD", "0.5"))

FACE_DET_INPUT_H = 480
FACE_DET_INPUT_W = 640
FACE_DET_STRIDE = 8
FACE_DET_SCORE_THRESHOLD = float(os.environ.get("FACE_SCORE_THRESHOLD", "0.5"))
FACE_DET_NMS_IOU = float(os.environ.get("FACE_NMS_IOU", "0.3"))

AGE_INPUT_SIZE = (224, 224)
AGE_MARGIN_RATE = float(os.environ.get("AGE_MARGIN_RATE", "0.4"))


# ---------------------------------------------------------------------------
# Vendored face-detection decode (numpy only)
# ---------------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _maxpool3x3(hm: np.ndarray) -> np.ndarray:
    """3x3 max pool, stride 1, pad 1 -- matches F.max_pool2d(hm, 3, 1, 1)."""
    padded = np.pad(hm, 1, mode="constant", constant_values=-np.inf)
    h, w = hm.shape
    out = np.full_like(hm, -np.inf)
    for dy in range(3):
        for dx in range(3):
            out = np.maximum(out, padded[dy : dy + h, dx : dx + w])
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _BBox:
    __slots__ = ("x", "y", "r", "b", "score", "landmark")

    def __init__(self, xyrb, score, landmark):
        x, y, r, b = xyrb
        self.x, self.y, self.r, self.b = min(x, r), min(y, b), max(x, r), max(y, b)
        self.score = score
        self.landmark = landmark

    @property
    def box(self):
        return [self.x, self.y, self.r, self.b]


def _nms(objs: List[_BBox], iou: float) -> List[_BBox]:
    if iou == -1 or objs is None or len(objs) <= 1:
        return objs
    objs = sorted(objs, key=lambda o: o.score, reverse=True)
    keep: List[_BBox] = []
    flags = [0] * len(objs)
    for i, obj in enumerate(objs):
        if flags[i]:
            continue
        keep.append(obj)
        for j in range(i + 1, len(objs)):
            if flags[j] == 0 and _iou(np.array(obj.box), np.array(objs[j].box)) > iou:
                flags[j] = 1
    return keep


def _decode(
    hm: np.ndarray,
    box: np.ndarray,
    landmark: np.ndarray,
    threshold: float,
    nms_iou: float,
    stride: int,
) -> List[_BBox]:
    """Reimplements qai_hub face_det_lite detect() in numpy.

    hm:       (1, 1, H, W)
    box:      (1, 4, H, W)
    landmark: (1, 10, H, W)
    """
    hm = _sigmoid(hm)[0, 0]  # (H, W)
    hmp = _maxpool3x3(hm)
    peak = ((hm == hmp).astype(np.float32) * hm).ravel()

    h, w = hm.shape
    k = min(peak.size, 2000)
    top = np.argpartition(peak, -k)[-k:]
    top = top[np.argsort(peak[top])[::-1]]  # descending by score

    objs: List[_BBox] = []
    for idx in top:
        score = float(peak[idx])
        if score < threshold:
            break
        cy, cx = divmod(int(idx), w)
        x, y, r, b = box[0, :, cy, cx]
        xyrb = ((np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride).tolist()
        x5y5 = landmark[0, :, cy, cx]
        x5y5 = (x5y5 + ([cx] * 5 + [cy] * 5)) * stride
        lm = list(zip(x5y5[:5], x5y5[5:]))
        objs.append(_BBox(xyrb, score, lm))

    return _nms(objs, nms_iou)


# ---------------------------------------------------------------------------
# Session loader (handles IR-version downgrade)
# ---------------------------------------------------------------------------
def _make_session(
    path: str, sess_options: "ort.SessionOptions", providers: List[str]
) -> "ort.InferenceSession":
    """Build an InferenceSession, downgrading the model IR version if ort
    cannot load it directly (e.g. face_det_lite ships as IR 12)."""
    try:
        return ort.InferenceSession(path, sess_options=sess_options, providers=providers)
    except Exception as exc:
        if "IR version" not in str(exc):
            raise
        import onnx  # lazy: only needed for the downgrade path

        model = onnx.load(path)  # pulls in external .data if present
        if model.ir_version > _MAX_IR_VERSION:
            model.ir_version = _MAX_IR_VERSION
        return ort.InferenceSession(
            model.SerializeToString(), sess_options=sess_options, providers=providers
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class AgePipeline:
    def __init__(
        self,
        face_model_path: str = FACE_DET_MODEL_PATH,
        age_model_path: str = AGE_MODEL_PATH,
        gender_model_path: str = GENDER_MODEL_PATH,
        providers: Optional[List[str]] = None,
    ):
        if providers is None:
            avail = ort.get_available_providers()
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in avail
                else ["CPUExecutionProvider"]
            )
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.face_sess = _make_session(face_model_path, so, providers)
        self.age_sess = _make_session(age_model_path, so, providers)
        self.gender_sess = _make_session(gender_model_path, so, providers)
        self.face_in = self.face_sess.get_inputs()[0].name
        # outputs: heatmap, bbox, landmark (declared order in the graph)
        self.face_out = [o.name for o in self.face_sess.get_outputs()]
        self.age_in = self.age_sess.get_inputs()[0].name
        self.age_out = self.age_sess.get_outputs()[0].name
        self.gender_in = self.gender_sess.get_inputs()[0].name
        self.gender_out = self.gender_sess.get_outputs()[0].name

        # face_det_lite is w8a8: uint8 in, uint8 (quantized) out. Load the
        # quantization params so we can quantize the input and dequantize the
        # heatmap/bbox/landmark tensors before decoding.
        self.face_in_dtype = self.face_sess.get_inputs()[0].type
        self._face_quant = self._load_quant_params(face_model_path)

    @staticmethod
    def _load_quant_params(model_path: str) -> dict:
        """Read quantization_parameters from the sibling metadata.json, if any.

        Returns {"input": (scale, zp), <output_name>: (scale, zp), ...}.
        """
        import json

        meta_path = os.path.join(os.path.dirname(model_path), "metadata.json")
        if not os.path.exists(meta_path):
            return {}
        with open(meta_path) as fh:
            meta = json.load(fh)
        fname = os.path.basename(model_path)
        spec = meta.get("model_files", {}).get(fname, {})
        out: dict = {}
        for inp in spec.get("inputs", {}).values():
            qp = inp.get("quantization_parameters")
            if qp:
                out["input"] = (qp["scale"], qp["zero_point"])
        for name, o in spec.get("outputs", {}).items():
            qp = o.get("quantization_parameters")
            if qp:
                out[name] = (qp["scale"], qp["zero_point"])
        return out

    def _dequant(self, arr: np.ndarray, name: str) -> np.ndarray:
        """Dequantize a uint8 output tensor: float = (q - zero_point) * scale."""
        if arr.dtype != np.uint8:
            return arr.astype(np.float32)
        scale, zp = self._face_quant.get(name, (1.0, 0))
        return (arr.astype(np.float32) - zp) * scale

    # ---- face detection ----
    def detect_faces(
        self, image_bgr: np.ndarray
    ) -> List[Tuple[List[float], float, list]]:
        orig_h, orig_w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # Letterbox: resize with a single scale (preserve aspect) then pad to the
        # model's fixed WxH. A plain resize to 640x480 stretches non-landscape
        # frames (e.g. mobile portrait selfies, h>w) -> faces get squashed off
        # the detector's training distribution and bboxes come out the wrong
        # shape. One scale + centering offset keeps geometry correct.
        scale = min(FACE_DET_INPUT_W / orig_w, FACE_DET_INPUT_H / orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        pad_x = (FACE_DET_INPUT_W - new_w) // 2
        pad_y = (FACE_DET_INPUT_H - new_h) // 2
        resized = np.zeros((FACE_DET_INPUT_H, FACE_DET_INPUT_W), dtype=gray.dtype)
        resized[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = cv2.resize(
            gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
        # Model expects values in [0, 1]. For the w8a8 model that means a
        # uint8 input quantized with the input scale/zero-point.
        norm = (resized.astype(np.float32) / 255.0)[None, None, ...]  # NCHW
        if "uint8" in self.face_in_dtype:
            q_scale, zp = self._face_quant.get("input", (1.0 / 255.0, 0))
            q = np.rint(norm / q_scale + zp)
            inp = np.ascontiguousarray(np.clip(q, 0, 255).astype(np.uint8))
        else:
            inp = np.ascontiguousarray(norm)

        hm, bx, lm = self.face_sess.run(self.face_out, {self.face_in: inp})
        hm = self._dequant(hm, "heatmap")
        bx = self._dequant(bx, "bbox")
        lm = self._dequant(lm, "landmark")

        dets = _decode(
            hm, bx, lm,
            threshold=FACE_DET_SCORE_THRESHOLD,
            nms_iou=FACE_DET_NMS_IOU,
            stride=FACE_DET_STRIDE,
        )

        # Undo the letterbox: subtract the pad offset, then divide by the single
        # scale to map model-space coords back to the original frame.
        results = []
        for d in dets:
            x1 = max(0.0, min((d.x - pad_x) / scale, orig_w - 1))
            y1 = max(0.0, min((d.y - pad_y) / scale, orig_h - 1))
            x2 = max(0.0, min((d.r - pad_x) / scale, orig_w - 1))
            y2 = max(0.0, min((d.b - pad_y) / scale, orig_h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            landmark = [
                ((float(lx) - pad_x) / scale, (float(ly) - pad_y) / scale)
                for lx, ly in d.landmark
            ]
            results.append(([x1, y1, x2, y2], float(d.score), landmark))
        return results

    # ---- age preprocess (align + margin + letterbox), matches notebook ----
    @staticmethod
    def _preprocess_face_crop(
        image_bgr: np.ndarray,
        xyxy: List[float],
        landmark: Optional[list] = None,
        target_size=AGE_INPUT_SIZE,
        margin: float = AGE_MARGIN_RATE,
    ) -> Optional[np.ndarray]:
        h, w = image_bgr.shape[:2]
        if landmark is not None and len(landmark) >= 2:
            (lx, ly), (rx, ry) = landmark[0], landmark[1]
            eye_c = ((lx + rx) / 2.0, (ly + ry) / 2.0)
            angle = float(np.degrees(np.arctan2(ry - ly, rx - lx)))
            M = cv2.getRotationMatrix2D(eye_c, angle, 1.0)
            image_bgr = cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_LINEAR)

        x1, y1, x2, y2 = xyxy
        bw, bh = x2 - x1, y2 - y1
        x1 -= bw * margin; x2 += bw * margin
        y1 -= bh * margin; y2 += bh * margin
        x1 = int(max(0, min(x1, w - 1))); y1 = int(max(0, min(y1, h - 1)))
        x2 = int(max(0, min(x2, w - 1))); y2 = int(max(0, min(y2, h - 1)))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = image_bgr[y1:y2, x1:x2]

        ch, cw = crop.shape[:2]
        factor = min(target_size[0] / ch, target_size[1] / cw)
        dsize = (max(1, int(cw * factor)), max(1, int(ch * factor)))
        img = cv2.resize(crop, dsize)
        d0 = target_size[0] - img.shape[0]
        d1 = target_size[1] - img.shape[1]
        img = np.pad(
            img,
            ((d0 // 2, d0 - d0 // 2), (d1 // 2, d1 - d1 // 2), (0, 0)),
            "constant",
        )
        if img.shape[:2] != tuple(target_size):
            img = cv2.resize(img, target_size)
        return np.ascontiguousarray((img.astype(np.float32) / 255.0)[None, ...])

    @staticmethod
    def _postprocess_age(output: np.ndarray) -> float:
        indices = np.arange(0, 101)
        return float(np.sum(output[0] * indices))

    @staticmethod
    def _postprocess_gender(output: np.ndarray) -> Tuple[str, float]:
        # output: (1, 2) softmax over [Female, Male].
        # Threshold p(Female) instead of argmax to debias the male skew.
        probs = output[0]
        p_female = float(probs[0])
        if p_female >= GENDER_FEMALE_THRESHOLD:
            return "Female", p_female
        return "Male", float(probs[1])

    # ---- full pipeline ----
    def predict(
        self,
        image_bgr: np.ndarray,
        detect_image: Optional[np.ndarray] = None,
    ) -> List[dict]:
        """Detect all faces, return list of {box, score, age, gender, gender_score}.

        Detection runs on `detect_image` when given (keep the detector in its
        training distribution), while age/gender crops come from `image_bgr`
        (e.g. a low-light-enhanced copy). Both must share the same dimensions
        so boxes map across. Falls back to `image_bgr` for detection.
        """
        det_src = detect_image if detect_image is not None else image_bgr
        faces = self.detect_faces(det_src)
        out = []
        for xyxy, score, landmark in faces:
            inp = self._preprocess_face_crop(image_bgr, xyxy, landmark)
            if inp is None:
                continue
            # age + gender share the same aligned face crop
            age_pred = self.age_sess.run([self.age_out], {self.age_in: inp})
            age = self._postprocess_age(age_pred[0])
            gen_pred = self.gender_sess.run([self.gender_out], {self.gender_in: inp})
            gender, gender_score = self._postprocess_gender(gen_pred[0])
            out.append(
                {
                    "box": [round(v, 1) for v in xyxy],
                    "score": round(score, 4),
                    "age": round(age, 1),
                    "gender": gender,
                    "gender_score": round(gender_score, 4),
                }
            )
        return out


_pipeline: Optional[AgePipeline] = None


def get_pipeline() -> AgePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AgePipeline()
    return _pipeline
