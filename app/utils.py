from typing import List, Optional, Tuple

import cv2
import numpy as np

# Affine age calibration: true ≈ a*pred + b. Counters IMDB-WIKI mean-reversion.
# Fitted on 1000 AgeDB samples (model_benchmark.ipynb).

_MALE_CALIB = (1.3218, -11.2277)
_FEMALE_CALIB = (1.2113, -2.9902)
AGE_CALIB = {
    "Female": _MALE_CALIB,
    "Male": _MALE_CALIB,
}


def calibrate_age(faces: List[dict], calib: dict = AGE_CALIB) -> None:
    """Apply per-gender affine calibration in place, clamped to [0, 100].

    Proportional correction (slope, not flat offset): small near the model's
    center, large at the extremes -- matches the mean-reversion error shape.
    """
    for face in faces:
        age = face.get("age")
        if age is None:
            continue
        a, b = calib.get(face.get("gender"), (1.0, 0.0))
        face["age"] = round(float(np.clip(a * age + b, 0.0, 100.0)), 1)

def gamma_correct(img: np.ndarray, gamma: float = 0.8) -> np.ndarray:
    """Power-law tone curve. Pixels in [0,1] raised to `gamma`, so gamma<1
    brightens shadows nonlinearly, gamma>1 darkens (LUT over 0..255)."""
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def white_balance_grayworld(img: np.ndarray) -> np.ndarray:
    """Gray-world white balance: scale each BGR channel so its mean matches the
    overall gray mean. Removes the webcam color cast that the (neutral) training
    images don't have.
    """
    res = img.astype(np.float32)
    means = res.reshape(-1, 3).mean(axis=0)
    gray = means.mean()
    scale = gray / np.clip(means, 1e-6, None)
    res *= scale
    return np.clip(res, 0, 255).astype(np.uint8)


def retinex_msr(
    l: np.ndarray, sigmas: Tuple[float, ...] = (15.0, 80.0, 250.0)
) -> np.ndarray:
    """Multi-Scale Retinex on a single (luminance) channel.

    reflectance = log(I) - log(blur(I)); summed over Gaussian scales. The blur
    estimates the illumination, so subtracting it removes the lighting gradient
    (shadows across the face) and keeps the reflectance (skin/structure). Per
    pixel & local -> a bright background can't starve the face the way a global
    histogram does. Output rescaled to 0..255.
    """
    f = l.astype(np.float32) + 1.0
    log_f = np.log(f)
    acc = np.zeros_like(f)
    for s in sigmas:
        blur = cv2.GaussianBlur(f, (0, 0), s)
        acc += log_f - np.log(blur + 1.0)
    acc /= len(sigmas)
    # robust rescale: clip 1st/99th percentile so a few outliers don't crush range
    lo, hi = np.percentile(acc, (1.0, 99.0))
    acc = np.clip((acc - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return (acc * 255.0).astype(np.uint8)


def enhance_lowlight(
    img: np.ndarray,
    sigmas: Tuple[float, ...] = (15.0, 80.0, 250.0),
    clip_limit: float = 1.0,
    tile_grid: Tuple[int, int] = (8, 8),
    blend: float = 0.3,
    gamma: float = 0.9,
    contrast: float = 1.0,
    saturation: float = 1.0,
    white_balance: bool = False,
) -> np.ndarray:
    """Gently illumination-normalize a face toward the even-lit training
    distribution. Tuned soft to avoid over-processing already-lit frames.

    Pipeline:
      1. (optional) gray-world WB       -> off by default: on warm/non-neutral
                                            scenes it overcorrects to a green
                                            cast, so only enable for true casts
      2. multi-scale Retinex on LAB-L  -> flatten the lighting gradient (shadows)
      3. CLAHE on the result           -> local contrast (clip kept low=1.0 so it
                                            doesn't amplify sensor noise/JPEG)
      4. blend Retinex-L back toward L  -> low blend (0.3) keeps most original
                                            luminance, so skin texture / wrinkles
                                            (key age cues) survive
      5. gamma lift on L                -> mild brightness (0.9)
      6. saturation on a/b              -> off by default

    Geometry unchanged -> detection boxes from the original image map 1:1.
    """
    src = white_balance_grayworld(img) if white_balance else img
    lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    retinex_l = retinex_msr(l, sigmas)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    retinex_l = clahe.apply(retinex_l)

    # low blend -> output stays close to original L, preserving fine texture
    out_l = cv2.addWeighted(retinex_l, blend, l, 1.0 - blend, 0.0)
    # brightness lift on luminance only (gamma<1 brightens shadows)
    if gamma != 1.0:
        out_l = gamma_correct(out_l, gamma)
    # contrast: stretch L around mid-gray to add punch Retinex removed
    if contrast != 1.0:
        out_l = np.clip(
            (out_l.astype(np.float32) - 128.0) * contrast + 128.0, 0, 255
        ).astype(np.uint8)
    # saturation: scale a/b chroma around the neutral point (128)
    if saturation != 1.0:
        a = np.clip((a.astype(np.float32) - 128.0) * saturation + 128.0, 0, 255).astype(np.uint8)
        b = np.clip((b.astype(np.float32) - 128.0) * saturation + 128.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((out_l, a, b)), cv2.COLOR_LAB2BGR)
