#!/usr/bin/env python3
"""Download every source model artifact required by the ONNX export workflow."""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = REPO_ROOT / "models"

FACE_DETECTOR_URL = (
    "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/"
    "models/face_det_lite/releases/v0.59.0/face_det_lite-onnx-w8a8.zip"
)
FACE_DETECTOR_ARCHIVE = "face_det_lite-onnx-w8a8.zip"
FACE_DETECTOR_ONNX = Path("face_det_lite-onnx-w8a8/face_det_lite.onnx")
FACE_DETECTOR_DATA = Path("face_det_lite-onnx-w8a8/face_det_lite.data")

AGE_WEIGHTS_URL = (
    "https://github.com/serengil/deepface_models/releases/download/"
    "v1.0/age_model_weights.h5"
)
AGE_WEIGHTS_FILENAME = "age_model_weights.h5"

GENDER_REPOSITORY = "DatinAI/AdaFace_gender"
GENDER_ONNX_FILENAME = "adaface_ir50_ms1mv2_gender.onnx"
GENDER_ONNX_URL = (
    f"https://huggingface.co/{GENDER_REPOSITORY}/resolve/main/"
    f"{GENDER_ONNX_FILENAME}"
)

DOWNLOAD_BLOCK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class Download:
    name: str
    url: str
    filename: str


AGE_WEIGHTS = Download("DeepFace age weights", AGE_WEIGHTS_URL, AGE_WEIGHTS_FILENAME)
GENDER_ONNX = Download(
    "AdaFace gender ONNX model", GENDER_ONNX_URL, GENDER_ONNX_FILENAME
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download face, age, and gender model artifacts."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"Target model directory (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download files again even when they already exist",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded face-detector ZIP archive",
    )
    return parser.parse_args(argv)


def download_file(url: str, destination: Path, *, force: bool = False) -> Path:
    """Download a URL atomically, preserving any complete existing file."""
    destination = destination.expanduser().resolve()
    if destination.is_file() and destination.stat().st_size > 0 and not force:
        print(f"Using existing file: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    print(f"Downloading {url}\n  -> {destination}")
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as output:
            total_header = response.getheader("Content-Length")
            total = int(total_header) if total_header else None
            downloaded = 0

            while chunk := response.read(DOWNLOAD_BLOCK_SIZE):
                output.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded / total * 100
                    print(
                        f"\r  {downloaded:,}/{total:,} bytes ({percent:5.1f}%)",
                        end="",
                        flush=True,
                    )
                else:
                    print(f"\r  {downloaded:,} bytes", end="", flush=True)
        print()
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)

    return destination


def extract_zip(archive_path: Path, destination: Path) -> None:
    """Extract a ZIP after rejecting members that escape the target directory."""
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if not member_path.is_relative_to(destination):
                raise ValueError(f"Unsafe path in {archive_path}: {member.filename}")
        archive.extractall(destination)


def download_face_detector(
    models_dir: Path,
    *,
    force: bool = False,
    keep_zip: bool = False,
) -> None:
    model_path = models_dir / FACE_DETECTOR_ONNX
    data_path = models_dir / FACE_DETECTOR_DATA
    if model_path.is_file() and data_path.is_file() and not force:
        print(f"Using existing face detector: {model_path}")
        return

    archive_path = models_dir / FACE_DETECTOR_ARCHIVE
    try:
        download_file(FACE_DETECTOR_URL, archive_path, force=force)
        extract_zip(archive_path, models_dir)
    finally:
        if not keep_zip:
            archive_path.unlink(missing_ok=True)

    if not model_path.is_file() or not data_path.is_file():
        raise RuntimeError(f"Face-detector archive did not contain {model_path}")
    print(f"Face detector ready: {model_path}")


def download_artifact(
    artifact: Download,
    models_dir: Path,
    *,
    force: bool = False,
) -> Path:
    print(f"Preparing {artifact.name}...")
    return download_file(artifact.url, models_dir / artifact.filename, force=force)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    models_dir = args.models_dir.expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    download_face_detector(
        models_dir,
        force=args.force,
        keep_zip=args.keep_zip,
    )
    download_artifact(AGE_WEIGHTS, models_dir, force=args.force)
    download_artifact(GENDER_ONNX, models_dir, force=args.force)

    print(f"All source model artifacts are ready in {models_dir}")
    print("Run scripts/onnx_export.py to export and validate the runtime models.")


if __name__ == "__main__":
    main()
