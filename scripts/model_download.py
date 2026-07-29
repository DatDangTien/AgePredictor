#!/usr/bin/env python3
"""Script to download required ONNX models into models/ directory."""

import argparse
import os
import sys
import urllib.request
import zipfile

FACE_DET_URL = (
    "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/"
    "models/face_det_lite/releases/v0.59.0/face_det_lite-onnx-w8a8.zip"
)

HF_GENDER_REPO = "DatinAI/AdaFace_gender"
HF_GENDER_FILENAME = "adaface_ir50_ms1mv2_gender.onnx"
HF_GENDER_DIRECT_URL = (
    f"https://huggingface.co/{HF_GENDER_REPO}/resolve/main/{HF_GENDER_FILENAME}"
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MODELS_DIR = os.path.join(REPO_ROOT, "models")


def download_file(url: str, dest_path: str) -> None:
    print(f"Downloading {url} -> {dest_path} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        total_size = response.getheader("Content-Length")
        if total_size is not None:
            total_size = int(total_size)

        downloaded = 0
        block_size = 8192
        while True:
            buffer = response.read(block_size)
            if not buffer:
                break
            downloaded += len(buffer)
            out_file.write(buffer)
            if total_size:
                percent = downloaded / total_size * 100
                print(
                    f"\rProgress: {downloaded}/{total_size} bytes ({percent:.1f}%)",
                    end="",
                    flush=True,
                )
            else:
                print(f"\rDownloaded {downloaded} bytes", end="", flush=True)
        print()


def extract_zip(zip_path: str, extract_to: str) -> None:
    print(f"Extracting {zip_path} to {extract_to} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")


def download_face_det(models_dir: str, keep_zip: bool = False) -> None:
    zip_dest_path = os.path.join(models_dir, "face_det_lite-onnx-w8a8.zip")
    try:
        download_file(FACE_DET_URL, zip_dest_path)
        extract_zip(zip_dest_path, models_dir)
    finally:
        if not keep_zip and os.path.exists(zip_dest_path):
            os.remove(zip_dest_path)
            print(f"Cleaned up {zip_dest_path}")


def download_gender_model(models_dir: str) -> None:
    print(f"Downloading gender model from Hugging Face ({HF_GENDER_REPO}) ...")
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=HF_GENDER_REPO,
            filename=HF_GENDER_FILENAME,
            local_dir=models_dir,
        )
        print(f"Downloaded {HF_GENDER_FILENAME} using huggingface_hub")
    except Exception as e:
        print(f"huggingface_hub download failed/unavailable ({e}), falling back to direct URL...")
        dest_path = os.path.join(models_dir, HF_GENDER_FILENAME)
        download_file(HF_GENDER_DIRECT_URL, dest_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract required models into models/."
    )
    parser.add_argument(
        "--models-dir",
        default=DEFAULT_MODELS_DIR,
        help="Target directory for models (default: models/)",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep downloaded zip file after extraction",
    )
    args = parser.parse_args()

    models_dir = os.path.abspath(args.models_dir)
    os.makedirs(models_dir, exist_ok=True)

    download_face_det(models_dir, keep_zip=args.keep_zip)
    download_gender_model(models_dir)

    print(f"All models successfully downloaded and placed in {models_dir}")


if __name__ == "__main__":
    main()
