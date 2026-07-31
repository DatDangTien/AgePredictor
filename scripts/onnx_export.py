#!/usr/bin/env python3
"""Export source checkpoints and validate every runtime ONNX model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_OPSET = 13

FACE_MODEL_RELATIVE_PATH = Path(
    "face_det_lite-onnx-w8a8/face_det_lite.onnx"
)
AGE_WEIGHTS_FILENAME = "age_model_weights.h5"
AGE_ONNX_FILENAME = "age.onnx"
GENDER_CHECKPOINT_FILENAME = "gender_best.pth"
GENDER_ONNX_FILENAME = "adaface_ir50_ms1mv2_gender.onnx"

for import_root in (REPO_ROOT, SCRIPTS_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def export_age_model(
    weights_path: Path,
    output_path: Path,
    *,
    opset: int = DEFAULT_OPSET,
) -> Path:
    """Export and validate one DeepFace age model.

    The ONNX file is written through a temporary sibling so a failed conversion
    cannot replace an existing working model.
    """
    weights_path = weights_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"DeepFace age weights not found: {weights_path}\n"
            f"Download them with:\n  {sys.executable} "
            f"{REPO_ROOT / 'scripts' / 'model_download.py'}"
        )

    onnx, tf, tf2onnx, vggface = _load_export_dependencies()
    model = _build_age_model(weights_path, tf, vggface)
    input_signature = (
        tf.TensorSpec(AGE_INPUT_SHAPE, tf.float32, name="input"),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f".{output_path.name}.tmp")
    temporary_output.unlink(missing_ok=True)

    print(f"Exporting DeepFace age model: {weights_path} -> {output_path}")
    try:
        tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=opset,
            output_path=str(temporary_output),
        )
        exported = onnx.load(str(temporary_output), load_external_data=True)
        onnx.checker.check_model(exported)
        temporary_output.replace(output_path)
    finally:
        temporary_output.unlink(missing_ok=True)

    print(f"Exported and verified: {output_path} ({output_path.stat().st_size:,} bytes)")
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export local age/gender checkpoints when needed and validate all "
            "runtime ONNX models."
        )
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"Model directory (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--gender-checkpoint",
        "--checkpoint",
        dest="gender_checkpoint",
        type=Path,
        help=(
            "Optional trained AdaFace gender checkpoint. Defaults to "
            "MODELS_DIR/gender_best.pth when that file exists."
        ),
    )
    parser.add_argument(
        "--gender-output",
        "--output",
        dest="gender_output",
        type=Path,
        help=(
            "Gender ONNX output. Defaults to "
            "MODELS_DIR/adaface_ir50_ms1mv2_gender.onnx."
        ),
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        help=f"ONNX opset used for conversions (default: {DEFAULT_OPSET})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-export models even when their ONNX files are up to date",
    )
    return parser.parse_args(argv)


def _load_onnx() -> Any:
    try:
        import onnx
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing export dependency 'onnx'. Install export dependencies with:\n"
            f"  {sys.executable} -m pip install -r "
            f"{REPO_ROOT / 'requirements-export.txt'}"
        ) from exc
    return onnx


def validate_onnx(path: Path, onnx: Any | None = None) -> Path:
    """Run ONNX's structural checker, including any sibling external data."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    onnx = onnx or _load_onnx()
    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model)
    print(f"Validated ONNX: {path} ({path.stat().st_size:,} bytes)")
    return path


def _needs_export(source: Path, output: Path, *, force: bool) -> bool:
    if force or not output.is_file():
        return True
    return source.stat().st_mtime_ns > output.stat().st_mtime_ns


def export_gender_model(
    checkpoint_path: Path,
    output_path: Path,
    *,
    opset: int = DEFAULT_OPSET,
) -> Path:
    """Export a trained AdaFace IR-50 gender checkpoint and validate it."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    adaface_source = REPO_ROOT / "thirdparty" / "adaface" / "net.py"

    if not adaface_source.is_file():
        raise FileNotFoundError(
            "AdaFace source is missing. Initialize the Git submodule first:\n"
            f"  git -C {REPO_ROOT} submodule update --init --recursive"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Gender checkpoint not found: {checkpoint_path}")

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing export dependency 'torch'. Install export dependencies with:\n"
            f"  {sys.executable} -m pip install -r "
            f"{REPO_ROOT / 'requirements-export.txt'}"
        ) from exc

    from src.model import GenderClassifier
    from thirdparty.adaface.net import Backbone

    backbone = Backbone(input_size=(224, 224), num_layers=50, mode="ir")
    model = GenderClassifier(backbone)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            f"Invalid checkpoint: {checkpoint_path} does not contain 'state_dict'"
        )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f".{output_path.name}.tmp")
    temporary_output.unlink(missing_ok=True)

    print(f"Exporting AdaFace gender model: {checkpoint_path} -> {output_path}")
    try:
        torch.onnx.export(
            model,
            (torch.randn(1, 3, 224, 224),),
            str(temporary_output),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset,
            dynamo=False,
        )
        validate_onnx(temporary_output)
        temporary_output.replace(output_path)
    finally:
        temporary_output.unlink(missing_ok=True)

    print(f"Exported and verified: {output_path}")
    return output_path


def export_age_if_needed(
    weights_path: Path,
    output_path: Path,
    *,
    opset: int,
    force: bool,
) -> None:
    if weights_path.is_file():
        if _needs_export(weights_path, output_path, force=force):
            export_age_model(weights_path, output_path, opset=opset)
        else:
            print(f"Age ONNX is up to date: {output_path}")
        return

    if force or not output_path.is_file():
        raise FileNotFoundError(
            f"DeepFace age weights not found: {weights_path}\n"
            "Run scripts/model_download.py before exporting."
        )
    print(f"Age H5 source is absent; validating existing ONNX: {output_path}")


def export_gender_if_needed(
    checkpoint_path: Path,
    output_path: Path,
    *,
    opset: int,
    force: bool,
) -> None:
    if checkpoint_path.is_file():
        if _needs_export(checkpoint_path, output_path, force=force):
            export_gender_model(checkpoint_path, output_path, opset=opset)
        else:
            print(f"Gender ONNX is up to date: {output_path}")
        return

    if not output_path.is_file():
        raise FileNotFoundError(
            f"Neither a gender checkpoint nor an ONNX model was found:\n"
            f"  checkpoint: {checkpoint_path}\n"
            f"  ONNX:      {output_path}\n"
            "Run scripts/model_download.py to download the pre-exported gender model."
        )
    print(f"No local gender checkpoint; using downloaded ONNX: {output_path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    models_dir = args.models_dir.expanduser().resolve()
    gender_checkpoint = (
        args.gender_checkpoint.expanduser().resolve()
        if args.gender_checkpoint
        else models_dir / GENDER_CHECKPOINT_FILENAME
    )
    gender_output = (
        args.gender_output.expanduser().resolve()
        if args.gender_output
        else models_dir / GENDER_ONNX_FILENAME
    )

    face_model = models_dir / FACE_MODEL_RELATIVE_PATH
    age_weights = models_dir / AGE_WEIGHTS_FILENAME
    age_output = models_dir / AGE_ONNX_FILENAME

    try:
        export_age_if_needed(
            age_weights,
            age_output,
            opset=args.opset,
            force=args.force,
        )
        export_gender_if_needed(
            gender_checkpoint,
            gender_output,
            opset=args.opset,
            force=args.force,
        )

        onnx = _load_onnx()
        for model_path in (face_model, age_output, gender_output):
            validate_onnx(model_path, onnx)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print("All runtime models are exported and ONNX-valid.")


if __name__ == "__main__":
    main()
