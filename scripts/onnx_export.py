#!/usr/bin/env python3
"""Ensure the face, age, and gender runtime models are valid ONNX models.

The face detector and fallback gender model are distributed as pre-exported
ONNX assets. The age model is converted from DeepFace H5 weights, while the
gender model is converted from a local AdaFace checkpoint when one is present.
Every resulting model is checked structurally and against the tensor contract
used by ``app.inference.AgePipeline``.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
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

AGE_CLASSES = 101
AGE_INPUT_SHAPE = (None, 224, 224, 3)
GENDER_INPUT_SHAPE = (1, 3, 224, 224)
EXPORT_REQUIREMENTS = REPO_ROOT / "requirements-export.txt"

for import_root in (REPO_ROOT, SCRIPTS_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

TensorShape = tuple[int | None, ...]


@dataclass(frozen=True)
class OnnxContract:
    name: str
    input_shape: TensorShape
    output_shapes: tuple[TensorShape, ...]
    input_dtype: str = "FLOAT"
    output_dtypes: tuple[str, ...] = ("FLOAT",)
    output_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ExportPaths:
    face_model: Path
    age_weights: Path
    age_model: Path
    gender_checkpoint: Path
    gender_model: Path


FACE_CONTRACT = OnnxContract(
    name="face detector",
    input_shape=(None, 1, 480, 640),
    output_shapes=(
        (None, 1, 60, 80),
        (None, 4, 60, 80),
        (None, 10, 60, 80),
    ),
    input_dtype="UINT8",
    output_dtypes=("UINT8", "UINT8", "UINT8"),
    output_names=("heatmap", "bbox", "landmark"),
)
AGE_CONTRACT = OnnxContract(
    name="age model",
    input_shape=(None, 224, 224, 3),
    output_shapes=((None, AGE_CLASSES),),
)
GENDER_CONTRACT = OnnxContract(
    name="gender model",
    input_shape=(None, 3, 224, 224),
    output_shapes=((None, 2),),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"Model directory (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--face-model",
        type=Path,
        help="Pre-exported face detector ONNX path",
    )
    parser.add_argument(
        "--age-weights",
        "--weights",
        dest="age_weights",
        type=Path,
        help="DeepFace age H5 weights",
    )
    parser.add_argument(
        "--age-output",
        type=Path,
        help="Age ONNX output path",
    )
    parser.add_argument(
        "--gender-checkpoint",
        "--checkpoint",
        dest="gender_checkpoint",
        type=Path,
        help="Optional trained AdaFace gender checkpoint",
    )
    parser.add_argument(
        "--gender-output",
        type=Path,
        help="Gender ONNX output path",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        help=f"ONNX opset used for local conversions (default: {DEFAULT_OPSET})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-export convertible models when their source files are present",
    )
    return parser.parse_args(argv)


def _resolve_path(value: Path | None, default: Path) -> Path:
    return (value or default).expanduser().resolve()


def export_paths_from_args(args: argparse.Namespace) -> ExportPaths:
    models_dir = args.models_dir.expanduser().resolve()
    return ExportPaths(
        face_model=_resolve_path(
            args.face_model,
            models_dir / FACE_MODEL_RELATIVE_PATH,
        ),
        age_weights=_resolve_path(
            args.age_weights,
            models_dir / AGE_WEIGHTS_FILENAME,
        ),
        age_model=_resolve_path(
            args.age_output,
            models_dir / AGE_ONNX_FILENAME,
        ),
        gender_checkpoint=_resolve_path(
            args.gender_checkpoint,
            models_dir / GENDER_CHECKPOINT_FILENAME,
        ),
        gender_model=_resolve_path(
            args.gender_output,
            models_dir / GENDER_ONNX_FILENAME,
        ),
    )


def _dependency_error(name: str) -> SystemExit:
    return SystemExit(
        f"Missing export dependency '{name}'. Install export dependencies with:\n"
        f"  {sys.executable} -m pip install -r {EXPORT_REQUIREMENTS}"
    )


def _load_onnx() -> Any:
    try:
        import onnx
    except ModuleNotFoundError as exc:
        raise _dependency_error(exc.name or "onnx") from exc
    return onnx


def _tensor_shape(value_info: Any) -> TensorShape:
    dimensions: list[int | None] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        dimensions.append(dimension.dim_value or None)
    return tuple(dimensions)


def _tensor_dtype(value_info: Any, onnx: Any) -> str:
    element_type = value_info.type.tensor_type.elem_type
    return onnx.TensorProto.DataType.Name(element_type)


def _shape_matches(actual: TensorShape, expected: TensorShape) -> bool:
    return len(actual) == len(expected) and all(
        expected_dim is None
        or actual_dim is None
        or actual_dim == expected_dim
        for actual_dim, expected_dim in zip(actual, expected)
    )


def validate_onnx(
    path: Path,
    contract: OnnxContract,
    onnx: Any | None = None,
) -> Path:
    """Validate an ONNX graph and its production tensor contract."""
    path = path.expanduser().resolve()
    if path.suffix.lower() != ".onnx":
        raise ValueError(f"{contract.name} output must use the .onnx suffix: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{contract.name.title()} ONNX not found: {path}")

    onnx = onnx or _load_onnx()
    try:
        model = onnx.load(str(path), load_external_data=True)
        onnx.checker.check_model(model)
    except Exception as exc:
        raise ValueError(f"Invalid {contract.name} ONNX file {path}: {exc}") from exc

    initializer_names = {initializer.name for initializer in model.graph.initializer}
    inputs = [
        value_info
        for value_info in model.graph.input
        if value_info.name not in initializer_names
    ]
    outputs = list(model.graph.output)
    errors: list[str] = []

    if len(inputs) != 1:
        errors.append(f"expected 1 input, found {len(inputs)}")
    else:
        actual_shape = _tensor_shape(inputs[0])
        actual_dtype = _tensor_dtype(inputs[0], onnx)
        if not _shape_matches(actual_shape, contract.input_shape):
            errors.append(
                f"input shape {actual_shape}, expected {contract.input_shape}"
            )
        if actual_dtype != contract.input_dtype:
            errors.append(
                f"input dtype {actual_dtype}, expected {contract.input_dtype}"
            )

    if len(outputs) != len(contract.output_shapes):
        errors.append(
            f"expected {len(contract.output_shapes)} outputs, found {len(outputs)}"
        )
    else:
        for index, (output, expected_shape, expected_dtype) in enumerate(
            zip(outputs, contract.output_shapes, contract.output_dtypes)
        ):
            actual_shape = _tensor_shape(output)
            actual_dtype = _tensor_dtype(output, onnx)
            if not _shape_matches(actual_shape, expected_shape):
                errors.append(
                    f"output {index} shape {actual_shape}, "
                    f"expected {expected_shape}"
                )
            if actual_dtype != expected_dtype:
                errors.append(
                    f"output {index} dtype {actual_dtype}, "
                    f"expected {expected_dtype}"
                )

        if contract.output_names:
            actual_names = tuple(output.name for output in outputs)
            if actual_names != contract.output_names:
                errors.append(
                    f"output names {actual_names}, "
                    f"expected {contract.output_names}"
                )

    if errors:
        details = "\n  - ".join(errors)
        raise ValueError(
            f"{contract.name.title()} ONNX contract mismatch: {path}\n"
            f"  - {details}"
        )

    print(
        f"Validated {contract.name}: {path} "
        f"({path.stat().st_size:,} bytes)"
    )
    return path


def _temporary_output_path(output_path: Path) -> Path:
    return output_path.with_name(
        f".{output_path.stem}.tmp{output_path.suffix}"
    )


def _needs_export(source: Path, output: Path, *, force: bool) -> bool:
    return (
        force
        or not output.is_file()
        or source.stat().st_mtime_ns > output.stat().st_mtime_ns
    )


def _load_age_dependencies() -> tuple[Any, Any, Any]:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
        import tf2onnx
        from deepface.models.facial_recognition import VGGFace
    except ModuleNotFoundError as exc:
        raise _dependency_error(exc.name or "age export dependency") from exc
    return tf, tf2onnx, VGGFace


def _build_age_model(weights_path: Path, tf: Any, vggface: Any) -> Any:
    """Rebuild DeepFace's age network and load its H5 weights."""
    backbone = vggface.base_model()
    output = tf.keras.layers.Conv2D(
        AGE_CLASSES,
        (1, 1),
        name="predictions",
    )(backbone.layers[-4].output)
    output = tf.keras.layers.Flatten()(output)
    output = tf.keras.layers.Activation("softmax")(output)
    model = tf.keras.Model(
        inputs=backbone.input,
        outputs=output,
        name="deepface_age",
    )
    model.load_weights(str(weights_path))
    return model


def export_age_model(
    weights_path: Path,
    output_path: Path,
    *,
    opset: int = DEFAULT_OPSET,
    onnx: Any | None = None,
) -> Path:
    """Convert DeepFace H5 age weights to an atomic, validated ONNX file."""
    weights_path = weights_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"DeepFace age weights not found: {weights_path}\n"
            "Run scripts/model_download.py before exporting."
        )
    if output_path.suffix.lower() != ".onnx":
        raise ValueError(f"Age output must use the .onnx suffix: {output_path}")

    tf, tf2onnx, vggface = _load_age_dependencies()
    model = _build_age_model(weights_path, tf, vggface)
    input_signature = (
        tf.TensorSpec(AGE_INPUT_SHAPE, tf.float32, name="input"),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = _temporary_output_path(output_path)
    temporary_output.unlink(missing_ok=True)

    print(f"Exporting age model: {weights_path} -> {output_path}")
    try:
        tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=opset,
            output_path=str(temporary_output),
        )
        validate_onnx(temporary_output, AGE_CONTRACT, onnx)
        temporary_output.replace(output_path)
    finally:
        temporary_output.unlink(missing_ok=True)

    return output_path


def _load_checkpoint_state_dict(checkpoint: Any, checkpoint_path: Path) -> dict:
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid gender checkpoint: {checkpoint_path}")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Invalid gender checkpoint state_dict: {checkpoint_path}"
        )
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    return state_dict


def export_gender_model(
    checkpoint_path: Path,
    output_path: Path,
    *,
    opset: int = DEFAULT_OPSET,
    onnx: Any | None = None,
) -> Path:
    """Convert a trained AdaFace gender checkpoint to validated ONNX."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    adaface_source = REPO_ROOT / "thirdparty" / "adaface" / "net.py"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Gender checkpoint not found: {checkpoint_path}"
        )
    if not adaface_source.is_file():
        raise FileNotFoundError(
            "AdaFace source is missing. Initialize the Git submodule first:\n"
            f"  git -C {REPO_ROOT} submodule update --init --recursive"
        )
    if output_path.suffix.lower() != ".onnx":
        raise ValueError(f"Gender output must use the .onnx suffix: {output_path}")

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise _dependency_error(exc.name or "torch") from exc

    from src.model import GenderClassifier
    from thirdparty.adaface.net import Backbone

    backbone = Backbone(input_size=(224, 224), num_layers=50, mode="ir")
    model = GenderClassifier(backbone)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(
        _load_checkpoint_state_dict(checkpoint, checkpoint_path)
    )
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = _temporary_output_path(output_path)
    temporary_output.unlink(missing_ok=True)

    print(f"Exporting gender model: {checkpoint_path} -> {output_path}")
    try:
        torch.onnx.export(
            model,
            (torch.randn(*GENDER_INPUT_SHAPE),),
            str(temporary_output),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch"},
                "logits": {0: "batch"},
            },
            opset_version=opset,
            dynamo=False,
        )
        validate_onnx(temporary_output, GENDER_CONTRACT, onnx)
        temporary_output.replace(output_path)
    finally:
        temporary_output.unlink(missing_ok=True)

    return output_path


def ensure_face_model(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(
            f"Pre-exported face detector ONNX not found: {path}\n"
            "Run scripts/model_download.py before exporting."
        )
    return "pre-exported"


def ensure_age_model(
    weights_path: Path,
    output_path: Path,
    *,
    opset: int,
    force: bool,
    onnx: Any,
) -> str:
    if weights_path.is_file() and _needs_export(
        weights_path,
        output_path,
        force=force,
    ):
        export_age_model(
            weights_path,
            output_path,
            opset=opset,
            onnx=onnx,
        )
        return "exported"
    if output_path.is_file():
        return "existing"
    raise FileNotFoundError(
        f"Neither age weights nor an ONNX model was found:\n"
        f"  weights: {weights_path}\n"
        f"  ONNX:    {output_path}\n"
        "Run scripts/model_download.py before exporting."
    )


def ensure_gender_model(
    checkpoint_path: Path,
    output_path: Path,
    *,
    opset: int,
    force: bool,
    onnx: Any,
) -> str:
    if checkpoint_path.is_file() and _needs_export(
        checkpoint_path,
        output_path,
        force=force,
    ):
        export_gender_model(
            checkpoint_path,
            output_path,
            opset=opset,
            onnx=onnx,
        )
        return "exported"
    if output_path.is_file():
        return "existing"
    raise FileNotFoundError(
        f"Neither a gender checkpoint nor an ONNX model was found:\n"
        f"  checkpoint: {checkpoint_path}\n"
        f"  ONNX:      {output_path}\n"
        "Run scripts/model_download.py to obtain the pre-exported fallback."
    )


def ensure_runtime_models(
    paths: ExportPaths,
    *,
    opset: int = DEFAULT_OPSET,
    force: bool = False,
) -> dict[str, str]:
    """Export available checkpoints and validate all three runtime models."""
    onnx = _load_onnx()
    actions = {
        "face": ensure_face_model(paths.face_model),
        "age": ensure_age_model(
            paths.age_weights,
            paths.age_model,
            opset=opset,
            force=force,
            onnx=onnx,
        ),
        "gender": ensure_gender_model(
            paths.gender_checkpoint,
            paths.gender_model,
            opset=opset,
            force=force,
            onnx=onnx,
        ),
    }

    validate_onnx(paths.face_model, FACE_CONTRACT, onnx)
    validate_onnx(paths.age_model, AGE_CONTRACT, onnx)
    validate_onnx(paths.gender_model, GENDER_CONTRACT, onnx)
    return actions


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = export_paths_from_args(args)
    try:
        actions = ensure_runtime_models(
            paths,
            opset=args.opset,
            force=args.force,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print("\nAll runtime models are ONNX-valid:")
    print(f"  face   [{actions['face']}] {paths.face_model}")
    print(f"  age    [{actions['age']}] {paths.age_model}")
    print(f"  gender [{actions['gender']}] {paths.gender_model}")


if __name__ == "__main__":
    main()
