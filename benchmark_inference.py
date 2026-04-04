"""Benchmark YOLO inference speed on Apple Silicon.

This script measures end-to-end model inference latency/FPS for:
- PyTorch on MPS (Apple GPU) when available
- PyTorch on CPU
- Core ML model (optional), which may leverage Apple Neural Engine

Examples:
    python benchmark_inference.py
    python benchmark_inference.py --runs 120 --warmup 20
    python benchmark_inference.py --export-coreml
    python benchmark_inference.py --coreml-model model/yolov10/YOLOv10b_VietFood67_SGD_new_bigger.mlpackage
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Prefer pip-installed ultralytics over the local workspace copy.
_SCRIPT_DIR = Path(__file__).resolve().parent
_ORIG_SYSPATH = list(sys.path)
sys.path = [p for p in sys.path if Path(p or ".").resolve() != _SCRIPT_DIR]
try:
    from ultralytics import YOLO
finally:
    sys.path = _ORIG_SYSPATH

try:
    import coremltools as ct
except Exception:  # pragma: no cover
    ct = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

DEFAULT_MODEL = "model/yolov10/YOLOv10b_VietFood67_SGD_new_bigger.pt"
DEFAULT_IMAGE = "noodle.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLO inference backends.")
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL), help="Path to .pt model.")
    parser.add_argument("--image", type=Path, default=Path(DEFAULT_IMAGE), help="Path to test image.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.65, help="Confidence threshold.")
    parser.add_argument("--warmup", type=int, default=15, help="Warmup iterations.")
    parser.add_argument("--runs", type=int, default=80, help="Timed iterations.")
    parser.add_argument(
        "--export-coreml",
        action="store_true",
        help="Export provided .pt model to Core ML and benchmark it.",
    )
    parser.add_argument(
        "--coreml-model",
        type=Path,
        default=None,
        help="Path to existing Core ML model (.mlmodel/.mlpackage) to benchmark.",
    )
    parser.add_argument("--skip-mps", action="store_true", help="Skip MPS benchmark.")
    parser.add_argument("--skip-cpu", action="store_true", help="Skip CPU benchmark.")
    parser.add_argument(
        "--npu-vs-gpu-only",
        action="store_true",
        help="Run only Core ML NPU-vs-GPU benchmark (requires Core ML model).",
    )
    parser.add_argument(
        "--coreml-npu-only",
        action="store_true",
        help="Run only Core ML NPU benchmark (CPU_AND_NE).",
    )
    return parser.parse_args()


def sync_device(device: str | None) -> None:
    if torch is None:
        return
    if device == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_backend(
    model_path: Path,
    image_bgr: Any,
    imgsz: int,
    conf: float,
    warmup: int,
    runs: int,
    device: str | None,
    label: str,
) -> dict[str, float | str]:
    model = YOLO(str(model_path))
    kwargs: dict[str, Any] = {"imgsz": imgsz, "conf": conf, "verbose": False}
    if device is not None:
        kwargs["device"] = device

    for _ in range(max(0, warmup)):
        model.predict(image_bgr, **kwargs)
    sync_device(device)

    durations_ms: list[float] = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        model.predict(image_bgr, **kwargs)
        sync_device(device)
        durations_ms.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = statistics.fmean(durations_ms)
    p50_ms = statistics.median(durations_ms)
    p95_ms = sorted(durations_ms)[int(len(durations_ms) * 0.95) - 1]
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    return {
        "backend": label,
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "fps": fps,
    }


def maybe_export_coreml(pt_model_path: Path, imgsz: int) -> Path:
    model = YOLO(str(pt_model_path))

    # Some custom YOLO checkpoints fail Core ML export when the export wrapper
    # applies NMS with a class-count split assumption. For benchmarking runtime,
    # we can export without NMS and still compare compute units fairly.
    export_attempts = [
        {"half": True, "nms": True},
        {"half": True, "nms": False},
        {"half": False, "nms": False},
    ]

    last_error: Exception | None = None
    for i, opts in enumerate(export_attempts, start=1):
        try:
            print(
                f"[Info] Core ML export attempt {i}/{len(export_attempts)} "
                f"(half={opts['half']}, nms={opts['nms']})"
            )
            exported = model.export(format="coreml", imgsz=imgsz, **opts)
            return Path(exported)
        except Exception as exc:  # pragma: no cover
            last_error = exc
            print(f"[Warn] Core ML export attempt {i} failed: {exc}")

    raise RuntimeError(
        "All Core ML export attempts failed. "
        "Try providing an existing exported model via --coreml-model."
    ) from last_error


def _prepare_coreml_input(mlmodel: Any, image_bgr: Any, imgsz: int) -> dict[str, Any]:
    spec = mlmodel.get_spec()
    if not spec.description.input:
        raise RuntimeError("Core ML model has no inputs.")

    inp = spec.description.input[0]
    input_name = inp.name
    input_type = inp.type.WhichOneof("Type")

    if input_type == "imageType":
        if Image is None:
            raise RuntimeError("Pillow is required for image-type Core ML inputs.")

        h = int(inp.type.imageType.height) if int(inp.type.imageType.height) > 0 else imgsz
        w = int(inp.type.imageType.width) if int(inp.type.imageType.width) > 0 else imgsz
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        return {input_name: Image.fromarray(rgb)}

    if input_type == "multiArrayType":
        shape = list(inp.type.multiArrayType.shape)
        if not shape:
            shape = [1, 3, imgsz, imgsz]

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if len(shape) == 4 and shape[1] in (1, 3):
            h, w = int(shape[2]), int(shape[3])
            resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            arr = resized.astype(np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))[None, ...]
        elif len(shape) == 4 and shape[-1] in (1, 3):
            h, w = int(shape[1]), int(shape[2])
            resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            arr = (resized.astype(np.float32) / 255.0)[None, ...]
        else:
            raise RuntimeError(f"Unsupported Core ML input shape: {shape}")

        return {input_name: arr}

    raise RuntimeError(f"Unsupported Core ML input type: {input_type}")


def benchmark_coreml_compute_unit(
    coreml_model_path: Path,
    image_bgr: Any,
    imgsz: int,
    warmup: int,
    runs: int,
    compute_unit: Any,
    label: str,
) -> dict[str, float | str]:
    if ct is None:
        raise RuntimeError("coremltools is not installed.")

    mlmodel = ct.models.MLModel(str(coreml_model_path), compute_units=compute_unit)
    model_input = _prepare_coreml_input(mlmodel, image_bgr, imgsz)

    for _ in range(max(0, warmup)):
        mlmodel.predict(model_input)

    durations_ms: list[float] = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        mlmodel.predict(model_input)
        durations_ms.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = statistics.fmean(durations_ms)
    p50_ms = statistics.median(durations_ms)
    p95_ms = sorted(durations_ms)[int(len(durations_ms) * 0.95) - 1]
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    return {
        "backend": label,
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "fps": fps,
    }


def print_report(rows: list[dict[str, float | str]]) -> None:
    if not rows:
        print("No benchmark results.")
        return

    print("\n=== Inference Benchmark ===")
    print(f"{'Backend':28s} {'Avg(ms)':>10s} {'P50(ms)':>10s} {'P95(ms)':>10s} {'FPS':>10s}")
    print("-" * 72)
    for row in rows:
        print(
            f"{str(row['backend']):28s} "
            f"{float(row['avg_ms']):10.2f} "
            f"{float(row['p50_ms']):10.2f} "
            f"{float(row['p95_ms']):10.2f} "
            f"{float(row['fps']):10.2f}"
        )

    fastest = max(rows, key=lambda r: float(r["fps"]))
    print("\nFastest:", fastest["backend"], f"({float(fastest['fps']):.2f} FPS)")


def main() -> None:
    args = parse_args()

    model_path = args.model.resolve()
    image_path = args.image.resolve()

    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")
    if not image_path.exists():
        raise SystemExit(f"Image file not found: {image_path}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise SystemExit(f"Failed to read image: {image_path}")

    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print(f"Runs: warmup={args.warmup}, timed={args.runs}, imgsz={args.imgsz}\n")

    rows: list[dict[str, float | str]] = []

    mps_available = bool(torch is not None and hasattr(torch, "backends") and torch.backends.mps.is_available())

    if not args.npu_vs_gpu_only:
        if not args.skip_mps:
            if mps_available:
                rows.append(
                    benchmark_backend(
                        model_path=model_path,
                        image_bgr=image_bgr,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        warmup=args.warmup,
                        runs=args.runs,
                        device="mps",
                        label="PyTorch MPS (Apple GPU)",
                    )
                )
            else:
                print("[Skip] MPS backend unavailable on this setup.")

        if not args.skip_cpu:
            rows.append(
                benchmark_backend(
                    model_path=model_path,
                    image_bgr=image_bgr,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    warmup=args.warmup,
                    runs=args.runs,
                    device="cpu",
                    label="PyTorch CPU",
                )
            )

    coreml_path = args.coreml_model.resolve() if args.coreml_model else None
    if args.export_coreml:
        print("[Info] Exporting Core ML model...")
        coreml_path = maybe_export_coreml(model_path, args.imgsz).resolve()
        print(f"[Info] Exported Core ML model: {coreml_path}")

    if coreml_path is not None:
        if not coreml_path.exists():
            print(f"[Skip] Core ML model path does not exist: {coreml_path}")
        else:
            if ct is None:
                print("[Skip] coremltools is not installed; cannot run NPU-vs-GPU Core ML benchmark.")
            else:
                if not args.coreml_npu_only:
                    rows.append(
                        benchmark_coreml_compute_unit(
                            coreml_model_path=coreml_path,
                            image_bgr=image_bgr,
                            imgsz=args.imgsz,
                            warmup=args.warmup,
                            runs=args.runs,
                            compute_unit=ct.ComputeUnit.CPU_AND_GPU,
                            label="Core ML GPU (CPU_AND_GPU)",
                        )
                    )
                rows.append(
                    benchmark_coreml_compute_unit(
                        coreml_model_path=coreml_path,
                        image_bgr=image_bgr,
                        imgsz=args.imgsz,
                        warmup=args.warmup,
                        runs=args.runs,
                        compute_unit=ct.ComputeUnit.CPU_AND_NE,
                        label="Core ML NPU (CPU_AND_NE)",
                    )
                )

    print_report(rows)


if __name__ == "__main__":
    main()
