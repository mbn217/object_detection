"""
convert_model.py

Converts a YOLO .pt model to the optimal format for the current system,
or to a user-specified format.

Auto-detection logic:
  Windows  x86/x64           → ONNX   (ONNX Runtime, OpenVINO, TensorRT ready)
  Linux    ARM (Pi / Jetson) → NCNN   (mobile/embedded optimised)
  Linux    x86/x64           → ONNX
  macOS    Apple Silicon     → NCNN   (or CoreML via --format coreml)
  macOS    Intel             → ONNX
  Any      NVIDIA GPU found  → ONNX   (TensorRT-ready; override with --format ncnn)

Usage:
  # Auto-detect format
  python convert_model.py yolo26s.pt

  # Force a specific format
  python convert_model.py yolo26s.pt --format ncnn
  python convert_model.py yolo26s.pt --format onnx
  python convert_model.py yolo26s.pt --format openvino

  # Extra export options
  python convert_model.py yolo26s.pt --half --imgsz 640
  python convert_model.py yolo26s.pt --format onnx --dynamic --simplify
"""

import argparse
import platform
import sys
import os


def detect_platform() -> dict:
    """Return a dict describing the current hardware / OS."""
    info = {
        "os":      platform.system(),          # Windows | Linux | Darwin
        "arch":    platform.machine().lower(), # x86_64 | amd64 | aarch64 | arm64 | armv7l
        "is_arm":  False,
        "has_cuda": False,
        "has_apple_silicon": False,
    }

    arm_archs = {"aarch64", "arm64", "armv7l", "armv8l"}
    info["is_arm"] = info["arch"] in arm_archs

    # Apple Silicon
    if info["os"] == "Darwin" and info["arch"] in {"arm64", "aarch64"}:
        info["has_apple_silicon"] = True

    # NVIDIA GPU (optional — don't hard-fail if torch not installed)
    try:
        import torch
        info["has_cuda"] = torch.cuda.is_available()
    except ImportError:
        pass

    return info


def choose_format(info: dict) -> str:
    """Return the recommended export format for the detected platform."""
    if info["os"] == "Windows":
        return "onnx"          # ONNX Runtime ships on Windows; also OpenVINO / TRT ready

    if info["os"] == "Darwin":
        if info["has_apple_silicon"]:
            return "ncnn"      # NCNN runs well on Apple Silicon; use --format coreml for CoreML
        return "onnx"

    # Linux
    if info["is_arm"]:
        return "ncnn"          # Raspberry Pi, Jetson Nano, Android-based boards
    return "onnx"              # Linux x86 — ONNX Runtime or OpenVINO downstream


FORMAT_NOTES = {
    "onnx": (
        "ONNX — cross-platform, universal format.\n"
        "  • Load with: YOLO('model.onnx')\n"
        "  • Works with ONNX Runtime, OpenVINO, TensorRT\n"
        "  • Best for: Windows, Linux x86, cloud/server deployment"
    ),
    "ncnn": (
        "NCNN — Tencent's mobile / embedded inference engine.\n"
        "  • Load with: YOLO('model_ncnn_model/')\n"
        "  • Supports Vulkan GPU acceleration\n"
        "  • Best for: Raspberry Pi, Android, Apple Silicon, Jetson"
    ),
    "openvino": (
        "OpenVINO — Intel acceleration toolkit.\n"
        "  • Load with: YOLO('model_openvino_model/')\n"
        "  • Supports Intel CPU, iGPU, NPU\n"
        "  • Best for: Intel hardware, up to 3× CPU speedup"
    ),
    "coreml": (
        "CoreML — Apple on-device inference.\n"
        "  • Best for: macOS (12+), iOS, iPadOS\n"
        "  • Leverages Apple Neural Engine"
    ),
    "tflite": (
        "TFLite — TensorFlow Lite.\n"
        "  • Best for: Android, embedded Linux\n"
        "  • INT8 quantisation supported"
    ),
}


def build_export_kwargs(args: argparse.Namespace) -> dict:
    """Build the kwargs dict to pass to model.export()."""
    kwargs = {
        "format": args.format,
        "imgsz":  args.imgsz,
        "half":   args.half,
    }
    if args.format == "onnx":
        kwargs["dynamic"]  = args.dynamic
        kwargs["simplify"] = args.simplify
        if args.opset:
            kwargs["opset"] = args.opset
    if args.batch and args.batch > 1:
        kwargs["batch"] = args.batch
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a YOLO .pt model to ONNX or NCNN (auto-detected from system).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model", help="Path to the .pt model file (e.g. yolo26s.pt)")
    parser.add_argument(
        "--format",
        default=None,
        choices=["onnx", "ncnn", "openvino", "coreml", "tflite"],
        help="Export format. Auto-detected from system if omitted.",
    )
    parser.add_argument("--imgsz",    type=int,   default=640,   help="Input image size (default: 640)")
    parser.add_argument("--half",     action="store_true",        help="FP16 export (smaller, faster on supported HW)")
    parser.add_argument("--dynamic",  action="store_true",        help="ONNX: dynamic input shapes")
    parser.add_argument("--simplify", action="store_true", default=True, help="ONNX: simplify graph (default: True)")
    parser.add_argument("--opset",    type=int,   default=None,   help="ONNX: opset version")
    parser.add_argument("--batch",    type=int,   default=1,      help="Export batch size (default: 1)")

    args = parser.parse_args()

    # ── Validate input ────────────────────────────────────────────────────────
    if not os.path.isfile(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        sys.exit(1)

    # ── Platform detection ───────────────────────────────────────────────────
    info = detect_platform()
    auto_format = choose_format(info)

    print("\n" + "═" * 55)
    print(" YOLO Model Converter")
    print("═" * 55)
    print(f"  OS       : {info['os']} ({info['arch']})")
    print(f"  ARM      : {info['is_arm']}")
    print(f"  CUDA GPU : {info['has_cuda']}")
    print(f"  Apple Si : {info['has_apple_silicon']}")
    print(f"  Model    : {args.model}")

    if args.format:
        chosen = args.format
        print(f"  Format   : {chosen}  (user override)")
    else:
        chosen = auto_format
        print(f"  Format   : {chosen}  (auto-detected)")

    print("\n" + FORMAT_NOTES.get(chosen, "") + "\n")
    print("═" * 55)

    args.format = chosen

    # ── Import and run export ─────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics is not installed. Run: pip install -U ultralytics")
        sys.exit(1)

    print(f"[INFO] Loading {args.model} ...")
    model = YOLO(args.model)

    export_kwargs = build_export_kwargs(args)
    print(f"[INFO] Exporting with settings: {export_kwargs}")

    output_path = model.export(**export_kwargs)

    print("\n" + "═" * 55)
    print(f"[DONE] Export complete!")
    print(f"       Output : {output_path}")
    print("═" * 55)

    # ── Quick usage hint ──────────────────────────────────────────────────────
    if chosen == "onnx":
        print(f"\n  Load: YOLO('{output_path}')")
    elif chosen in {"ncnn", "openvino"}:
        out_dir = str(output_path)
        print(f"\n  Load: YOLO('{out_dir}')")
        if chosen == "ncnn":
            print(f"  Vulkan: YOLO('{out_dir}').predict(..., device='vulkan:0')")


if __name__ == "__main__":
    main()
