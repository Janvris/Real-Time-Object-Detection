"""
export.py — Model Download & ONNX Export Script
================================================
Phase 1 of the CPU-Optimized Real-Time Object Detection System.

This script:
  1. Downloads the YOLOv8n nano model weights (smallest/fastest architecture).
  2. Exports the model to ONNX format with CPU-specific optimizations.
  3. Verifies the exported model is valid and prints its input/output shapes.

Usage:
    python export.py
    python export.py --model yolo11n   # Use YOLO11 nano instead
    python export.py --imgsz 320       # Export for 320px inference (faster)
    python export.py --imgsz 640       # Export for 640px inference (more accurate)
"""

import argparse
import os
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and export YOLO nano model to optimized ONNX format"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n",
        choices=["yolov8n", "yolo11n"],
        help="Nano model variant to use (default: yolov8n)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        choices=[320, 416, 640],
        help="Input image size for ONNX export (default: 640). "
             "Use 320 or 416 for faster CPU inference.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to store exported ONNX model (default: ./models)",
    )
    return parser.parse_args()


def check_dependencies():
    """Ensure required packages are installed before proceeding."""
    print("[CHECK] Checking dependencies...")
    missing = []
    try:
        import ultralytics
        print(f"  [OK] ultralytics {ultralytics.__version__}")
    except ImportError:
        missing.append("ultralytics")

    try:
        import onnxruntime as ort
        print(f"  [OK] onnxruntime {ort.__version__}")
        # Warn if GPU version is accidentally installed
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            print(
                "  [WARN] onnxruntime-gpu detected. For pure CPU optimization,\n"
                "        consider uninstalling it and using onnxruntime (CPU-only)."
            )
        else:
            print(f"  [OK] CPU-only ONNX Runtime confirmed. Providers: {providers}")
    except ImportError:
        missing.append("onnxruntime")

    try:
        import cv2
        print(f"  [OK] opencv {cv2.__version__}")
    except ImportError:
        missing.append("opencv-python-headless")

    if missing:
        print(f"\n[ERROR] Missing packages: {', '.join(missing)}")
        print("   Please run: pip install -r requirements.txt")
        sys.exit(1)

    print("  All dependencies satisfied.\n")


def export_model(model_name: str, imgsz: int, output_dir: str) -> Path:
    """
    Download YOLO nano weights and export to ONNX.

    Args:
        model_name: 'yolov8n' or 'yolo11n'
        imgsz:      Input resolution for ONNX graph (320, 416, or 640)
        output_dir: Target directory for the .onnx file

    Returns:
        Path to the exported .onnx file
    """
    from ultralytics import YOLO

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    weights_file = f"{model_name}.pt"
    onnx_output  = output_path / f"{model_name}_imgsz{imgsz}.onnx"

    # -------------------------------------------------------------------------
    # Step 1: Load / Download weights
    # -------------------------------------------------------------------------
    print(f"[DOWNLOAD] Loading model: {weights_file}")
    print("   (Ultralytics will auto-download if not cached locally)\n")
    start = time.time()
    model = YOLO(weights_file)
    print(f"   Model loaded in {time.time() - start:.2f}s\n")

    # Print model info
    print("[INFO] Model Summary:")
    print(f"   Architecture : {model_name} (nano — smallest/fastest)")
    print(f"   Task         : Object Detection")
    print(f"   Classes      : {model.names}\n")

    # -------------------------------------------------------------------------
    # Step 2: Export to ONNX
    # -------------------------------------------------------------------------
    print("[EXPORT] Exporting to ONNX...")
    print(f"   Target input size : {imgsz}×{imgsz} px")
    print(f"   Output file       : {onnx_output}\n")

    export_start = time.time()
    exported_path = model.export(
        format="onnx",         # Export to ONNX format
        imgsz=imgsz,           # Fixed input resolution baked into graph
        dynamic=False,         # Static shapes = faster CPU inference
        simplify=True,         # Run ONNX simplifier to remove redundant ops
        opset=17,              # ONNX opset 17 — well-supported & optimized
        half=False,            # FP16 disabled — CPUs prefer FP32
    )
    export_time = time.time() - export_start
    print(f"\n   Export completed in {export_time:.2f}s")

    # Move to our models/ directory if not already there
    exported_file = Path(exported_path)
    if exported_file.resolve() != onnx_output.resolve():
        import shutil
        shutil.move(str(exported_file), str(onnx_output))
        print(f"   Moved to: {onnx_output}")

    return onnx_output


def verify_model(onnx_path: Path):
    """
    Load the exported ONNX model with ONNX Runtime and print input/output specs.
    This confirms the model is valid and ready for CPU inference.
    """
    import onnxruntime as ort
    import numpy as np

    print(f"\n[VERIFY] Verifying ONNX model: {onnx_path.name}")
    print("-" * 55)

    # Session options for CPU optimization
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = os.cpu_count()  # Use all CPU cores

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    # Print input details
    print("\n[INPUT] Model Inputs:")
    for inp in session.get_inputs():
        print(f"   Name  : {inp.name}")
        print(f"   Shape : {inp.shape}")
        print(f"   Type  : {inp.type}")

    # Print output details
    print("\n[OUTPUT] Model Outputs:")
    for out in session.get_outputs():
        print(f"   Name  : {out.name}")
        print(f"   Shape : {out.shape}")
        print(f"   Type  : {out.type}")

    # Run a warm-up inference to confirm it works end-to-end
    print("\n[WARMUP] Running warm-up inference...")
    inp = session.get_inputs()[0]
    _, _, h, w = inp.shape  # (batch, channels, height, width)
    dummy = np.random.rand(1, 3, h, w).astype(np.float32)
    
    warmup_start = time.time()
    outputs = session.run(None, {inp.name: dummy})
    warmup_time = (time.time() - warmup_start) * 1000

    print("   [OK] Inference successful!")
    print(f"   Output tensor shape : {outputs[0].shape}")
    print(f"   Warm-up latency     : {warmup_time:.1f} ms")
    print(f"   Estimated max FPS   : ~{1000/warmup_time:.0f} FPS (single thread, no pre/post)")

    print("\n" + "-" * 55)
    print("[READY] Model is ready for deployment!\n")


def print_usage_guide(onnx_path: Path):
    """Print next steps for the user."""
    print("=" * 55)
    print("NEXT STEPS")
    print("=" * 55)
    print(f"\n[DONE] ONNX model saved to: {onnx_path}")
    print("\nThe model path is used automatically by detector.py.")
    print("You can also override it via the MODEL_PATH env variable:\n")
    print(f"   set MODEL_PATH={onnx_path}")
    print("  app.py       — Streamlit UI\n")


def main():
    print("=" * 55)
    print("  CPU-Optimized Object Detection — Phase 1 Setup")
    print("=" * 55)
    print()

    args = parse_args()

    # 1. Verify environment
    check_dependencies()

    # 2. Download + export model
    onnx_path = export_model(
        model_name=args.model,
        imgsz=args.imgsz,
        output_dir=args.output_dir,
    )

    # 3. Verify the exported model
    verify_model(onnx_path)

    # 4. Print usage guide
    print_usage_guide(onnx_path)


if __name__ == "__main__":
    main()
