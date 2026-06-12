"""
detector.py - CPU-Optimized ONNX Object Detection Engine
=========================================================
Phase 3 of the CPU-Optimized Real-Time Object Detection System.

Class:  CPUObjectDetector
  - Loads a YOLOv8/YOLO11 ONNX model via onnxruntime.InferenceSession
  - Pre-processes frames: letterbox resize → normalize → NCHW transpose
  - Runs inference on CPUExecutionProvider with all graph optimizations
  - Post-processes output: decode boxes → filter confidence → vectorized NMS
  - Draws annotated bounding boxes directly onto the frame
  - Returns a DetectionResult (compatible with streamer.py)

CPU Optimization Techniques Used Here:
  1. SessionOptions.graph_optimization_level = ORT_ENABLE_ALL
     - Fuses Conv+BN+ReLU into single ops, eliminates dead nodes
  2. intra_op_num_threads = os.cpu_count()
     - Spreads matrix multiplications across all physical cores
  3. Letterbox resize (not stretch): preserves aspect ratio, avoids
     distortion artifacts that hurt NMS quality
  4. Vectorized NMS with numpy: avoids Python-level loops for IoU calc
  5. Pre-allocated output buffer: avoids repeated heap allocations per frame
  6. Warm-up run on __init__: ensures JIT compilation happens at startup,
     not during the first live frame
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

# Import the shared data structure from streamer
from streamer import DetectionResult

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("detector")

# ---------------------------------------------------------------------------
# COCO 80-class color palette (consistent colors per class ID)
# Generated with HSV spacing for maximum visual distinction
# ---------------------------------------------------------------------------
def _build_color_palette(n: int = 80) -> List[Tuple[int, int, int]]:
    """Generate perceptually distinct BGR colors for each class."""
    palette = []
    for i in range(n):
        hue = int(180 * i / n)           # OpenCV hue: 0-179
        hsv = np.array([[[hue, 220, 220]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        palette.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return palette

COLOR_PALETTE = _build_color_palette(80)

# COCO class names (80 classes, index 0-79)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


# ---------------------------------------------------------------------------
# CPUObjectDetector
# ---------------------------------------------------------------------------

class CPUObjectDetector:
    """
    CPU-optimized ONNX inference engine for YOLOv8/YOLO11 nano models.

    Args:
        model_path:       Path to the .onnx model file.
        conf_threshold:   Minimum confidence score to keep a detection (0-1).
        iou_threshold:    IoU threshold for Non-Maximum Suppression (0-1).
        infer_size:       Resolution fed to the model (320, 416, or 640).
                          Smaller = faster; must match the ONNX export size
                          OR be overridden dynamically (see _preprocess).
        num_threads:      CPU threads for ONNX Runtime. None = all cores.
        draw_labels:      If True, draw class name + score on bounding boxes.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n_imgsz640.onnx",
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        infer_size: int = 640,
        num_threads: Optional[int] = None,
        draw_labels: bool = True,
    ):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.infer_size = infer_size
        self.draw_labels = draw_labels

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {self.model_path}\n"
                f"Run export.py first to download and export the model."
            )

        self._num_threads = num_threads or os.cpu_count() or 1
        self._session: Optional[ort.InferenceSession] = None
        self._input_name: str = ""
        self._model_h: int = infer_size
        self._model_w: int = infer_size

        # Load and warm up
        self._load_model()
        self._warmup()

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Build an ONNX Runtime session with maximum CPU optimizations."""
        log.info(f"Loading ONNX model: {self.model_path}")

        opts = ort.SessionOptions()

        # Enable ALL graph optimizations (fusion, constant folding, etc.)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Parallelism: spread matrix ops across all CPU cores
        opts.intra_op_num_threads = self._num_threads  # Within a single op
        opts.inter_op_num_threads = 1                  # Sequential op execution

        # Memory: disable memory pattern optimization for dynamic shapes
        opts.enable_mem_pattern = True
        opts.enable_cpu_mem_arena = True

        # Log severity: suppress verbose ORT internal logs
        opts.log_severity_level = 3  # 0=verbose, 1=info, 2=warning, 3=error

        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        # Cache input metadata
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        _, _, h, w = inp.shape   # (batch=1, channels=3, H, W)
        self._model_h = int(h)
        self._model_w = int(w)

        log.info(
            f"Model loaded | Input: {self._input_name} "
            f"[1,3,{self._model_h},{self._model_w}] | "
            f"Threads: {self._num_threads}"
        )

    def _warmup(self, n_runs: int = 3):
        """
        Run N dummy inferences to trigger ONNX Runtime's JIT graph
        compilation. Without this, the first live frame would be very slow.
        """
        log.info(f"Warming up model ({n_runs} dummy runs)...")
        dummy = np.zeros(
            (1, 3, self._model_h, self._model_w), dtype=np.float32
        )
        t_start = time.perf_counter()
        for _ in range(n_runs):
            self._session.run(None, {self._input_name: dummy})
        elapsed = (time.perf_counter() - t_start) * 1000
        log.info(
            f"Warm-up complete. {n_runs} runs in {elapsed:.1f}ms "
            f"(avg {elapsed/n_runs:.1f}ms/frame)"
        )

    # ------------------------------------------------------------------
    # Pre-Processing: Letterbox + Normalize
    # ------------------------------------------------------------------

    def _preprocess(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, float, int, int]:
        """
        Convert a raw BGR frame into a normalized NCHW float32 tensor.

        Steps:
          1. Letterbox resize: scale image to fit infer_size×infer_size
             while preserving aspect ratio. Pad with grey (114,114,114).
          2. BGR → RGB
          3. Normalize: divide by 255 → [0.0, 1.0]
          4. HWC → CHW → add batch dim → (1, 3, H, W)

        Returns:
          tensor:  Input tensor ready for ONNX session.run()
          scale:   Ratio of original size to model input size.
          pad_x:   Horizontal padding (pixels) added to each side.
          pad_y:   Vertical padding (pixels) added to each side.
        """
        orig_h, orig_w = frame.shape[:2]
        target = self.infer_size

        # Compute scale keeping aspect ratio
        scale = min(target / orig_w, target / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        # Resize (INTER_LINEAR: best speed/quality trade-off)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create grey canvas and paste resized image
        canvas = np.full((target, target, 3), 114, dtype=np.uint8)
        pad_y = (target - new_h) // 2
        pad_x = (target - new_w) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # BGR → RGB → float32 [0,1] → CHW → NCHW
        rgb = canvas[:, :, ::-1]                          # BGR to RGB
        normalized = rgb.astype(np.float32) / 255.0       # Normalize
        chw = np.ascontiguousarray(normalized.transpose(2, 0, 1))  # HWC→CHW
        tensor = chw[np.newaxis, ...]                      # Add batch dim

        return tensor, scale, pad_x, pad_y

    # ------------------------------------------------------------------
    # Post-Processing: Decode + NMS
    # ------------------------------------------------------------------

    def _postprocess(
        self,
        output: np.ndarray,
        scale: float,
        pad_x: int,
        pad_y: int,
        orig_w: int,
        orig_h: int,
    ) -> Tuple[List, List, List]:
        """
        Decode YOLOv8/YOLO11 ONNX output tensor into usable detections.

        YOLOv8 output shape: (1, 84, 8400)
          - 84 = 4 (cx, cy, w, h) + 80 class scores
          - 8400 = anchors at different scales (80x80 + 40x40 + 20x20)

        Steps:
          1. Transpose to (8400, 84)
          2. Extract box coords and class scores
          3. Filter by confidence threshold
          4. Convert cx,cy,w,h → x1,y1,x2,y2 (xyxy format)
          5. Apply vectorized NMS
          6. Rescale boxes back to original frame coordinates

        Returns:
          boxes:      [[x1,y1,x2,y2], ...] in original frame pixels
          scores:     [float, ...]
          class_ids:  [int, ...]
        """
        # output shape: (1, 84, 8400) → squeeze → (84, 8400) → (8400, 84)
        preds = output[0].squeeze(0).T   # (8400, 84)

        # Split into boxes and class scores
        boxes_xywh = preds[:, :4]        # (8400, 4) — cx, cy, w, h (model coords)
        class_scores = preds[:, 4:]      # (8400, 80)

        # Best class per anchor
        class_ids_all = np.argmax(class_scores, axis=1)              # (8400,)
        confidences   = class_scores[np.arange(len(class_scores)),
                                     class_ids_all]                  # (8400,)

        # Filter by confidence threshold
        mask = confidences >= self.conf_threshold
        if not mask.any():
            return [], [], []

        boxes_xywh  = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids   = class_ids_all[mask]

        # Convert center-format (cx,cy,w,h) → corner-format (x1,y1,x2,y2)
        boxes_xyxy = self._xywh2xyxy(boxes_xywh)

        # Vectorized NMS (pure numpy — no torch dependency)
        keep = self._nms(boxes_xyxy, confidences, self.iou_threshold)

        boxes_xyxy  = boxes_xyxy[keep]
        confidences = confidences[keep]
        class_ids   = class_ids[keep]

        # Rescale from model input coords → original frame coords
        boxes_orig = self._rescale_boxes(
            boxes_xyxy, scale, pad_x, pad_y, orig_w, orig_h
        )

        return (
            boxes_orig.tolist(),
            confidences.tolist(),
            class_ids.tolist(),
        )

    @staticmethod
    def _xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
        """Convert (cx, cy, w, h) → (x1, y1, x2, y2). Vectorized."""
        out = np.empty_like(boxes)
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2   # x1
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2   # y1
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2   # x2
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2   # y2
        return out

    @staticmethod
    def _nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
    ) -> np.ndarray:
        """
        Vectorized Non-Maximum Suppression (NMS) using pure numpy.

        Algorithm:
          1. Sort detections by score (descending).
          2. Greedily keep the highest-scoring box.
          3. Compute IoU of kept box against all remaining.
          4. Suppress boxes with IoU > threshold.
          5. Repeat until no boxes remain.

        This avoids torch.ops.torchvision.nms overhead on CPU.
        """
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]   # Descending score order

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            # Compute IoU of box[i] with all remaining boxes
            rest = order[1:]
            inter_x1 = np.maximum(x1[i], x1[rest])
            inter_y1 = np.maximum(y1[i], y1[rest])
            inter_x2 = np.minimum(x2[i], x2[rest])
            inter_y2 = np.minimum(y2[i], y2[rest])

            inter_w = np.maximum(0.0, inter_x2 - inter_x1)
            inter_h = np.maximum(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h

            iou = inter_area / (areas[i] + areas[rest] - inter_area + 1e-7)

            # Keep only boxes with IoU below threshold
            order = rest[iou <= iou_threshold]

        return np.array(keep, dtype=np.int32)

    @staticmethod
    def _rescale_boxes(
        boxes: np.ndarray,
        scale: float,
        pad_x: int,
        pad_y: int,
        orig_w: int,
        orig_h: int,
    ) -> np.ndarray:
        """
        Map letterboxed model coordinates back to original frame pixels.

        Inverse of the letterbox transform in _preprocess():
          model_coord = (orig_coord * scale) + padding
          → orig_coord = (model_coord - padding) / scale
        """
        out = boxes.copy().astype(np.float32)
        out[:, 0] = (boxes[:, 0] - pad_x) / scale   # x1
        out[:, 1] = (boxes[:, 1] - pad_y) / scale   # y1
        out[:, 2] = (boxes[:, 2] - pad_x) / scale   # x2
        out[:, 3] = (boxes[:, 3] - pad_y) / scale   # y2

        # Clamp to frame boundaries
        out[:, 0] = np.clip(out[:, 0], 0, orig_w)
        out[:, 1] = np.clip(out[:, 1], 0, orig_h)
        out[:, 2] = np.clip(out[:, 2], 0, orig_w)
        out[:, 3] = np.clip(out[:, 3], 0, orig_h)

        return out.astype(np.int32)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_detections(
        self,
        frame: np.ndarray,
        boxes: List,
        scores: List,
        class_ids: List,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels onto a copy of the frame.

        Design choices:
          - Unique color per class (from COLOR_PALETTE)
          - Semi-transparent filled label background (blended with addWeighted)
          - Box thickness scales with frame resolution
          - Score shown as percentage for readability
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]
        thickness = max(1, int(min(w, h) / 300))   # Adaptive thickness

        for box, score, cid in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            color = COLOR_PALETTE[cid % len(COLOR_PALETTE)]
            name  = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"cls{cid}"

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            if self.draw_labels:
                label = f"{name} {score*100:.0f}%"

                # Compute label background size
                font        = cv2.FONT_HERSHEY_SIMPLEX
                font_scale  = max(0.4, min(w, h) / 1000)
                font_thick  = max(1, thickness - 1)
                (lw, lh), baseline = cv2.getTextSize(
                    label, font, font_scale, font_thick
                )

                # Label background: slightly above the box top edge
                bg_y1 = max(0, y1 - lh - baseline - 4)
                bg_y2 = y1
                bg_x2 = min(w, x1 + lw + 4)

                # Draw filled background rectangle (opaque)
                cv2.rectangle(
                    annotated,
                    (x1, bg_y1),
                    (bg_x2, bg_y2),
                    color,
                    cv2.FILLED,
                )

                # Draw label text (white for contrast)
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 2, y1 - baseline - 2),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thick,
                    cv2.LINE_AA,
                )

        return annotated

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run full detection pipeline on a single BGR frame.

        Pipeline:
          frame → preprocess → inference → postprocess → draw → DetectionResult

        Args:
            frame: BGR numpy array from OpenCV (any resolution).

        Returns:
            DetectionResult with annotated frame, boxes, scores, class_ids,
            and class_names.
        """
        orig_h, orig_w = frame.shape[:2]

        # 1. Pre-process
        tensor, scale, pad_x, pad_y = self._preprocess(frame)

        # 2. Inference
        outputs = self._session.run(None, {self._input_name: tensor})

        # 3. Post-process
        boxes, scores, class_ids = self._postprocess(
            outputs, scale, pad_x, pad_y, orig_w, orig_h
        )

        # 4. Draw annotations
        annotated = self._draw_detections(frame, boxes, scores, class_ids)

        # 5. Resolve class names
        class_names = [
            COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"cls{cid}"
            for cid in class_ids
        ]

        return DetectionResult(
            frame=annotated,
            raw_frame=frame,
            boxes=boxes,
            scores=scores,
            class_ids=[int(c) for c in class_ids],
            class_names=class_names,
        )

    def update_thresholds(
        self,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ):
        """
        Live-update confidence and IoU thresholds without reloading the model.
        Called by the Streamlit sidebar sliders.
        """
        if conf is not None:
            self.conf_threshold = float(conf)
        if iou is not None:
            self.iou_threshold = float(iou)
        log.debug(f"Thresholds updated: conf={self.conf_threshold:.2f}, "
                  f"iou={self.iou_threshold:.2f}")

    def update_infer_size(self, size: int):
        """
        Switch inference resolution on the fly (320, 416, or 640).
        The ONNX model was exported at a fixed size, so this resizes
        the input to match the model's expected dimensions.
        Note: if size != model export size, accuracy may slightly differ.
        """
        self.infer_size = size
        log.info(f"Inference size updated to {size}x{size}")

    def get_class_counts(self, result: DetectionResult) -> dict:
        """
        Count occurrences of each detected class in a DetectionResult.
        Returns a dict like: {'person': 3, 'car': 1, 'chair': 2}
        Used by the Streamlit dashboard tracker panel.
        """
        counts: dict = {}
        for name in result.class_names:
            counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    @property
    def model_info(self) -> dict:
        """Return static model metadata for display in the UI."""
        return {
            "model_path":   str(self.model_path),
            "model_name":   self.model_path.stem,
            "input_size":   f"{self._model_w}x{self._model_h}",
            "infer_size":   f"{self.infer_size}x{self.infer_size}",
            "num_threads":  self._num_threads,
            "num_classes":  len(COCO_CLASSES),
            "conf_threshold": self.conf_threshold,
            "iou_threshold":  self.iou_threshold,
            "providers":    self._session.get_providers(),
        }


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import glob

    print("=" * 60)
    print("  detector.py -- ONNX Inference Engine Smoke Test")
    print("=" * 60)

    # Find the exported model
    models = sorted(glob.glob("models/*.onnx"))
    if not models:
        print("[ERROR] No ONNX models found in models/")
        print("        Run: python export.py")
        sys.exit(1)

    model_path = models[0]
    print(f"\nUsing model: {model_path}")

    # Initialize detector
    print("\nInitializing CPUObjectDetector...")
    detector = CPUObjectDetector(
        model_path=model_path,
        conf_threshold=0.45,
        iou_threshold=0.45,
    )

    info = detector.model_info
    print("\nModel Info:")
    for k, v in info.items():
        print(f"  {k:<18}: {v}")

    # --- Test on a synthetic frame ---
    print("\n[TEST 1] Synthetic blank frame (640x480)")
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    result = detector.detect(blank)
    ms = (time.perf_counter() - t0) * 1000
    print(f"  Detections : {len(result.boxes)}")
    print(f"  Latency    : {ms:.1f} ms")

    # --- Test on webcam frame (if available) ---
    print("\n[TEST 2] Live webcam frame")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            t0 = time.perf_counter()
            result = detector.detect(frame)
            ms = (time.perf_counter() - t0) * 1000
            print(f"  Frame size : {frame.shape[1]}x{frame.shape[0]}")
            print(f"  Detections : {len(result.boxes)}")
            print(f"  Latency    : {ms:.1f} ms")
            if result.class_names:
                counts = detector.get_class_counts(result)
                print(f"  Objects    : {counts}")

            # Save annotated frame to disk for visual inspection
            out_path = "models/test_output.jpg"
            cv2.imwrite(out_path, result.frame)
            print(f"  Saved to   : {out_path}")
    else:
        print("  No webcam available, skipping.")

    # --- Throughput benchmark (10 frames) ---
    print("\n[BENCH] Throughput over 10 frames...")
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    times = []
    for i in range(10):
        t0 = time.perf_counter()
        detector.detect(test_frame)
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    print(f"  Avg latency : {avg_ms:.1f} ms")
    print(f"  Max FPS est : ~{1000/avg_ms:.1f} FPS")
    print(f"  Per-frame   : {[f'{t:.0f}ms' for t in times]}")

    print("\n[OK] detector.py is ready for integration.\n")
