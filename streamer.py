"""
streamer.py - Multi-Threaded Webcam Frame Handler
==================================================
Phase 2 of the CPU-Optimized Real-Time Object Detection System.

Architecture:
    +------------------+       Queue        +---------------------+
    |  CaptureThread   |  --> [f0,f1,f2...] --> | Consumer (detector) |
    |  (runs at 30fps) |   max_size=1 (drop   |  (runs at ~5-15fps) |
    |  OpenCV grab()   |   stale frames)      |  ONNX inference     |
    +------------------+                      +---------------------+

Key Design Decisions:
  - Queue maxsize=1: Only the LATEST frame is kept. When inference is busy,
    older frames are silently dropped so the display is never stale.
  - Daemon threads: Automatically killed when the main process exits.
  - Frame grab() + retrieve() split: grab() is non-blocking and very fast
    (~0.1ms). It advances the internal camera buffer without decoding, so
    we don't decode every frame — only the one we actually need.
  - Separate lock for the annotated (result) frame: The detector writes
    annotated frames; the UI reads them. A lock prevents tearing.
"""

import os
import time
import threading
import queue
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("streamer")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class FrameStats:
    """
    Thread-safe performance counters updated by the streamer.
    Consumers (UI layer) read these without acquiring any lock —
    individual attribute reads are atomic in CPython (GIL-protected).
    """
    capture_fps: float = 0.0      # Raw webcam acquisition rate
    inference_fps: float = 0.0    # ONNX inference throughput
    inference_ms: float = 0.0     # Last inference latency in milliseconds
    frames_captured: int = 0      # Total frames grabbed from camera
    frames_dropped: int = 0       # Frames discarded due to slow inference
    frames_processed: int = 0     # Total frames that completed inference
    is_running: bool = False       # Whether the streamer is active


@dataclass
class DetectionResult:
    """Holds one frame's inference output."""
    frame: np.ndarray               # Annotated BGR frame (ready to display)
    raw_frame: np.ndarray           # Original un-annotated frame
    boxes: list = field(default_factory=list)     # [[x1,y1,x2,y2], ...]
    scores: list = field(default_factory=list)    # [0.91, 0.78, ...]
    class_ids: list = field(default_factory=list) # [0, 2, 56, ...]
    class_names: list = field(default_factory=list) # ['person', 'car', ...]
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Capture Thread
# ---------------------------------------------------------------------------

class CaptureThread(threading.Thread):
    """
    Continuously pulls frames from a webcam into a bounded queue.

    The thread runs as a daemon and uses grab()/retrieve() instead of
    read() to avoid buffer bloat. grab() advances the camera buffer
    without full decoding, keeping the internal camera queue fresh.
    Only the frame we actually send to the queue is fully decoded.
    """

    def __init__(
        self,
        frame_queue: "queue.Queue[np.ndarray]",
        camera_index: int = 0,
        target_width: int = 640,
        target_height: int = 480,
    ):
        super().__init__(daemon=True, name="CaptureThread")
        self.frame_queue = frame_queue
        self.camera_index = camera_index
        self.target_width = target_width
        self.target_height = target_height

        self._stop_event = threading.Event()
        self._cap: Optional[cv2.VideoCapture] = None
        self.stats = FrameStats()

        # FPS tracking (exponential moving average)
        self._fps_alpha = 0.1   # Smoothing factor: lower = smoother
        self._last_ts = time.perf_counter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stop(self):
        """Signal the thread to stop and wait for it to exit."""
        log.info("CaptureThread stop requested.")
        self._stop_event.set()
        self.join(timeout=3.0)
        if self._cap and self._cap.isOpened():
            self._cap.release()
            log.info("Camera released.")

    @property
    def is_alive_and_running(self) -> bool:
        return self.is_alive() and not self._stop_event.is_set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open_camera(self) -> bool:
        """Try to open the camera; return True on success."""
        log.info(f"Opening camera index={self.camera_index} ...")
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)  # CAP_DSHOW = faster on Windows

        if not cap.isOpened():
            # Fallback: try without backend hint
            cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            log.error(f"Cannot open camera at index {self.camera_index}.")
            return False

        # Request resolution from the camera (it may not honour exactly)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize internal buffer lag

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(f"Camera opened: {actual_w}x{actual_h} @ "
                 f"{cap.get(cv2.CAP_PROP_FPS):.0f}fps")

        self._cap = cap
        return True

    def _update_fps(self):
        """Update capture FPS using exponential moving average."""
        now = time.perf_counter()
        dt = now - self._last_ts
        if dt > 0:
            instant_fps = 1.0 / dt
            self.stats.capture_fps = (
                self._fps_alpha * instant_fps
                + (1 - self._fps_alpha) * self.stats.capture_fps
            )
        self._last_ts = now

    def run(self):
        """Main capture loop."""
        self.stats.is_running = True

        if not self._open_camera():
            self.stats.is_running = False
            return

        log.info("CaptureThread started.")
        consecutive_failures = 0
        MAX_FAILURES = 10

        while not self._stop_event.is_set():
            # grab() is fast — it advances camera buffer but doesn't decode
            grabbed = self._cap.grab()
            if not grabbed:
                consecutive_failures += 1
                log.warning(f"Frame grab failed ({consecutive_failures}/{MAX_FAILURES})")
                if consecutive_failures >= MAX_FAILURES:
                    log.error("Too many consecutive grab failures. Stopping.")
                    break
                time.sleep(0.05)
                continue

            consecutive_failures = 0

            # retrieve() does the actual JPEG decode — only when queue has space
            try:
                # Non-blocking put: if queue is full, discard old frame first
                ret, frame = self._cap.retrieve()
                if not ret or frame is None:
                    continue

                self.stats.frames_captured += 1
                self._update_fps()

                try:
                    # put_nowait raises Full if queue is at maxsize
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Drop the oldest frame, insert the latest
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.frame_queue.put_nowait(frame)
                    self.stats.frames_dropped += 1

            except Exception as exc:
                log.error(f"Unexpected error in CaptureThread: {exc}", exc_info=True)

        self.stats.is_running = False
        if self._cap:
            self._cap.release()
        log.info("CaptureThread exited.")


# ---------------------------------------------------------------------------
# Inference Worker Thread
# ---------------------------------------------------------------------------

class InferenceThread(threading.Thread):
    """
    Consumes raw frames from the queue, runs ONNX inference, and stores
    the annotated result so the UI can display it.

    The detector (CPUObjectDetector) is injected — this thread only
    orchestrates the pipeline, keeping concerns separated.
    """

    def __init__(
        self,
        frame_queue: "queue.Queue[np.ndarray]",
        detector,          # CPUObjectDetector instance (injected, see detector.py)
        result_lock: threading.Lock,
    ):
        super().__init__(daemon=True, name="InferenceThread")
        self.frame_queue = frame_queue
        self.detector = detector
        self.result_lock = result_lock

        self._stop_event = threading.Event()
        self._latest_result: Optional[DetectionResult] = None

        # FPS / latency tracking
        self._fps_alpha = 0.15
        self._infer_fps: float = 0.0
        self._infer_ms: float = 0.0
        self._frames_processed: int = 0
        self._last_ts = time.perf_counter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stop(self):
        log.info("InferenceThread stop requested.")
        self._stop_event.set()
        self.join(timeout=5.0)

    def get_latest_result(self) -> Optional[DetectionResult]:
        """Thread-safe read of the most recently processed frame."""
        with self.result_lock:
            return self._latest_result

    @property
    def inference_fps(self) -> float:
        return self._infer_fps

    @property
    def inference_ms(self) -> float:
        return self._infer_ms

    @property
    def frames_processed(self) -> int:
        return self._frames_processed

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_stats(self, latency_ms: float):
        now = time.perf_counter()
        dt = now - self._last_ts
        if dt > 0:
            instant_fps = 1.0 / dt
            self._infer_fps = (
                self._fps_alpha * instant_fps
                + (1 - self._fps_alpha) * self._infer_fps
            )
        self._infer_ms = (
            self._fps_alpha * latency_ms
            + (1 - self._fps_alpha) * self._infer_ms
        )
        self._last_ts = now
        self._frames_processed += 1

    def run(self):
        log.info("InferenceThread started.")

        while not self._stop_event.is_set():
            try:
                # Block up to 0.5s waiting for a frame
                frame = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                t0 = time.perf_counter()
                result = self.detector.detect(frame)
                latency_ms = (time.perf_counter() - t0) * 1000

                self._update_stats(latency_ms)

                # Attach timing info to the result
                result.timestamp = time.time()

                with self.result_lock:
                    self._latest_result = result

            except Exception as exc:
                log.error(f"Inference error: {exc}", exc_info=True)

        log.info("InferenceThread exited.")


# ---------------------------------------------------------------------------
# High-Level VideoStreamer (Facade)
# ---------------------------------------------------------------------------

class VideoStreamer:
    """
    Public facade that manages the full pipeline:
        Camera  →  CaptureThread  →  Queue  →  InferenceThread  →  Result

    Usage:
        streamer = VideoStreamer(detector, camera_index=0, infer_width=320)
        streamer.start()

        while True:
            result = streamer.get_latest_result()
            stats  = streamer.get_stats()
            ...

        streamer.stop()
    """

    def __init__(
        self,
        detector,                      # CPUObjectDetector instance
        camera_index: int = 0,
        capture_width: int = 640,
        capture_height: int = 480,
        queue_maxsize: int = 1,        # Keep only the freshest frame
    ):
        self.detector = detector
        self.camera_index = camera_index
        self.capture_width = capture_width
        self.capture_height = capture_height

        # Shared queue: producer (camera) → consumer (inference)
        self._frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_maxsize)
        self._result_lock = threading.Lock()

        # Threads
        self._capture_thread: Optional[CaptureThread] = None
        self._inference_thread: Optional[InferenceThread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the capture and inference threads."""
        if self._running:
            log.warning("VideoStreamer is already running.")
            return

        log.info("Starting VideoStreamer...")

        self._capture_thread = CaptureThread(
            frame_queue=self._frame_queue,
            camera_index=self.camera_index,
            target_width=self.capture_width,
            target_height=self.capture_height,
        )

        self._inference_thread = InferenceThread(
            frame_queue=self._frame_queue,
            detector=self.detector,
            result_lock=self._result_lock,
        )

        self._capture_thread.start()
        # Small delay so camera has time to open before inference tries to read
        time.sleep(0.5)
        self._inference_thread.start()

        self._running = True
        log.info("VideoStreamer running.")

    def stop(self):
        """Gracefully stop both threads and release resources."""
        if not self._running:
            return

        log.info("Stopping VideoStreamer...")
        if self._inference_thread:
            self._inference_thread.stop()
        if self._capture_thread:
            self._capture_thread.stop()

        # Drain the queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        self._running = False
        log.info("VideoStreamer stopped.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ------------------------------------------------------------------
    # Data Access
    # ------------------------------------------------------------------

    def get_latest_result(self) -> Optional[DetectionResult]:
        """Return the most recently completed DetectionResult, or None."""
        if self._inference_thread:
            return self._inference_thread.get_latest_result()
        return None

    def get_stats(self) -> dict:
        """
        Return a unified stats dictionary for display in the UI.
        All reads are safe: floats/ints are GIL-protected in CPython.
        """
        cap_stats = self._capture_thread.stats if self._capture_thread else FrameStats()
        inf = self._inference_thread

        return {
            "capture_fps":       round(cap_stats.capture_fps, 1),
            "inference_fps":     round(inf.inference_fps if inf else 0.0, 1),
            "inference_ms":      round(inf.inference_ms if inf else 0.0, 1),
            "frames_captured":   cap_stats.frames_captured,
            "frames_dropped":    cap_stats.frames_dropped,
            "frames_processed":  inf.frames_processed if inf else 0,
            "drop_rate_pct":     _safe_pct(cap_stats.frames_dropped,
                                           cap_stats.frames_captured),
            "is_running":        self._running,
        }

    @property
    def is_running(self) -> bool:
        return self._running


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_pct(numerator: int, denominator: int) -> float:
    """Return percentage, handling division by zero."""
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 1)


def list_available_cameras(max_index: int = 5) -> list[int]:
    """
    Probe camera indices 0..max_index and return those that open successfully.
    Useful for the Streamlit sidebar camera selector.
    """
    available = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    return available


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Quick standalone test: runs the CaptureThread alone (no detector)
    and prints FPS + drop stats every second for 10 seconds.
    Press Ctrl+C to exit early.
    """
    import sys

    print("=" * 55)
    print("  streamer.py — Standalone Camera Test")
    print("=" * 55)

    cameras = list_available_cameras()
    if not cameras:
        print("[ERROR] No cameras found. Check your webcam connection.")
        sys.exit(1)

    print(f"Found cameras at indices: {cameras}")
    cam_idx = cameras[0]

    raw_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
    cap_thread = CaptureThread(
        frame_queue=raw_q,
        camera_index=cam_idx,
        target_width=640,
        target_height=480,
    )

    print(f"\nStarting capture on camera {cam_idx} for 10 seconds...")
    cap_thread.start()

    try:
        for i in range(10):
            time.sleep(1)
            s = cap_thread.stats
            print(
                f"  [{i+1:02d}s] "
                f"Capture FPS: {s.capture_fps:5.1f} | "
                f"Captured: {s.frames_captured:5d} | "
                f"Dropped: {s.frames_dropped:4d} | "
                f"Queue size: {raw_q.qsize()}"
            )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap_thread.stop()
        print("\n[OK] Camera test complete.")
