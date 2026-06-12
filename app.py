"""
app.py - Streamlit Real-Time Object Detection UI
=================================================
Phase 4 of the CPU-Optimized Real-Time Object Detection System.

Launch:
    streamlit run app.py

Features:
  - Live webcam feed with ONNX-annotated bounding boxes
  - Real-time FPS, latency (ms), and CPU usage metrics
  - Sidebar controls: model resolution, confidence, IoU, camera selector
  - Live object tracker dashboard (counts per class)
  - Session stats: total frames, drop rate
  - Dark-mode premium design with custom CSS
"""

import glob
import os
import time
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import psutil
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Page config — MUST be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CPU Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Premium dark UI
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0d0f14 0%, #111520 50%, #0a0d12 100%);
        color: #e2e8f0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #13161f 0%, #0d1018 100%);
        border-right: 1px solid rgba(99,102,241,0.2);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #a5b4fc;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(139,92,246,0.08) 100%);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99,102,241,0.5);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #a5b4fc;
        line-height: 1.1;
        letter-spacing: -0.5px;
    }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 500;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #475569;
        margin-top: 2px;
    }

    /* ── Status Badge ── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .status-live {
        background: rgba(34,197,94,0.15);
        border: 1px solid rgba(34,197,94,0.35);
        color: #4ade80;
    }
    .status-stopped {
        background: rgba(239,68,68,0.15);
        border: 1px solid rgba(239,68,68,0.35);
        color: #f87171;
    }
    .pulse {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #4ade80;
        animation: pulse 1.5s infinite;
        display: inline-block;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%       { opacity: 0.4; transform: scale(0.8); }
    }

    /* ── Object Tracker Table ── */
    .tracker-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 6px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.05);
        transition: background 0.2s;
    }
    .tracker-row:hover { background: rgba(99,102,241,0.08); }
    .tracker-class {
        font-size: 0.88rem;
        font-weight: 500;
        color: #cbd5e1;
        text-transform: capitalize;
    }
    .tracker-count {
        font-size: 0.88rem;
        font-weight: 700;
        color: #a5b4fc;
        background: rgba(99,102,241,0.2);
        padding: 2px 10px;
        border-radius: 12px;
    }
    .tracker-bar-wrap {
        flex: 1;
        height: 4px;
        background: rgba(255,255,255,0.06);
        border-radius: 4px;
        margin: 0 10px;
        overflow: hidden;
    }
    .tracker-bar {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        transition: width 0.3s ease;
    }

    /* ── Section Headers ── */
    .section-header {
        font-size: 0.72rem;
        font-weight: 600;
        color: #4f46e5;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 16px 0 10px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(99,102,241,0.2);
    }

    /* ── Video Feed ── */
    .video-container {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(99,102,241,0.2);
        box-shadow: 0 0 40px rgba(99,102,241,0.08);
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    /* ── Slider accent ── */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #6366f1 !important;
    }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(99,102,241,0.3) !important;
        color: #e2e8f0 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0.55rem 1rem;
        transition: opacity 0.2s, transform 0.15s;
        letter-spacing: 0.3px;
    }
    .stButton > button:hover {
        opacity: 0.88;
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_models() -> list[str]:
    """Scan the models/ directory for .onnx files."""
    return sorted(glob.glob("models/*.onnx"))


def model_label(path: str) -> str:
    """Human-friendly model name for the selectbox."""
    stem = Path(path).stem                       # e.g. yolov8n_imgsz640
    parts = stem.replace("_imgsz", " @ ").replace("yolov8n", "YOLOv8n")
    return parts


def frame_to_pil(bgr_frame: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR frame to PIL RGB for Streamlit display."""
    return Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))


def _metric_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return f"""<div class="metric-card">
<div class="metric-value">{value}</div>
<div class="metric-label">{label}</div>
{sub_html}
</div>"""


def _tracker_row(name: str, count: int, max_count: int) -> str:
    pct = int(100 * count / max_count) if max_count else 0
    return f"""<div class="tracker-row">
<span class="tracker-class">{name}</span>
<div class="tracker-bar-wrap">
<div class="tracker-bar" style="width:{pct}%"></div>
</div>
<span class="tracker-count">{count}</span>
</div>"""


# ---------------------------------------------------------------------------
# Session State Init
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "running":       False,
        "streamer":      None,
        "detector":      None,
        "model_path":    None,
        "infer_size":    320,
        "conf":          0.45,
        "iou":           0.45,
        "camera_idx":    0,
        "show_raw":      False,
        "frame_count":   0,
        "start_time":    None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ---------------------------------------------------------------------------
# Start / Stop logic
# ---------------------------------------------------------------------------

def start_detection():
    """Initialize detector + streamer and start threads."""
    from detector import CPUObjectDetector
    from streamer import VideoStreamer

    model_path  = st.session_state["model_path"]
    infer_size  = st.session_state["infer_size"]
    conf        = st.session_state["conf"]
    iou         = st.session_state["iou"]
    camera_idx  = st.session_state["camera_idx"]

    try:
        detector = CPUObjectDetector(
            model_path=model_path,
            conf_threshold=conf,
            iou_threshold=iou,
            infer_size=infer_size,
        )
        streamer = VideoStreamer(
            detector=detector,
            camera_index=camera_idx,
            capture_width=640,
            capture_height=480,
            queue_maxsize=1,
        )
        streamer.start()

        st.session_state["detector"]   = detector
        st.session_state["streamer"]   = streamer
        st.session_state["running"]    = True
        st.session_state["start_time"] = time.time()
        st.session_state["frame_count"] = 0

    except Exception as e:
        st.error(f"Failed to start: {e}")


def stop_detection():
    """Stop threads and clean up."""
    streamer: Optional["VideoStreamer"] = st.session_state.get("streamer")
    if streamer:
        streamer.stop()
    st.session_state["running"] = False
    st.session_state["streamer"] = None
    st.session_state["detector"] = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 10px 0 18px 0;">
            <div style="font-size:2rem;">🎯</div>
            <div style="font-size:1.1rem; font-weight:700; color:#a5b4fc; letter-spacing:-0.3px;">
                CPU Object Detection
            </div>
            <div style="font-size:0.72rem; color:#475569; margin-top:4px;">
                YOLOv8n · ONNX Runtime · Streamlit
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Model Selection ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Model</div>', unsafe_allow_html=True)

    models = find_models()
    if not models:
        st.error("No ONNX models found in models/\nRun: `python export.py`")
        st.stop()

    model_labels = [model_label(m) for m in models]
    sel_idx = st.selectbox(
        "Model",
        range(len(models)),
        format_func=lambda i: model_labels[i],
        key="model_sel_idx",
        label_visibility="collapsed",
    )
    st.session_state["model_path"] = models[sel_idx]

    # Show model file size
    size_mb = os.path.getsize(models[sel_idx]) / 1e6
    st.caption(f"`{Path(models[sel_idx]).name}` — {size_mb:.1f} MB")

    # ── Inference Resolution ─────────────────────────────────────────
    st.markdown('<div class="section-header">Inference Resolution</div>', unsafe_allow_html=True)

    res_choice = st.radio(
        "Resolution",
        [320, 416, 640],
        index=[320, 416, 640].index(st.session_state["infer_size"]),
        format_func=lambda x: f"{x}×{x} px {'⚡ Fast' if x==320 else ('⚖️ Balanced' if x==416 else '🔬 Accurate')}",
        horizontal=False,
        label_visibility="collapsed",
    )
    st.session_state["infer_size"] = res_choice

    # ── Detection Thresholds ─────────────────────────────────────────
    st.markdown('<div class="section-header">Detection Thresholds</div>', unsafe_allow_html=True)

    conf_val = st.slider(
        "Confidence",
        min_value=0.10,
        max_value=0.95,
        value=st.session_state["conf"],
        step=0.05,
        format="%.2f",
        help="Minimum confidence score to display a detection",
    )
    st.session_state["conf"] = conf_val

    iou_val = st.slider(
        "NMS IoU Threshold",
        min_value=0.10,
        max_value=0.90,
        value=st.session_state["iou"],
        step=0.05,
        format="%.2f",
        help="Higher = more overlapping boxes kept; Lower = stricter suppression",
    )
    st.session_state["iou"] = iou_val

    # Live-update thresholds if running
    if st.session_state["running"] and st.session_state["detector"]:
        st.session_state["detector"].update_thresholds(conf_val, iou_val)
        st.session_state["detector"].update_infer_size(res_choice)

    # ── Camera ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Camera</div>', unsafe_allow_html=True)

    cam_idx = st.number_input(
        "Camera Index",
        min_value=0,
        max_value=10,
        value=st.session_state["camera_idx"],
        step=1,
        help="0 = default webcam. Try 1, 2... for additional cameras.",
    )
    st.session_state["camera_idx"] = int(cam_idx)

    # ── Display Options ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Display</div>', unsafe_allow_html=True)
    show_raw = st.toggle("Show raw (unannotated) feed", value=False)
    st.session_state["show_raw"] = show_raw

    # ── Start / Stop ─────────────────────────────────────────────────
    st.markdown("---")
    if not st.session_state["running"]:
        if st.button("▶  Start Detection", key="btn_start"):
            start_detection()
            st.rerun()
    else:
        if st.button("⏹  Stop Detection", key="btn_stop"):
            stop_detection()
            st.rerun()

    # ── System Info ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">System</div>', unsafe_allow_html=True)
    cpu_pct  = psutil.cpu_percent(interval=None)
    mem      = psutil.virtual_memory()
    st.caption(
        f"CPU: **{cpu_pct:.0f}%** &nbsp;|&nbsp; "
        f"RAM: **{mem.percent:.0f}%** ({mem.used/1e9:.1f}/{mem.total/1e9:.1f} GB)"
    )
    st.caption(f"Cores: {os.cpu_count()} &nbsp;|&nbsp; Python {os.sys.version.split()[0]}")


# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------

# ── Page Title ──────────────────────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown(
        "<h1 style='margin:0; font-size:1.8rem; font-weight:700; color:#e2e8f0;"
        "letter-spacing:-0.5px;'>Real-Time Object Detection</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='margin:2px 0 16px 0; color:#475569; font-size:0.85rem;'>"
        "CPU-optimized · YOLOv8n ONNX · No GPU required</p>",
        unsafe_allow_html=True,
    )

with col_status:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.session_state["running"]:
        st.markdown(
            '<div class="status-badge status-live">'
            '<span class="pulse"></span> LIVE</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-badge status-stopped">⬛ STOPPED</div>',
            unsafe_allow_html=True,
        )

# ── Layout: Video (left) + Tracker (right) ──────────────────────────────────
vid_col, tracker_col = st.columns([3, 1], gap="medium")

# ── Metrics Row ─────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4, gap="small")

# Placeholders — updated each loop tick
video_ph   = vid_col.empty()
tracker_ph = tracker_col.empty()
m1_ph      = m1.empty()
m2_ph      = m2.empty()
m3_ph      = m3.empty()
m4_ph      = m4.empty()

# Bottom section: session stats
st.markdown("---")
stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
sc1 = stat_col1.empty()
sc2 = stat_col2.empty()
sc3 = stat_col3.empty()
sc4 = stat_col4.empty()

info_ph = st.empty()


# ---------------------------------------------------------------------------
# Initial / Idle State
# ---------------------------------------------------------------------------

def render_idle():
    """Show placeholder content when detection is not running."""
    video_ph.markdown(
        """
        <div style="
            background: rgba(255,255,255,0.02);
            border: 2px dashed rgba(99,102,241,0.25);
            border-radius: 14px;
            height: 400px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #334155;
        ">
            <div style="font-size:3rem; margin-bottom:12px;">📷</div>
            <div style="font-size:1rem; font-weight:600; color:#475569;">
                Camera feed will appear here
            </div>
            <div style="font-size:0.8rem; color:#334155; margin-top:6px;">
                Click <strong style="color:#6366f1">▶ Start Detection</strong> in the sidebar
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tracker_ph.markdown(
        """
        <div style="
            background: rgba(255,255,255,0.02);
            border: 1px dashed rgba(99,102,241,0.2);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            color: #334155;
            min-height: 200px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        ">
            <div style="font-size:2rem; margin-bottom:8px;">📊</div>
            <div style="font-size:0.85rem; color:#475569;">Object tracker</div>
            <div style="font-size:0.75rem; color:#334155; margin-top:4px;">
                Starts when detection is live
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for ph in [m1_ph, m2_ph, m3_ph, m4_ph]:
        ph.markdown(_metric_card("—", "—", ""), unsafe_allow_html=True)

    m1_ph.markdown(_metric_card("Capture FPS",   "—", "webcam"), unsafe_allow_html=True)
    m2_ph.markdown(_metric_card("Inference FPS", "—", "ONNX"),   unsafe_allow_html=True)
    m3_ph.markdown(_metric_card("Latency",       "— ms", "per frame"), unsafe_allow_html=True)
    m4_ph.markdown(_metric_card("CPU Usage",     f"{psutil.cpu_percent():.0f}%", "all cores"), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Live Update Loop
# ---------------------------------------------------------------------------

if not st.session_state["running"]:
    render_idle()
    st.stop()

# If we get here, detection IS running
streamer = st.session_state["streamer"]
detector = st.session_state["detector"]

if streamer is None:
    render_idle()
    st.stop()

# Frame-rate cap for Streamlit rendering (~15 UI FPS is smooth enough)
UI_FPS_CAP   = 15
LOOP_DELAY   = 1.0 / UI_FPS_CAP
MAX_NO_FRAME = 5   # seconds before giving up

last_frame_time = time.time()
no_frame_since  = time.time()

while st.session_state.get("running", False):

    result = streamer.get_latest_result()
    stats  = streamer.get_stats()
    cpu    = psutil.cpu_percent(interval=None)

    # ── Video Frame ──────────────────────────────────────────────────
    if result is not None:
        no_frame_since = time.time()
        frame = result.raw_frame if show_raw else result.frame
        pil   = frame_to_pil(frame)
        video_ph.image(pil, use_container_width=True, channels="RGB")

        # ── Object Tracker ────────────────────────────────────────────
        counts     = detector.get_class_counts(result)
        total_objs = sum(counts.values())
        max_count  = max(counts.values()) if counts else 1

        tracker_html = (
            "<div style='padding:4px 0;'>"
            f"<div class='section-header'>Detected Objects ({total_objs} total)</div>"
        )
        if counts:
            for cls_name, cnt in list(counts.items())[:10]:  # Show top 10
                tracker_html += _tracker_row(cls_name, cnt, max_count)
        else:
            tracker_html += (
                "<div style='text-align:center; color:#334155; "
                "padding:30px 0; font-size:0.85rem;'>No objects detected</div>"
            )
        tracker_html += "</div>"
        tracker_ph.markdown(tracker_html, unsafe_allow_html=True)

        st.session_state["frame_count"] += 1

    else:
        # No frame yet — show waiting indicator
        elapsed_wait = time.time() - no_frame_since
        if elapsed_wait > MAX_NO_FRAME:
            video_ph.error(
                "No frames received for 5 seconds. "
                "Check your webcam connection or camera index."
            )
        else:
            video_ph.markdown(
                f"""
                <div style="
                    height:400px; display:flex; align-items:center;
                    justify-content:center; border-radius:14px;
                    border: 1px solid rgba(99,102,241,0.15);
                    background: rgba(255,255,255,0.01);
                    flex-direction:column; gap:12px;
                ">
                    <div style="
                        width:40px; height:40px; border-radius:50%;
                        border:3px solid rgba(99,102,241,0.2);
                        border-top-color:#6366f1;
                        animation: spin 0.8s linear infinite;
                    "></div>
                    <div style="color:#475569; font-size:0.88rem;">
                        Initializing camera & model...
                    </div>
                </div>
                <style>
                @keyframes spin {{
                    to {{ transform: rotate(360deg); }}
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )

    # ── Metric Cards ──────────────────────────────────────────────────
    cap_fps   = stats["capture_fps"]
    inf_fps   = stats["inference_fps"]
    inf_ms    = stats["inference_ms"]

    # Color-code inference FPS
    if inf_fps >= 10:
        fps_color = "#4ade80"     # Green
    elif inf_fps >= 5:
        fps_color = "#facc15"     # Yellow
    else:
        fps_color = "#f87171"     # Red

    m1_ph.markdown(
        _metric_card("Capture FPS", f"{cap_fps:.1f}", "webcam input"),
        unsafe_allow_html=True,
    )
    m2_ph.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{fps_color}">{inf_fps:.1f}</div>
            <div class="metric-label">Inference FPS</div>
            <div class="metric-sub">ONNX output</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    m3_ph.markdown(
        _metric_card("Latency", f"{inf_ms:.0f} ms", "per frame"),
        unsafe_allow_html=True,
    )
    m4_ph.markdown(
        _metric_card("CPU Usage", f"{cpu:.0f}%", f"{os.cpu_count()} cores"),
        unsafe_allow_html=True,
    )

    # ── Session Stats ─────────────────────────────────────────────────
    frames_proc   = stats["frames_processed"]
    frames_cap    = stats["frames_captured"]
    drop_pct      = stats["drop_rate_pct"]
    elapsed       = time.time() - (st.session_state["start_time"] or time.time())
    elapsed_str   = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"

    sc1.metric("Session Time",      elapsed_str)
    sc2.metric("Frames Captured",   f"{frames_cap:,}")
    sc3.metric("Frames Processed",  f"{frames_proc:,}")
    sc4.metric("Drop Rate",         f"{drop_pct:.1f}%",
               delta=None if drop_pct < 90 else "High — try 320px",
               delta_color="inverse")

    # ── Loop timing ───────────────────────────────────────────────────
    elapsed_tick = time.time() - last_frame_time
    sleep_time   = max(0.0, LOOP_DELAY - elapsed_tick)
    time.sleep(sleep_time)
    last_frame_time = time.time()
