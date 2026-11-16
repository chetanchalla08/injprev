"""Video validation, normalisation and recording-quality checks.

Every upload is re-encoded to one predictable format (H.264 MP4, constant frame rate,
rotation applied, no audio, longest side <= 1280 px). The analysis and the browser both
use that file, so a frame index always maps to the same timestamp in both places.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .errors import PipelineError

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def probe(path: Path) -> dict:
    """Read container metadata with ffmpeg. Raises PipelineError for non-video files."""
    try:
        out = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True, timeout=60, errors="replace"
        ).stderr
    except subprocess.TimeoutExpired as e:
        raise PipelineError("unreadable_file", "The file could not be read in time.") from e
    stream = re.search(r"Stream #.*Video: (\w+).*?(\d{2,5})x(\d{2,5})", out)
    if not stream:
        raise PipelineError(
            "unsupported_file",
            "This file does not contain a video we can read.",
            "Upload an MP4 or MOV recording straight from your phone or camera.",
        )
    dur = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", out)
    duration = int(dur[1]) * 3600 + int(dur[2]) * 60 + float(dur[3]) if dur else None
    fps_m = re.search(r"([\d.]+) fps", out)
    rot = re.search(r"rotation of (-?[\d.]+) degrees", out)
    return {
        "codec": stream[1],
        "width": int(stream[2]),
        "height": int(stream[3]),
        "duration_s": duration,
        "fps": float(fps_m[1]) if fps_m else None,
        "rotation_deg": float(rot[1]) if rot else 0.0,
    }


def normalize(src: Path, dst: Path, *, max_duration_s: float, min_duration_s: float) -> dict:
    """Validate ``src`` and write the normalised analysis/playback copy to ``dst``."""
    info = probe(src)
    if info["duration_s"] is not None:
        if info["duration_s"] > max_duration_s + 0.5:
            raise PipelineError(
                "too_long",
                f"The video is {info['duration_s']:.0f} s long; the limit is {max_duration_s:.0f} s.",
                "Trim it to the rally or set of serves you want reviewed and upload again.",
            )
        if info["duration_s"] < min_duration_s:
            raise PipelineError("too_short", "The video is too short to contain a stroke.", "Record at least a few seconds.")
    src_fps = info["fps"] or 30.0
    fps = int(round(min(max(src_fps, 15.0), 60.0)))
    scale = "scale='if(gte(iw,ih),min(1280,iw),-2)':'if(gte(iw,ih),-2,min(1280,ih))'"
    cmd = [
        ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
        "-an", "-sn", "-map", "0:v:0", "-vf", f"{scale},fps={fps}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
        "-g", str(max(fps // 2, 1)), "-movflags", "+faststart", str(dst),
    ]  # fmt: skip
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=900, errors="replace")
    except subprocess.TimeoutExpired as e:
        raise PipelineError("decode_failed", "Converting the video took too long.") from e
    if res.returncode != 0 or not dst.exists() or dst.stat().st_size < 1000:
        dst.unlink(missing_ok=True)
        raise PipelineError(
            "decode_failed",
            "The video appears to be damaged or uses a format we could not decode.",
            "Try re-exporting it from your phone's gallery, or record a new clip.",
        )
    cap = cv2.VideoCapture(str(dst))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, _ = cap.read()
    cap.release()
    if not ok or n < 2:
        dst.unlink(missing_ok=True)
        raise PipelineError("decode_failed", "No frames could be decoded from the video.")
    duration = n / fps
    if duration > max_duration_s + 0.5:
        dst.unlink(missing_ok=True)
        raise PipelineError("too_long", f"The video is {duration:.0f} s long; the limit is {max_duration_s:.0f} s.")
    if duration < min_duration_s:
        dst.unlink(missing_ok=True)
        raise PipelineError("too_short", "The video is too short to contain a stroke.", "Record at least a few seconds.")
    return {
        "width": w,
        "height": h,
        "fps": float(fps),
        "n_frames": n,
        "duration_s": round(duration, 3),
        "source": {k: info[k] for k in ("codec", "width", "height", "fps", "rotation_deg")},
    }


def iter_frames(path: Path, stride: int = 1):
    """Yield (frame_index, BGR frame) for every ``stride``-th frame."""
    cap = cv2.VideoCapture(str(path))
    try:
        i = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i % stride == 0:
                ok, frame = cap.retrieve()
                if ok:
                    yield i, frame
            i += 1
    finally:
        cap.release()


def read_frame(path: Path, index: int):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


# --- recording quality -----------------------------------------------------------------

# Heuristic thresholds. They decide which plain-language warnings are shown; they are not
# pass/fail gates on the analysis (pose visibility does that job, per measurement).
DARK_LUMA = 60.0
SOFT_LAPLACIAN_VAR = 35.0
LOW_FPS = 50.0
CAMERA_SHIFT_FRAC = 0.012


def quality_checks(path: Path, meta: dict) -> dict:
    n, fps = meta["n_frames"], meta["fps"]
    sample_idx = np.unique(np.linspace(0, n - 1, 12).astype(int))
    lumas, sharps, shifts = [], [], []
    prev = None
    cap = cv2.VideoCapture(str(path))
    half = max(int(fps // 2), 1)
    for idx in sample_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (640, max(int(640 * gray.shape[0] / gray.shape[1]), 2)))
        lumas.append(float(small.mean()))
        sharps.append(float(cv2.Laplacian(small, cv2.CV_64F).var()))
        # camera motion: global shift between this frame and one half a second later
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(min(idx + half, n - 1)))
        ok2, frame2 = cap.read()
        if ok2:
            g2 = cv2.resize(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), (small.shape[1], small.shape[0]))
            (dx, dy), _ = cv2.phaseCorrelate(np.float32(small), np.float32(g2))
            shifts.append(float(np.hypot(dx, dy) / small.shape[1]))
        prev = small
    cap.release()
    del prev
    metrics = {
        "mean_brightness": round(float(np.median(lumas)), 1) if lumas else None,
        "sharpness": round(float(np.median(sharps)), 1) if sharps else None,
        "camera_shift": round(float(np.median(shifts)), 4) if shifts else None,
        "fps": fps,
        "resolution": f"{meta['width']}x{meta['height']}",
    }
    issues: list[dict] = []
    if metrics["mean_brightness"] is not None and metrics["mean_brightness"] < DARK_LUMA:
        issues.append({
            "code": "low_light",
            "message": "The recording is dark. Body landmarks are harder to locate in low light.",
            "suggestion": "Record in daylight or under full court lighting, with the light behind the camera.",
        })
    if metrics["sharpness"] is not None and metrics["sharpness"] < SOFT_LAPLACIAN_VAR:
        issues.append({
            "code": "soft_image",
            "message": "The picture is soft or blurred, which reduces landmark accuracy, especially for the hitting arm.",
            "suggestion": "Clean the lens, tap to focus on the player, and avoid digital zoom.",
        })
    if fps < LOW_FPS:
        issues.append({
            "code": "low_frame_rate",
            "message": f"Recorded at {fps:.0f} frames per second. A swing lasts only a few frames at this rate, so timing "
            "around contact is approximate and the arm may be motion-blurred.",
            "suggestion": "Use 60 fps or your phone's slow-motion mode if available.",
        })
    if min(meta["width"], meta["height"]) < 480:
        issues.append({
            "code": "low_resolution",
            "message": "Low resolution limits how precisely joints can be located.",
            "suggestion": "Record at 1080p.",
        })
    if metrics["camera_shift"] is not None and metrics["camera_shift"] > CAMERA_SHIFT_FRAC:
        issues.append({
            "code": "camera_motion",
            "message": "The camera appears to move. Speeds and positions are measured in the image, so camera motion distorts them.",
            "suggestion": "Use a tripod or prop the phone against the fence.",
        })
    return {"metrics": metrics, "issues": issues}
