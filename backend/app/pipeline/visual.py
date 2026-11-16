"""One pass over the pixels (the pose model only gives landmarks): scene cuts, ball candidates,
what each detected person looks like, and a few frames for court-line detection.

Scene cuts: a cut is a frame whose colour histogram and brightness pattern both differ
sharply from the previous frame, far more than the frames around it differ from each other.
Runs of near-black frames (fades) also end a shot. Slow cross-dissolves are not detected.

Appearance: a coarse colour histogram of the torso and of the upper legs for every detected
person. It is only used to decide whether a person seen after a cut or after leaving the
frame could be the selected player; it is not identification and it cannot tell apart two
people dressed alike.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import ball
from .pose import FramePoses
from .signals import L_HIP, L_KNEE, L_SH, R_HIP, R_KNEE, R_SH
from .video import iter_frames

CUT_HIST_DIST = 0.35
CUT_MIN_DIFF = 14.0
CUT_RATIO = 3.5
DARK_LUMA = 14.0
APPEARANCE_DIM = 2 * (12 * 3 + 3)


@dataclass
class Visual:
    cuts: list[int] = field(default_factory=list)  # analysed-frame indices at which a new shot starts
    dark: np.ndarray | None = None  # (T,) frame is near-black
    ball: list[list[ball.Candidate]] = field(default_factory=list)
    appearance: np.ndarray | None = None  # (T, P, D) colour descriptor per detected person, NaN where unavailable
    samples: dict[int, np.ndarray] = field(default_factory=dict)  # analysed index -> BGR frame, about one per second


def _region_hist(hsv: np.ndarray, poly: np.ndarray) -> np.ndarray | None:
    x0, y0 = np.floor(poly.min(0)).astype(int)
    x1, y1 = np.ceil(poly.max(0)).astype(int)
    H, W = hsv.shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull((poly - [x0, y0]).astype(np.int32)), 1)
    px = hsv[y0:y1, x0:x1][mask > 0]
    if len(px) < 12:
        return None
    h, s, v = px[:, 0].astype(int), px[:, 1].astype(int), px[:, 2].astype(int)
    chroma = (s >= 50) & (v >= 50)
    out = np.zeros(12 * 3 + 3)
    np.add.at(out, (h[chroma] * 12 // 180) * 3 + np.minimum((s[chroma] - 50) // 70, 2), 1)
    np.add.at(out, 36 + np.digitize(v[~chroma], [70, 170]), 1)
    return out / out.sum()


def describe(hsv: np.ndarray, lm: np.ndarray) -> np.ndarray:
    """Torso + upper-leg colour descriptor for one person (lm: (33, 4) normalised). NaN-filled if the regions are not visible."""
    H, W = hsv.shape[:2]
    out = np.full(APPEARANCE_DIM, np.nan)
    pts = lm[:, :2] * [W, H]
    for k, idx in enumerate(([L_SH, R_SH, R_HIP, L_HIP], [L_HIP, R_HIP, R_KNEE, L_KNEE])):
        if lm[idx, 3].min() < 0.5:
            continue
        poly = pts[idx]
        poly = poly.mean(0) + (poly - poly.mean(0)) * 0.85  # stay inside the body
        h = _region_hist(hsv, poly)
        if h is not None:
            out[k * 39 : (k + 1) * 39] = h
    return out


def similarity(a: np.ndarray, b: np.ndarray) -> float | None:
    """Histogram intersection (0-1) averaged over the body regions both descriptors have. None when nothing can be compared."""
    vals = []
    for k in range(2):
        x, y = a[k * 39 : (k + 1) * 39], b[k * 39 : (k + 1) * 39]
        if np.isfinite(x).all() and np.isfinite(y).all():
            vals.append(float(np.minimum(x, y).sum()))
    return float(np.mean(vals)) if vals else None


def find_cuts(diffs: np.ndarray, hist_dists: np.ndarray, dark: np.ndarray) -> list[int]:
    T = len(diffs)
    cuts = []
    for i in range(1, T):
        if dark[i] != dark[i - 1]:
            if not dark[i]:
                cuts.append(i)  # first visible frame after a fade or a black gap starts a new shot
            continue
        if dark[i]:
            continue
        lo, hi = max(1, i - 12), min(T, i + 13)
        around = np.concatenate([diffs[lo:i], diffs[i + 1 : hi]])
        base = float(np.median(around)) if around.size else 0.0
        if hist_dists[i] >= CUT_HIST_DIST and diffs[i] >= CUT_MIN_DIFF and diffs[i] >= CUT_RATIO * max(base, 1.0):
            cuts.append(i)
    merged: list[int] = []
    for c in cuts:
        if not merged or c - merged[-1] > 2:
            merged.append(c)
    return merged


def visual_pass(video: Path, frame_idx: list[int], detections: list[FramePoses], *, want_ball: bool = True, want_appearance: bool = True, fps: float = 30.0) -> Visual:
    T = len(frame_idx)
    pos = {f: i for i, f in enumerate(frame_idx)}
    diffs, hd, dark = np.zeros(T), np.zeros(T), np.zeros(T, bool)
    P = max((d.lm.shape[0] for d in detections), default=0)
    app = np.full((T, max(P, 1), APPEARANCE_DIM), np.nan, np.float32)
    balls: list[list[ball.Candidate]] = [[] for _ in range(T)]
    samples: dict[int, np.ndarray] = {}
    stride = frame_idx[1] - frame_idx[0] if T > 1 else 1
    every = max(int(round(fps / stride)), 1)
    prev_small = prev_hist = None
    window: list[tuple[int, np.ndarray]] = []
    for f, frame in iter_frames(video, 1):
        i = pos.get(f)
        if i is None:
            continue
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 48)).astype(np.float32)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([cv2.resize(hsv, (96, 72))], [0, 1], None, [12, 4], [0, 180, 0, 256]).ravel()
        hist /= max(hist.sum(), 1.0)
        dark[i] = small.mean() < DARK_LUMA
        if prev_small is not None:
            diffs[i] = float(np.abs(small - prev_small).mean())
            hd[i] = float(np.sqrt(max(0.0, 1.0 - np.sqrt(hist * prev_hist).sum())))
        prev_small, prev_hist = small, hist
        if want_appearance:
            for p in range(detections[i].lm.shape[0]):
                app[i, p] = describe(hsv, detections[i].lm[p])
        if i % every == 0:
            samples[i] = frame
        if want_ball:
            window.append((i, frame))
            if len(window) == 3:
                balls[window[1][0]] = ball.find_candidates(window[0][1], window[1][1], window[2][1])
                window.pop(0)
    return Visual(cuts=find_cuts(diffs, hd, dark), dark=dark, ball=balls, appearance=app, samples=samples)
