"""Pose track container and geometry helpers shared by segmentation and measurement."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

# MediaPipe Pose landmark indices ("left"/"right" are the player's own left/right).
NOSE = 0
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 23, 24, 25, 26, 27, 28
L_HEEL, R_HEEL, L_TOE, R_TOE = 29, 30, 31, 32
N_LANDMARKS = 33

# Landmarks sent to the browser for the skeleton overlay.
OVERLAY_POINTS = [NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK, L_HEEL, R_HEEL, L_TOE, R_TOE]

MIN_VIS = 0.5  # a landmark below this visibility is not used for any measurement


@dataclass(frozen=True)
class Side:
    sh: int
    el: int
    wr: int
    hip: int
    knee: int
    ank: int
    toe: int


LEFT = Side(L_SH, L_EL, L_WR, L_HIP, L_KNEE, L_ANK, L_TOE)
RIGHT = Side(R_SH, R_EL, R_WR, R_HIP, R_KNEE, R_ANK, R_TOE)


def sides(handedness: str) -> tuple[Side, Side]:
    """Return (dominant, non_dominant) landmark sets."""
    return (LEFT, RIGHT) if handedness == "left" else (RIGHT, LEFT)


def angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle at ``b`` in degrees for points of shape (..., D). NaN-propagating."""
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(v1 * v2, axis=-1) / n
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def fill_short_gaps(a: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate NaN runs of at most ``max_gap`` samples along axis 0."""
    out = a.astype(float).copy()
    flat = out.reshape(out.shape[0], -1)
    idx = np.arange(flat.shape[0])
    for c in range(flat.shape[1]):
        col = flat[:, c]
        ok = ~np.isnan(col)
        if ok.all() or ok.sum() < 2:
            continue
        good = idx[ok]
        filled = np.interp(idx, good, col[ok])
        prev_good = np.maximum.accumulate(np.where(ok, idx, -1))
        next_good = np.minimum.accumulate(np.where(ok, idx, flat.shape[0] * 2)[::-1])[::-1]
        gap = next_good - prev_good - 1
        fillable = (~ok) & (prev_good >= 0) & (next_good < flat.shape[0]) & (gap <= max_gap)
        col[fillable] = filled[fillable]
    return out


def smooth(a: np.ndarray, k: int) -> np.ndarray:
    """NaN-aware binomial smoothing along axis 0 with an odd window ``k`` (3 or 5)."""
    if k <= 1 or a.shape[0] < k:
        return a.copy()
    kernel = {3: [1, 2, 1], 5: [1, 4, 6, 4, 1]}[k]
    half = k // 2
    valid = ~np.isnan(a)
    vals = np.where(valid, a, 0.0)
    num = np.zeros_like(vals, dtype=float)
    den = np.zeros_like(vals, dtype=float)
    T = a.shape[0]
    for off, w in zip(range(-half, half + 1), kernel):
        src = slice(max(0, off), T + min(0, off))
        dst = slice(max(0, -off), T + min(0, -off))
        num[dst] += w * vals[src]
        den[dst] += w * valid[src]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~valid] = np.nan
    return out


def rolling_nanmedian(x: np.ndarray, win: int) -> np.ndarray:
    win = max(win | 1, 3)
    half = win // 2
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        seg = x[max(0, i - half) : i + half + 1]
        if np.isfinite(seg).any():
            out[i] = np.nanmedian(seg)
    return out


@dataclass
class PoseTrack:
    """Landmarks of the selected player on the analysed frames.

    xy     (T, 33, 2) pixel coordinates in the normalised video, NaN when not tracked
    vis    (T, 33)    MediaPipe visibility estimate, 0 when not tracked
    world  (T, 33, 3) MediaPipe "world" landmarks in metres, hip-centred. These are a
                      single-camera estimate of 3D shape, not a validated 3D measurement.
    frame_idx (T,)    index of each analysed frame in the normalised video
    """

    fps: float  # frame rate of the normalised video
    width: int
    height: int
    frame_idx: np.ndarray
    xy: np.ndarray
    vis: np.ndarray
    world: np.ndarray
    seg: np.ndarray | None = None  # (T,) which tracked stretch each frame belongs to, -1 when not tracked
    segments: list | None = None  # one dict per stretch: seed, or re-acquired after a cut / leaving frame (see pose.track_player)

    def masked(self, confirmations: dict | None = None) -> "PoseTrack":
        """A copy without the stretches that still wait for, or were refused, the user's confirmation that it is the same player."""
        if self.seg is None or not self.segments:
            return self
        confirmations = confirmations or {}
        drop = [s["id"] for s in self.segments if s["status"] == "needs_confirmation" and confirmations.get(str(s["id"])) is not True]
        if not drop:
            return self
        hide = np.isin(self.seg, drop)
        xy, vis, world = self.xy.copy(), self.vis.copy(), self.world.copy()
        xy[hide], vis[hide], world[hide] = np.nan, 0.0, np.nan
        return PoseTrack(self.fps, self.width, self.height, self.frame_idx, xy, vis, world, np.where(hide, -1, self.seg), self.segments)

    # --- basic properties ---------------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.frame_idx)

    @cached_property
    def stride(self) -> int:
        return int(self.frame_idx[1] - self.frame_idx[0]) if self.n > 1 else 1

    @cached_property
    def rate(self) -> float:
        """Analysed frames per second."""
        return self.fps / self.stride

    @cached_property
    def t(self) -> np.ndarray:
        return self.frame_idx / self.fps

    @cached_property
    def tracked(self) -> np.ndarray:
        return (self.vis > 0).any(axis=1)

    def index_at(self, time_s: float) -> int:
        return int(np.clip(round(time_s * self.rate), 0, self.n - 1))

    def window(self, t0: float, t1: float) -> slice:
        a, b = self.index_at(t0), self.index_at(t1)
        return slice(a, max(b, a) + 1)

    # --- cleaned signals ----------------------------------------------------------------
    @cached_property
    def _k(self) -> int:
        return 5 if self.rate >= 45 else 3

    @cached_property
    def xy_s(self) -> np.ndarray:
        """Smoothed pixel coordinates; landmarks below MIN_VIS are masked out first."""
        xy = self.xy.copy()
        xy[self.vis < MIN_VIS] = np.nan
        xy = fill_short_gaps(xy, max_gap=int(round(0.12 * self.rate)))
        return smooth(xy, self._k)

    @cached_property
    def world_s(self) -> np.ndarray:
        w = self.world.copy()
        w[self.vis < MIN_VIS] = np.nan
        w = fill_short_gaps(w, max_gap=int(round(0.12 * self.rate)))
        return smooth(w, self._k)

    @cached_property
    def mid_hip(self) -> np.ndarray:
        return (self.xy_s[:, L_HIP] + self.xy_s[:, R_HIP]) / 2

    @cached_property
    def mid_sh(self) -> np.ndarray:
        return (self.xy_s[:, L_SH] + self.xy_s[:, R_SH]) / 2

    @cached_property
    def torso_len(self) -> np.ndarray:
        """Shoulder-midpoint to hip-midpoint distance in pixels, rolling 2 s median.

        Used as the body-size unit ("torso lengths") so speeds do not depend on how far
        the player is from the camera.
        """
        raw = np.linalg.norm(self.mid_sh - self.mid_hip, axis=1)
        return rolling_nanmedian(raw, int(2 * self.rate))

    def speed(self, landmark: int) -> np.ndarray:
        """Image-plane speed of a landmark in torso lengths per second."""
        p = self.xy_s[:, landmark]
        v = np.full(self.n, np.nan)
        if self.n >= 3:
            v[1:-1] = np.linalg.norm(p[2:] - p[:-2], axis=1) * self.rate / 2
        with np.errstate(invalid="ignore", divide="ignore"):
            return v / self.torso_len

    def hand_motion(self, wrist: int, min_vis: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
        """Wrist position relative to the shoulder midpoint (pixels) and its speed (torso lengths/s).

        Measuring relative to the torso removes running, walking and camera panning, which
        would otherwise look like hand speed. The wrist is kept down to a lower visibility
        than other landmarks because a fast-moving hand is often blurred; a 3-frame median
        removes single-frame landmark jumps before differentiating.
        """
        w = self.xy[:, wrist].copy()
        w[self.vis[:, wrist] < min_vis] = np.nan
        w = fill_short_gaps(w, max_gap=int(round(0.12 * self.rate)))
        rel = w - self.mid_sh
        if self.n >= 3:
            stack = np.stack([np.roll(rel, 1, axis=0), rel, np.roll(rel, -1, axis=0)])
            with np.errstate(all="ignore"):
                med = np.median(stack, axis=0)
            med[0], med[-1] = rel[0], rel[-1]
            med[np.isnan(rel)] = np.nan
            rel = np.where(np.isnan(med), rel, med)
        rel = smooth(rel, 3)
        v = np.full(self.n, np.nan)
        if self.n >= 3:
            v[1:-1] = np.linalg.norm(rel[2:] - rel[:-2], axis=1) * self.rate / 2
        with np.errstate(invalid="ignore", divide="ignore"):
            return rel, v / self.torso_len

    @cached_property
    def pelvis_azimuth(self) -> np.ndarray:
        """Angle between the hip line and the image plane, degrees, from world landmarks.

        ~0 means the pelvis faces the camera or faces directly away (frontal view);
        ~90 means it is seen side-on. Smoothed with a short median.
        """
        h = self.world_s[:, R_HIP] - self.world_s[:, L_HIP]
        az = np.degrees(np.arctan2(np.abs(h[:, 2]), np.abs(h[:, 0])))
        return rolling_nanmedian(az, int(0.15 * self.rate))

    def mean_vis(self, landmarks: list[int], sl: slice) -> float:
        v = self.vis[sl][:, landmarks]
        return float(v.mean()) if v.size else 0.0

    # --- persistence --------------------------------------------------------------------
    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, fps=self.fps, width=self.width, height=self.height, frame_idx=self.frame_idx,
            xy=self.xy.astype(np.float32), vis=self.vis.astype(np.float32), world=self.world.astype(np.float32),
            seg=self.seg if self.seg is not None else np.where((self.vis > 0).any(axis=1), 0, -1), segments=json.dumps(self.segments or []),
        )  # fmt: skip

    @classmethod
    def load(cls, path) -> "PoseTrack":
        d = np.load(path)
        seg = d["seg"] if "seg" in d else None
        segments = json.loads(str(d["segments"])) if "segments" in d else None
        return cls(float(d["fps"]), int(d["width"]), int(d["height"]), d["frame_idx"], d["xy"].astype(float), d["vis"].astype(float), d["world"].astype(float), seg, segments)

    def overlay_payload(self) -> dict:
        """Compact per-frame skeleton for the browser: x,y in 1/1000 of frame size, visibility 0-9."""
        frames: list = []
        for i in range(self.n):
            if not self.tracked[i]:
                frames.append(None)
                continue
            row: list[int] = []
            for j in OVERLAY_POINTS:
                x, y = self.xy[i, j]
                row += [int(round(x / self.width * 1000)), int(round(y / self.height * 1000)), int(round(self.vis[i, j] * 9))]
            frames.append(row)
        return {"fps": self.fps, "stride": self.stride, "points": OVERLAY_POINTS, "frames": frames}
