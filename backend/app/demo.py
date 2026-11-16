"""Synthetic player: a simple kinematic stick figure performing scripted strokes.

Used by the test suite (known ground truth for stroke times, types and knee bend) and by
the optional, clearly labelled demo session (ENABLE_DEMO=1). It is NOT a model of real
tennis and says nothing about accuracy on real footage. Nothing here runs for real uploads.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .pipeline.pose import FramePoses
from .pipeline.signals import N_LANDMARKS, PoseTrack

# body frame: X = player's right, Y = up, Z = forward (metres)
REST = np.array([0.30, 1.00, 0.25])
PATHS = {
    "forehand": [(-0.45, [0.60, 1.00, -0.50]), (-0.15, [0.75, 0.95, -0.35]), (0.0, [0.55, 1.00, 0.55]), (0.25, [-0.35, 1.50, 0.20]), (0.6, REST)],
    "backhand": [(-0.45, [-0.50, 1.00, -0.30]), (-0.15, [-0.55, 0.95, -0.25]), (0.0, [0.30, 1.00, 0.60]), (0.25, [0.75, 1.40, 0.20]), (0.6, REST)],
    # one-handed backhand whose follow-through finishes well above the head, like a serve's reach
    "backhand_high_finish": [(-0.45, [-0.50, 1.00, -0.30]), (-0.15, [-0.55, 0.95, -0.25]), (0.0, [0.30, 1.00, 0.60]), (0.3, [0.70, 2.15, 0.10]), (0.75, REST)],
    "serve": [(-0.9, [0.35, 1.00, 0.10]), (-0.25, [0.30, 1.40, -0.35]), (0.0, [0.25, 2.25, 0.30]), (0.3, [-0.30, 0.90, 0.30]), (0.7, REST)],
}
IDX = dict(nose=0, l_sh=11, r_sh=12, l_el=13, r_el=14, l_wr=15, r_wr=16, l_hip=23, r_hip=24, l_knee=25, r_knee=26, l_ank=27, r_ank=28, l_heel=29, r_heel=30, l_toe=31, r_toe=32)


def _wrist(t: float, events: list[tuple[float, str]]) -> np.ndarray:
    for tp, kind in events:
        keys = PATHS[kind]
        if tp + keys[0][0] - 0.2 <= t <= tp + keys[-1][0]:
            times = [tp + keys[0][0] - 0.2] + [tp + k[0] for k in keys]
            pts = [REST] + [np.asarray(k[1], float) for k in keys]
            for a in range(len(times) - 1):
                if times[a] <= t <= times[a + 1]:
                    u = (t - times[a]) / (times[a + 1] - times[a])
                    u = u * u * (3 - 2 * u)
                    return pts[a] + (pts[a + 1] - pts[a]) * u
    return REST + np.array([0.0, 0.02 * np.sin(2 * t), 0.0])


def _toss(t: float, events, two_handed: bool = False) -> np.ndarray:
    base = np.array([-0.30, 1.00, 0.25])
    for tp, kind in events:
        if two_handed and kind == "backhand" and tp - 0.65 <= t <= tp + 0.3:
            return _wrist(t, events) + np.array([-0.12, 0.0, 0.05])

        if kind == "serve" and tp - 1.3 <= t <= tp + 0.2:
            u = np.clip(1 - abs((t - (tp - 0.6)) / 0.7), 0, 1)
            return base + u * np.array([0.1, 1.0, 0.1])
    return base


def body_points(t: float, events, knee_flex: float, two_handed: bool = False) -> np.ndarray:
    """(33, 3) joints in the body frame for a right-hander."""
    flex = 5.0
    for tp, kind in events:
        lo, hi = (tp - 0.9, tp - 0.1) if kind == "serve" else (tp - 0.5, tp + 0.05)
        if lo <= t <= hi:
            u = np.sin(np.pi * (t - lo) / (hi - lo))
            flex = max(flex, 5.0 + (knee_flex - 5.0) * u)
    a = np.radians(flex / 2)
    seg = 0.45
    knee_y, hip_y = 0.05 + seg * np.cos(a), 0.05 + 2 * seg * np.cos(a)
    P = np.zeros((N_LANDMARKS, 3))
    for side, sx in (("r", 0.14), ("l", -0.14)):
        P[IDX[f"{side}_ank"]] = [sx, 0.05, 0]
        P[IDX[f"{side}_heel"]] = [sx, 0.0, -0.05]
        P[IDX[f"{side}_toe"]] = [sx, 0.0, 0.18]
        P[IDX[f"{side}_knee"]] = [sx, knee_y, seg * np.sin(a)]
        P[IDX[f"{side}_hip"]] = [sx, hip_y, 0]
        P[IDX[f"{side}_sh"]] = [sx * 1.35, hip_y + 0.5, 0]
    P[IDX["nose"]] = [0, hip_y + 0.67, 0.08]
    for j in range(1, 11):
        P[j] = P[0] + [0.01 * (j - 5), 0.02, 0]
    drop = np.array([0, hip_y - 0.95, 0])
    P[IDX["r_wr"]] = _wrist(t, events) + drop
    P[IDX["l_wr"]] = _toss(t, events, two_handed) + drop
    for side in "rl":
        s, w = P[IDX[f"{side}_sh"]], P[IDX[f"{side}_wr"]]
        P[IDX[f"{side}_el"]] = (s + w) / 2 + [0, -0.05, -0.05]
    for j in range(17, 23):
        P[j] = P[IDX["r_wr" if j % 2 == 0 else "l_wr"]]
    return P


def project(P: np.ndarray, yaw_deg: float, left_handed: bool, origin_px: tuple[float, float], scale: float):
    """Body frame -> (pixel xy (33,2), hip-centred camera-frame world (33,3))."""
    P = P.copy()
    if left_handed:  # mirror the body and swap left/right labels
        P[:, 0] *= -1
        for a, b in ((11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)):
            P[[a, b]] = P[[b, a]]
    th = np.radians(yaw_deg)
    xc = P[:, 0] * np.cos(th) + P[:, 2] * np.sin(th)
    zc = -P[:, 0] * np.sin(th) + P[:, 2] * np.cos(th)
    cam = np.stack([xc, -P[:, 1], zc], axis=1)
    xy = np.stack([origin_px[0] + scale * xc, origin_px[1] - scale * P[:, 1]], axis=1)
    return xy, cam - (cam[23] + cam[24]) / 2


def make_track(events, *, duration=None, fps=30.0, yaw=135.0, knee_flex=60.0, left_handed=False, two_handed=False, width=1280, height=720, noise=0.0, seed=0) -> PoseTrack:
    duration = duration or (max(t for t, _ in events) + 1.5 if events else 4.0)
    n = int(duration * fps)
    rng = np.random.default_rng(seed)
    xy = np.zeros((n, N_LANDMARKS, 2))
    world = np.zeros((n, N_LANDMARKS, 3))
    for i in range(n):
        xy[i], world[i] = project(body_points(i / fps, events, knee_flex, two_handed), yaw, left_handed, (width / 2, height * 0.92), height * 0.3)
    if noise:
        xy += rng.normal(0, noise, xy.shape)
    return PoseTrack(fps=fps, width=width, height=height, frame_idx=np.arange(n), xy=xy, vis=np.full((n, N_LANDMARKS), 0.95), world=world)


class FakeDetector:
    """Stands in for MediaPipe in API tests: returns scripted poses for however many frames the video has."""

    def __init__(self, events, *, people_x=(0.5,), fail_times=0, **kw) -> None:
        self.events, self.people_x, self.kw, self.fail_times, self.calls = events, people_x, kw, fail_times, 0

    def _poses(self, t: float, width: int, height: int) -> FramePoses:
        lm = np.zeros((len(self.people_x), N_LANDMARKS, 4), dtype=np.float32)
        world = np.zeros((len(self.people_x), N_LANDMARKS, 3), dtype=np.float32)
        for p, x in enumerate(self.people_x):
            ev = self.events if p == len(self.people_x) - 1 else []  # only the last person swings
            xy, w = project(body_points(t, ev, self.kw.get("knee_flex", 60.0)), self.kw.get("yaw", 135.0), False, (width * x, height * 0.92), height * 0.3)
            lm[p, :, 0], lm[p, :, 1], lm[p, :, 3] = xy[:, 0] / width, xy[:, 1] / height, 0.95
            world[p] = w
        return FramePoses(lm, world)

    def detect_image(self, frame) -> FramePoses:
        return self._poses(0.0, frame.shape[1], frame.shape[0])

    def detect_video(self, path: Path, stride: int, progress):
        import cv2

        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("simulated crash")
        cap = cv2.VideoCapture(str(path))
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        idx = list(range(0, n, stride))
        progress(1.0)
        return idx, [self._poses(i / fps, w, h) for i in idx]


def make_video(path: Path, *, duration=4.0, fps=30, size="640x360", source="testsrc2", extra: list[str] | None = None) -> Path:
    import subprocess

    from .pipeline.video import ffmpeg_exe

    cmd = [ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"{source}{':' if '=' in source else '='}size={size}:rate={fps}:duration={duration}", *(extra or []), "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


DEMO_EVENTS = [(2.0, "forehand"), (4.5, "backhand"), (7.5, "serve"), (10.5, "serve"), (13.5, "serve"), (16.0, "forehand")]
DEMO_MARKER = "DEMO"
BONES = [(11, 12), (11, 23), (12, 24), (23, 24), (11, 13), (13, 15), (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32)]


def render_demo_video(path: Path, *, duration=18.0, fps=30, width=960, height=540, yaw=120.0, knee_flex=18.0) -> Path:
    """Draw the scripted stick figure to a video file so the overlay can be checked against it."""
    import cv2

    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(int(duration * fps)):
        img = np.full((height, width, 3), (96, 132, 74), np.uint8)
        cv2.rectangle(img, (0, int(height * 0.92)), (width, height), (70, 100, 55), -1)
        cv2.putText(img, "DEMO - synthetic figure, not a real player", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        xy, _ = project(body_points(i / fps, DEMO_EVENTS, knee_flex), yaw, False, (width / 2, height * 0.92), height * 0.3)
        for a, b in BONES:
            cv2.line(img, tuple(map(int, xy[a])), tuple(map(int, xy[b])), (40, 40, 40), 6, cv2.LINE_AA)
        cv2.circle(img, tuple(map(int, xy[0])), int(height * 0.035), (40, 40, 40), -1, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    return path


def demo_detector() -> FakeDetector:
    return FakeDetector(DEMO_EVENTS, yaw=120.0, knee_flex=18.0)
