"""Ball tracking from ordinary video, and what it supports: contact time, contact location and hidden-arm recovery.

Detector choice (measured on the evaluation set; see docs/VALIDATION.md): a tennis ball in
phone footage is a few pixels wide and smeared by motion blur. Candidates are therefore
found from MOTION and COLOUR: pixels that differ from both the previous and the next frame
(after cancelling camera shift), grouped into small blobs and scored for tennis-ball
yellow-green. Candidates are then linked into short tracks that move smoothly; isolated
blobs are discarded. A pretrained general object detector (objects.py) was evaluated as an
alternative and is not used for the ball.

Everything here is image-plane (2D). When no usable ball track exists near a stroke the
stroke keeps its pose-based timestamp, and the report says which was used.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# --- heuristic constants (developer heuristics, tuned on the tune split only) ---------------
DIFF_THRESHOLD = 14  # grey-level change that counts as motion
MAX_BALL_FRAC = 0.035  # a ball blob is at most this fraction of frame height across (streaks can be longer)
MAX_STREAK_FRAC = 0.12
BALL_HUE = (20, 48)  # OpenCV hue range of tennis-ball yellow-green
MIN_SAT, MIN_VAL = 45, 90
MAX_STEP_FRAC = 0.16  # a ball moves at most this fraction of frame width between analysed frames
MIN_TRACK_LEN = 3
MIN_TRACK_TRAVEL_FRAC = 0.02


@dataclass
class Candidate:
    x: float
    y: float
    size: float  # longest side of the blob, px
    score: float  # 0-1, colour and shape


def shift_between(a_gray: np.ndarray, b_gray: np.ndarray) -> tuple[float, float]:
    """Global image shift from a to b (camera movement), from phase correlation on downscaled frames."""
    h, w = a_gray.shape
    s = 256.0 / max(h, w)
    a = cv2.resize(a_gray, (int(w * s), int(h * s))).astype(np.float32)
    b = cv2.resize(b_gray, (int(w * s), int(h * s))).astype(np.float32)
    (dx, dy), resp = cv2.phaseCorrelate(a, b)
    if resp < 0.05 or np.hypot(dx, dy) > 0.25 * a.shape[1]:
        return 0.0, 0.0
    return dx / s, dy / s


def _aligned(gray: np.ndarray, dx: float, dy: float) -> np.ndarray:
    if abs(dx) < 0.3 and abs(dy) < 0.3:
        return gray
    return cv2.warpAffine(gray, np.float32([[1, 0, dx], [0, 1, dy]]), (gray.shape[1], gray.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def find_candidates(prev_bgr: np.ndarray, cur_bgr: np.ndarray, next_bgr: np.ndarray, max_n: int = 12) -> list[Candidate]:
    """Small moving blobs in ``cur``, scored for looking like a tennis ball."""
    g0, g1, g2 = (cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in (prev_bgr, cur_bgr, next_bgr))
    g1b = cv2.GaussianBlur(g1, (3, 3), 0)
    d = []
    for g in (g0, g2):
        dx, dy = shift_between(g, g1)
        d.append(cv2.absdiff(g1b, cv2.GaussianBlur(_aligned(g, dx, dy), (3, 3), 0)))
    motion = (np.minimum(d[0], d[1]) > DIFF_THRESHOLD).astype(np.uint8)
    hsv = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2HSV)
    ballish = ((hsv[..., 0] >= BALL_HUE[0]) & (hsv[..., 0] <= BALL_HUE[1]) & (hsv[..., 1] >= MIN_SAT) & (hsv[..., 2] >= MIN_VAL)).astype(np.uint8)
    mask = cv2.dilate(motion, np.ones((3, 3), np.uint8)) & cv2.dilate(ballish, np.ones((3, 3), np.uint8))
    H, W = g1.shape
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        long_side, short_side = max(w, h), min(w, h)
        if area < 2 or long_side > MAX_STREAK_FRAC * H or short_side > MAX_BALL_FRAC * H:
            continue
        patch = ballish[y : y + h, x : x + w]
        colour = float(patch.mean())
        fill = area / float(w * h)
        score = min(1.0, 0.6 * colour + 0.4 * min(fill * 1.5, 1.0))
        if colour < 0.25:
            continue
        out.append(Candidate(float(cent[k][0]), float(cent[k][1]), float(long_side), score))
    out.sort(key=lambda c: -c.score)
    return out[:max_n]


@dataclass
class BallTrack:
    """One linked run of ball candidates. Positions are in pixels on analysed frames ``idx`` (indices into the pose track)."""

    idx: list[int] = field(default_factory=list)
    xy: list[tuple[float, float]] = field(default_factory=list)
    score: list[float] = field(default_factory=list)

    @property
    def travel(self) -> float:
        p = np.asarray(self.xy)
        return float(np.linalg.norm(p.max(0) - p.min(0))) if len(p) else 0.0

    def velocity(self, at_end: bool, n: int = 3) -> np.ndarray:
        """Mean velocity (px per analysed frame) over the last / first ``n`` steps."""
        p, i = np.asarray(self.xy), np.asarray(self.idx)
        sl = slice(-min(n + 1, len(p)), None) if at_end else slice(0, min(n + 1, len(p)))
        dt = i[sl][-1] - i[sl][0]
        return (p[sl][-1] - p[sl][0]) / dt if dt > 0 else np.zeros(2)


def link_tracks(per_frame: list[list[Candidate]], width: int, height: int, cut_frames: set[int] | None = None) -> list[BallTrack]:
    """Greedy constant-velocity linking. A track may skip up to two frames; isolated or stationary blobs are dropped."""
    cut_frames = cut_frames or set()
    max_step = MAX_STEP_FRAC * width
    active: list[tuple[BallTrack, int]] = []  # (track, frames since last hit)
    done: list[BallTrack] = []
    for i, cands in enumerate(per_frame):
        if i in cut_frames:
            done += [t for t, _ in active]
            active = []
        used: set[int] = set()
        nxt: list[tuple[BallTrack, int]] = []
        for tr, missed in sorted(active, key=lambda a: -len(a[0].idx)):
            gap = i - tr.idx[-1]
            last = np.asarray(tr.xy[-1])
            pred = last + tr.velocity(True) * gap if len(tr.idx) >= 2 else last
            gate = (0.035 * width + 0.5 * np.linalg.norm(tr.velocity(True))) * gap if len(tr.idx) >= 2 else max_step * gap
            best, best_d = -1, gate
            for k, c in enumerate(cands):
                if k in used:
                    continue
                dist = float(np.hypot(c.x - pred[0], c.y - pred[1]))
                if dist < best_d and np.hypot(c.x - last[0], c.y - last[1]) <= max_step * gap:
                    best, best_d = k, dist
            if best >= 0:
                used.add(best)
                c = cands[best]
                tr.idx.append(i), tr.xy.append((c.x, c.y)), tr.score.append(c.score)
                nxt.append((tr, 0))
            elif missed < 2:
                nxt.append((tr, missed + 1))
            else:
                done.append(tr)
        for k, c in enumerate(cands):
            if k not in used:
                nxt.append((BallTrack([i], [(c.x, c.y)], [c.score]), 0))
        active = nxt
    done += [t for t, _ in active]
    return [t for t in done if len(t.idx) >= MIN_TRACK_LEN and t.travel >= MIN_TRACK_TRAVEL_FRAC * width]


# --- contact ----------------------------------------------------------------------------------

FREE_TL = 2.8  # a real ball track has at least one point this many torso lengths from the hand (see contact_events)
REACH_TL = 2.2  # the ball must come within this many torso lengths of the hitting wrist to count as a hit
MAX_CONTACT_SHIFT_S = 0.3  # a ball-based contact further than this from the pose-based timestamp is not believed
MIN_TURN_DEG = 35.0


def _fit(idx: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Straight-line fit p(i) = a + b * i through up to four samples."""
    if len(idx) == 1:
        return xy[0], np.zeros(2)
    b = np.polyfit(idx, xy, 1)
    return b[1], b[0]


def contact_events(tracks: list[BallTrack], ref_xy: np.ndarray, torso_len: np.ndarray, lo: int, hi: int, rate: float, reach_tl: float = REACH_TL, target_xy: np.ndarray | None = None) -> list[dict]:
    """Places where the ball's path changes next to the player between analysed frames ``lo`` and ``hi``.

    ``ref_xy`` (T, 2) is the body point the ball must come near (the hitting wrist, or the chest when the arm is hidden).
    Three kinds of evidence, strongest first: an incoming and an outgoing path that meet; one path that turns sharply;
    one path that simply ends or begins at the racket. Each event: {"index" (fractional), "xy", "basis", "confidence"}.
    """
    n = len(ref_xy)
    target = target_xy if target_xy is not None else ref_xy

    def near(i: int, p) -> bool:
        i = int(np.clip(i, 0, n - 1))
        tl = torso_len[i]
        return bool(np.isfinite(tl) and np.isfinite(ref_xy[i]).all() and np.linalg.norm(np.asarray(p) - ref_xy[i]) <= reach_tl * tl)

    def free(t: BallTrack) -> bool:
        """A ball arrives from, or leaves to, somewhere beyond arm's reach. A blob that stays at racket distance from
        the hand for its whole life is the racket or the hand itself, not the ball."""
        for i, p in zip(t.idx, t.xy):
            i = int(np.clip(i, 0, n - 1))
            if np.isfinite(torso_len[i]) and np.isfinite(ref_xy[i]).all() and np.linalg.norm(np.asarray(p) - ref_xy[i]) > FREE_TL * torso_len[i]:
                return True
        return False

    out: list[dict] = []

    def offer(index: float, xy, basis: str, conf: float, both: bool) -> None:
        if lo <= index <= hi:
            out.append({"index": float(index), "xy": [float(xy[0]), float(xy[1])], "basis": basis, "confidence": conf, "two_sided": both})

    win = [t for t in tracks if t.idx[-1] >= lo - 2 and t.idx[0] <= hi + 2 and free(t)]
    for t in win:  # one track that turns sharply while within reach
        idx, xy = np.asarray(t.idx), np.asarray(t.xy)
        for k in range(1, len(idx) - 1):
            if not lo <= idx[k] <= hi or not near(idx[k], xy[k]):
                continue
            v1, v2 = (xy[k] - xy[k - 1]) / (idx[k] - idx[k - 1]), (xy[k + 1] - xy[k]) / (idx[k + 1] - idx[k])
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if min(n1, n2) < 1.0:
                continue
            turn = np.degrees(np.arccos(np.clip(v1 @ v2 / (n1 * n2), -1, 1)))
            if turn >= MIN_TURN_DEG or max(n1, n2) / min(n1, n2) >= 2.5:
                offer(float(idx[k]), xy[k], "the ball's path changed direction at the racket", 0.8, True)
    ends = [t for t in win if lo - 2 <= t.idx[-1] <= hi and near(t.idx[-1], t.xy[-1])]
    starts = [t for t in win if lo <= t.idx[0] <= hi + 2 and near(t.idx[0], t.xy[0])]
    for a in ends:  # incoming track ends and outgoing track starts within reach
        for b in starts:
            gap = b.idx[0] - a.idx[-1]
            if a is b or not 0 < gap <= max(2, int(0.25 * rate)):
                continue
            pa, va = _fit(np.asarray(a.idx[-4:]), np.asarray(a.xy[-4:]))
            pb, vb = _fit(np.asarray(b.idx[:4]), np.asarray(b.xy[:4]))
            dv = va - vb
            if np.linalg.norm(dv) < 1.0:
                continue
            ti = float(np.clip(-((pa - pb) @ dv) / (dv @ dv), a.idx[-1], b.idx[0]))  # closest approach of the two straight paths
            offer(ti, (pa + va * ti + pb + vb * ti) / 2, "an incoming and an outgoing ball path met at the racket", 0.9, True)
    for t, at_end in [(t, True) for t in ends] + [(t, False) for t in starts]:  # only one side seen: follow it to where it comes closest
        p0, v = _fit(np.asarray(t.idx[-4:] if at_end else t.idx[:4]), np.asarray(t.xy[-4:] if at_end else t.xy[:4]))
        edge = t.idx[-1] if at_end else t.idx[0]
        steps = np.arange(0, 0.16 * rate + 1e-6, 0.25) * (1 if at_end else -1) + edge
        cand = [(float(np.linalg.norm(p0 + v * s - target[int(np.clip(round(s), 0, n - 1))])), s) for s in steps if 0 <= round(s) < n and np.isfinite(target[int(round(s))]).all()]
        if cand:
            dist, s_ = min(cand)
            tl = torso_len[int(np.clip(round(s_), 0, n - 1))]
            if np.isfinite(tl) and dist <= 1.2 * tl:
                offer(float(s_), p0 + v * s_, "the incoming ball path ended at the racket" if at_end else "the outgoing ball path began at the racket", 0.55, False)
    return out


def estimate_contact(tracks: list[BallTrack], wrist_xy: np.ndarray, torso_len: np.ndarray, peak_i: int, rate: float, racket_xy: np.ndarray | None = None) -> dict | None:
    """The most believable ball-racket contact near analysed frame ``peak_i`` (the pose-based timestamp), or None.

    A contact more than MAX_CONTACT_SHIFT_S from the pose-based timestamp is not believed: the caller then keeps the
    pose-based time and says so.
    """
    w = int(round(0.4 * rate))
    best, best_score = None, -np.inf
    for e in contact_events(tracks, wrist_xy, torso_len, max(0, peak_i - w), min(len(wrist_xy) - 1, peak_i + w), rate, target_xy=racket_xy):
        shift = abs(e["index"] - peak_i) / (MAX_CONTACT_SHIFT_S * rate)
        if shift <= 1 and e["confidence"] - 0.3 * shift > best_score:
            best, best_score = e, e["confidence"] - 0.3 * shift
    return best


def to_json(tracks: list[BallTrack]) -> list[dict]:
    return [{"idx": t.idx, "xy": [[round(x, 1), round(y, 1)] for x, y in t.xy]} for t in tracks]


def from_json(data: list[dict] | None) -> list[BallTrack]:
    return [BallTrack(list(d["idx"]), [tuple(p) for p in d["xy"]], [1.0] * len(d["idx"])) for d in data or []]
