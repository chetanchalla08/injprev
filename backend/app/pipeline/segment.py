"""Find swings in a pose track and label them forehand / backhand / serve.

Rule-based and documented in docs/ANALYSIS_LOGIC.md. The constants were set by looking at
a small amount of real footage (see docs/VALIDATION.md for the measured accuracy and its
limits); they are heuristics, every label carries a confidence, and the user can correct
any label, after which measurements are recomputed.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass

import numpy as np

from .signals import NOSE, PoseTrack, sides

STROKE_TYPES = ("forehand", "backhand", "serve")

# --- heuristic constants ----------------------------------------------------------------
MIN_PEAK_SPEED = 6.0  # torso-relative hand speed (torso lengths/s) below which nothing is a swing
MIN_EXTENT = 1.2  # the hand must travel at least this far (torso lengths) within +-0.3 s of the peak
MIN_SWING_VIS = 0.45  # mean hitting-wrist visibility needed around the peak
MIN_GAP_S = 0.7  # swings closer together than this are merged (the faster one is kept)
SERVE_REACH = 0.5  # hand at least this many torso lengths above the nose
SIDE_MIN_SCORE = 0.6  # minimum lateral travel (hip widths) to call forehand vs backhand
HANDS_TOGETHER_M = 0.35  # two-handed backhand: wrists closer than this (model metres) through the swing
HANDS_APART_M = 0.42


@dataclass
class Stroke:
    id: str
    type: str  # forehand | backhand | serve | unclassified
    start: float
    end: float
    peak: float  # groundstrokes: peak hitting-hand speed; serves: highest hand position. Not ball contact.
    source: str = "auto"  # auto | manual
    label_confidence: float = 0.0
    label_basis: str = ""
    timing_source: str = "peak_hand_speed"  # ball_contact | peak_hand_speed | highest_hand | manual_range
    pose_time: float | None = None  # the pose-based timestamp, kept when ball contact replaces it
    contact: dict | None = None  # ball position at contact etc., when found

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("start", "end", "peak"):
            d[k] = round(d[k], 3)
        d["label_confidence"] = round(d["label_confidence"], 2)
        if d["pose_time"] is not None:
            d["pose_time"] = round(d["pose_time"], 3)
        return d


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def hand_speed(track: PoseTrack, handedness: str) -> np.ndarray:
    dom, _ = sides(handedness)
    _, v = track.hand_motion(dom.wr)
    return np.where(np.isfinite(v), v, 0.0)


def _extent(rel: np.ndarray, sl: slice, scale: float) -> float:
    seg = rel[sl]
    if not np.isfinite(seg).any() or not np.isfinite(scale) or scale <= 0:
        return 0.0
    return float(np.linalg.norm(np.nanmax(seg, axis=0) - np.nanmin(seg, axis=0)) / scale)


def find_swings(track: PoseTrack, handedness: str) -> list[int]:
    """Frame indices of candidate groundstroke swings: fast, large, visible hand movements."""
    dom, _ = sides(handedness)
    rel, v = track.hand_motion(dom.wr)
    v = np.where(np.isfinite(v), v, 0.0)
    w = max(int(round(0.3 * track.rate)), 2)
    cands = []
    for i in range(1, track.n - 1):
        if v[i] < MIN_PEAK_SPEED or v[i] < v[i - 1] or v[i] <= v[i + 1]:
            continue
        sl = slice(max(0, i - w), min(track.n, i + w + 1))
        if _extent(rel, sl, track.torso_len[i]) < MIN_EXTENT or track.vis[sl, dom.wr].mean() < MIN_SWING_VIS:
            continue
        cands.append(i)
    cands.sort(key=lambda i: -v[i])
    kept: list[int] = []
    for i in cands:
        if all(abs(i - k) >= MIN_GAP_S * track.rate for k in kept):
            kept.append(i)
    return sorted(kept)


FOLLOW_THROUGH_LOW = -0.5  # hand at least this many torso lengths BELOW the nose at its fastest
FOLLOW_THROUGH_LEAD_S = 0.1  # ...and fastest at least this long before it reached its highest point


def _is_follow_through(reach: np.ndarray, fastest: int, top: int, rate: float) -> bool:
    """True when a high hand is the finish of a groundstroke rather than a serve.

    In a serve the hitting hand is fastest near the top, around contact. In a groundstroke
    it is fastest at about waist height and rises only afterwards, in the follow-through;
    a one-handed backhand often finishes well above the head, which is what made it look
    like a serve. Set from the biomechanics of the two strokes and checked on synthetic
    swings; not tuned on the held-out footage.
    """
    h = reach[fastest]
    return bool(np.isfinite(h) and h <= FOLLOW_THROUGH_LOW and top - fastest >= FOLLOW_THROUGH_LEAD_S * rate)


def _fastest(track: PoseTrack, wrist: int, speed2d: np.ndarray, lo: int, hi: int) -> int:
    """Frame of fastest hitting-hand movement in [lo, hi). Uses the single-camera 3D estimate when it is
    available, because a swing toward or away from the camera nearly vanishes in the 2D image; 2D otherwise."""
    hi = min(hi, track.n)
    w = track.world[lo:hi, wrist] - (track.world[lo:hi, 11] + track.world[lo:hi, 12]) / 2
    ok = track.vis[lo:hi, wrist] >= 0.3
    if hi - lo >= 3 and ok.mean() >= 0.8 and np.isfinite(w).all():
        v = np.zeros(hi - lo)
        v[1:-1] = np.linalg.norm(w[2:] - w[:-2], axis=1)
        v[~ok] = 0.0
        return lo + int(np.argmax(v))
    return lo + int(np.argmax(speed2d[lo:hi]))


def find_serves(track: PoseTrack, handedness: str) -> list[tuple[int, float, str]]:
    """Serve / overhead events: (frame of highest hand position, confidence, basis)."""
    dom, non = sides(handedness)
    r = track.rate
    y = track.xy[:, :, 1]
    ok = (track.vis[:, dom.wr] >= 0.3) & track.tracked
    with np.errstate(invalid="ignore", divide="ignore"):
        reach = np.where(ok, (y[:, NOSE] - y[:, dom.wr]) / track.torso_len, np.nan)
        toss = np.where(track.vis[:, non.wr] >= 0.3, (y[:, NOSE] - y[:, non.wr]) / track.torso_len, np.nan)
    speed = hand_speed(track, handedness)
    high = np.nan_to_num(reach, nan=-9.0) >= SERVE_REACH
    out: list[tuple[int, float, str]] = []
    i = 0
    while i < track.n:
        if not high[i]:
            i += 1
            continue
        j = i
        while j + 1 < track.n and (high[j + 1] or (j + 2 < track.n and high[j + 2])):  # bridge one-frame dropouts
            j += 1
        dur = (j - i + 1) / r
        if 0.08 <= dur <= 1.5:
            top = i + int(np.nanargmax(reach[i : j + 1]))
            pre = toss[max(0, top - int(1.6 * r)) : max(1, top - int(0.1 * r))]
            has_toss = bool(np.isfinite(pre).any() and np.nanmax(pre) >= 0.2)
            lo = max(0, top - int(0.4 * r))
            near = speed[lo : top + int(0.4 * r) + 1]
            fast = float(near.max()) >= MIN_PEAK_SPEED
            if fast and _is_follow_through(reach, _fastest(track, dom.wr, speed, lo, top + int(0.4 * r) + 1), top, r):
                i = j + 1  # a groundstroke finishing high (e.g. a one-handed backhand); find_swings labels it
                continue
            if has_toss or fast:
                conf = 0.5 + (0.25 if has_toss else 0.0) + (0.15 if fast else 0.0)
                basis = "Hitting hand rose well above head height" + (
                    " after the other arm was raised (toss)." if has_toss else "; no toss-arm lift was seen, so this could be an overhead."
                )
                if not fast:
                    basis += " The hitting arm was hard to see during the swing, so its speed could not be confirmed."
                out.append((top, conf, basis))
        i = j + 1
    return out


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    ok = np.isfinite(values) & (weights >= 0.4)
    return float(values[ok].mean()) if ok.any() else float("nan")


def classify(track: PoseTrack, peak_i: int, handedness: str, backhand_style: str = "two_handed") -> tuple[str, float, str]:
    """Label a groundstroke swing forehand or backhand."""
    dom, non = sides(handedness)
    r, n = track.rate, track.n
    w = track.world
    vis_d, vis_n = track.vis[:, dom.wr], track.vis[:, non.wr]

    # If the hitting arm was hidden for most of the lead-up, what we see may be only the tail of a swing.
    lead = vis_d[max(0, peak_i - int(1.2 * r)) : max(1, peak_i - int(0.4 * r))]
    if lead.size >= 3 and (lead < 0.5).mean() > 0.5:
        return "unclassified", 0.1, "The hitting arm was hidden from the camera in the lead-up, so this may be only the end of a swing."

    # Cue 1: where the hand is along the hip line before vs after the peak (hip widths, + = hitting-arm side).
    axis = w[:, dom.hip] - w[:, non.hip]
    mid = (w[:, dom.hip] + w[:, non.hip]) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        lat = np.sum((w[:, dom.wr] - mid) * axis, axis=1) / np.sum(axis * axis, axis=1)
    a, b = slice(max(0, peak_i - int(0.4 * r)), max(1, peak_i - int(0.1 * r)) + 1), slice(min(n - 1, peak_i + int(0.05 * r)), min(n, peak_i + int(0.3 * r) + 1))
    pre, post = _weighted_mean(lat[a], vis_d[a]), _weighted_mean(lat[b], vis_d[b])
    score = pre - post if np.isfinite(pre) and np.isfinite(post) else float("nan")
    az = track.pelvis_azimuth[peak_i]
    side_on = np.isfinite(az) and az > 60  # hips overlap in the image, so left/right depth is unreliable

    # Cue 2 (two-handed backhand players): are both hands together through the swing?
    sep = float("nan")
    if backhand_style == "two_handed":
        c = slice(max(0, peak_i - int(0.4 * r)), peak_i + 1)
        both = (vis_d[c] >= 0.3) & (vis_n[c] >= 0.3)
        if both.sum() >= 3:
            sep = float(np.median(np.linalg.norm(w[c, dom.wr] - w[c, non.wr], axis=1)[both]))

    if np.isfinite(sep):
        if sep < HANDS_TOGETHER_M:
            agree = np.isfinite(score) and score < -SIDE_MIN_SCORE
            clash = np.isfinite(score) and score > 1.5
            return "backhand", 0.45 if clash else 0.85 if agree else 0.7, "Both hands stayed together through the swing" + (", and the hands travelled out toward the hitting-arm side." if agree else ".")
        if sep > HANDS_APART_M:
            if np.isfinite(score) and score < -1.5:
                return "unclassified", 0.2, "The hands were apart (like a forehand) but the hand travelled like a backhand; the view was not clear enough to decide."
            agree = np.isfinite(score) and score > SIDE_MIN_SCORE
            return "forehand", 0.85 if agree else 0.65, "The hitting hand swung on its own, away from the other hand" + (", from the hitting-arm side across the body." if agree else ".")

    if not np.isfinite(score):
        return "unclassified", 0.0, "The hitting arm or hips were not visible enough around this swing to label it."
    if abs(score) < SIDE_MIN_SCORE:
        return "unclassified", 0.2, "The hand's path across the body was too small or unclear to tell forehand from backhand."
    conf = float(np.clip(0.35 + 0.2 * abs(score), 0.0, 0.5 if side_on else 0.8))
    note = " Seen side-on, so left and right are harder to tell apart." if side_on else ""
    if score > 0:
        return "forehand", conf, "Hitting hand started on the hitting-arm side of the hips and travelled across the body." + note
    return "backhand", conf, "Hitting hand started across the body and travelled out to the hitting-arm side." + note


def _bounds(speed: np.ndarray, peak_i: int, rate: float, kind: str) -> tuple[int, int]:
    pk = max(speed[peak_i], MIN_PEAK_SPEED)
    lo = max(0, peak_i - int((1.8 if kind == "serve" else 1.2) * rate))
    hi = min(len(speed) - 1, peak_i + int(1.0 * rate))
    a = peak_i
    while a > lo and (speed[a] > 0.15 * pk or peak_i - a < 0.15 * rate):
        a -= 1
    b = peak_i
    while b < hi and (speed[b] > 0.2 * pk or b - peak_i < 0.15 * rate):
        b += 1
    pad_pre = int((1.0 if kind == "serve" else 0.3) * rate)  # include toss / unit turn
    return max(lo, a - pad_pre), min(len(speed) - 1, b + int(0.25 * rate))


def detect_strokes(track: PoseTrack, handedness: str, backhand_style: str = "two_handed") -> list[Stroke]:
    dom, _ = sides(handedness)
    r = track.rate
    speed = hand_speed(track, handedness)
    events: list[tuple[int, str, float, str]] = [(i, "serve", c, b) for i, c, b in find_serves(track, handedness)]
    serve_frames = [e[0] for e in events]
    for i in find_swings(track, handedness):
        # the wind-up and follow-through of a serve are part of that serve, not separate swings
        if any(-1.6 * r <= i - s <= 1.0 * r for s in serve_frames):
            continue
        events.append((i, *classify(track, i, handedness, backhand_style)))
    events.sort()
    merged: list[tuple[int, str, float, str]] = []
    for e in events:
        if merged and e[0] - merged[-1][0] < MIN_GAP_S * r:
            if e[1] == "serve" and merged[-1][1] != "serve":
                merged[-1] = e
            continue
        merged.append(e)
    strokes: list[Stroke] = []
    prev_end = 0
    for k, (p, kind, conf, basis) in enumerate(merged):
        a, b = _bounds(speed, p, r, kind)
        a = max(a, prev_end)
        if k + 1 < len(merged):
            b = min(b, (p + merged[k + 1][0]) // 2)
        while b > p and not track.tracked[b]:  # a stroke never extends over frames where the player was not tracked (a cut, or leaving the frame)
            b -= 1
        while a < p and not track.tracked[a]:
            a += 1
        prev_end = b
        strokes.append(Stroke(_new_id(), kind, float(track.t[a]), float(track.t[b]), float(track.t[p]), "auto", conf, basis, "highest_hand" if kind == "serve" else "peak_hand_speed", float(track.t[p])))
    return strokes


def manual_stroke(track: PoseTrack, handedness: str, start: float, end: float, kind: str, stroke_id: str | None = None) -> Stroke:
    """Build a stroke from a user-chosen time range and label."""
    start, end = max(0.0, min(start, end)), min(float(track.t[-1]), max(start, end))
    sl = track.window(start, end)
    speed = hand_speed(track, handedness)[sl]
    peak = float(track.t[sl.start + int(np.argmax(speed))]) if speed.size and speed.max() > 0 else (start + end) / 2
    return Stroke(stroke_id or _new_id(), kind, start, end, peak, "manual", 1.0, "Labelled by you.", "peak_hand_speed", peak)


TIMING_LABEL = {
    "ball_contact": "Ball contact, found by tracking the ball",
    "peak_hand_speed": "Peak hitting-hand speed (the ball was not tracked well enough here; this is near contact, not contact)",
    "highest_hand": "Highest point of the hitting hand (the ball was not tracked well enough here; this is near contact, not contact)",
}
HIDDEN_ARM_VIS = 0.45


def apply_ball(track: PoseTrack, strokes: list[Stroke], ball_tracks: list, handedness: str, recover_hidden: bool = True) -> list[Stroke]:
    """Replace each stroke's pose-based timestamp with ball contact where the ball's path shows one, and (optionally)
    add strokes that the pose alone missed because the hitting arm was hidden. Strokes without a contact keep the
    pose-based time; ``timing_source`` records which was used."""
    from . import ball

    dom, _ = sides(handedness)
    wrist = track.xy[:, dom.wr].copy()
    wrist[track.vis[:, dom.wr] < 0.3] = np.nan
    from .signals import fill_short_gaps

    wrist = fill_short_gaps(wrist, int(0.2 * track.rate))
    for s in strokes:
        s.pose_time = s.pose_time if s.pose_time is not None else s.peak
        s.peak, s.contact = s.pose_time, None
        if s.timing_source == "ball_contact":
            s.timing_source = "highest_hand" if s.type == "serve" else "peak_hand_speed"
        est = ball.estimate_contact(ball_tracks, wrist, track.torso_len, track.index_at(s.pose_time), track.rate) if ball_tracks else None
        if est is None:
            continue
        t = float(est["index"] / track.rate)
        if not s.start < t < s.end:
            continue
        s.peak, s.timing_source = t, "ball_contact"
        s.contact = {"xy": est["xy"], "basis": est["basis"], "confidence": est["confidence"], "shift_s": round(t - s.pose_time, 3)}
    if not (recover_hidden and ball_tracks):
        return strokes
    chest = track.mid_sh.copy()
    r = track.rate
    found: list[Stroke] = []
    for e in ball.contact_events(ball_tracks, chest, track.torso_len, 0, track.n - 1, r, reach_tl=2.0):
        i = int(round(e["index"]))
        t = float(e["index"] / r)
        if not e["two_sided"] or not track.tracked[i]:
            continue  # a path that merely ends near the body is too weak to claim a stroke on its own
        w = slice(max(0, i - int(0.3 * r)), min(track.n, i + int(0.3 * r) + 1))
        if track.vis[w, dom.wr].mean() >= HIDDEN_ARM_VIS:
            continue  # the arm was visible and no swing was found, so this is more likely a bounce or a ball passing by
        if any(abs(t - s.peak) < MIN_GAP_S or s.start <= t <= s.end for s in (*strokes, *found)):
            continue
        a, b = max(0.0, t - 0.8), min(float(track.t[-1]), t + 0.7)
        found.append(Stroke(_new_id(), "unclassified", a, b, t, "auto", 0.3, "Found from the ball changing direction next to the player while the hitting arm was hidden from the camera. "
                            "The stroke type could not be told.", "ball_contact", None, {"xy": e["xy"], "basis": e["basis"], "confidence": e["confidence"], "shift_s": None}))  # fmt: skip
    return sorted([*strokes, *found], key=lambda s: s.peak)


def phases(track: PoseTrack, stroke: Stroke, handedness: str) -> dict | None:
    """Preparation / forward swing / follow-through boundaries from the hitting-hand speed and the stroke timestamp."""
    speed = hand_speed(track, handedness)
    i = track.index_at(stroke.peak)
    a, b = track.index_at(stroke.start), track.index_at(stroke.end)
    if b - a < 4 or not a < i < b:
        return None
    near = slice(max(a, i - int(0.15 * track.rate)), min(b, i + int(0.15 * track.rate)) + 1)
    pk = float(speed[near].max())
    if pk <= 0:
        return None
    j = i
    while j > a and (speed[j] > 0.3 * pk or j > i - 2):
        j -= 1
    fwd = float(track.t[j])
    if not stroke.start < fwd < stroke.peak:
        return None
    return {"preparation": [round(stroke.start, 3), round(fwd, 3)], "forward_swing": [round(fwd, 3), round(stroke.peak, 3)], "follow_through": [round(stroke.peak, 3), round(stroke.end, 3)],
            "forward_swing_s": round(stroke.peak - fwd, 2), "boundary": "contact" if stroke.timing_source == "ball_contact" else "pose-based timestamp (not ball contact)",
            "note": "The forward swing is taken to start when the hitting hand last dropped below 30% of its peak speed. Image-plane hand speed; a rough guide, not a timing instrument."}  # fmt: skip
