"""Pipeline orchestration: prepare (validate + find people), analyze (pose + tracking + report), rebuild (after manual edits or a re-acquisition decision).

Files written per session: video.mp4 (normalised copy), candidates.jpg, track.npz (pose of the
selected player, with tracked stretches), aux.json (ball tracks, cuts, court positions: what a
rebuild needs without re-reading the video) and overlay.json (what the browser draws).
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import ball as ballmod
from . import court as courtmod
from .errors import PipelineError
from .findings import build_report
from .measure import measure_stroke
from .options import Options
from .pose import Detector, find_candidates, refine_with_crops, track_player
from .segment import STROKE_TYPES, Stroke, apply_ball, clear_ball_side, detect_strokes, manual_stroke, phases
from .signals import PoseTrack
from .video import normalize, quality_checks
from .visual import Visual, visual_pass

Progress = Callable[[str, float], None]
VIDEO, TRACK, OVERLAY, PICKER, AUX = "video.mp4", "track.npz", "overlay.json", "candidates.jpg", "aux.json"
MIN_TRACKED = 0.3
STROKE_FIELDS = ("id", "type", "start", "end", "peak", "source", "label_confidence", "label_basis", "timing_source", "pose_time", "contact")


def prepare(session_dir: Path, upload: Path, detector: Detector, *, max_duration_s: float, min_duration_s: float, progress: Progress) -> dict:
    progress("Checking and converting the video", 0.05)
    video = session_dir / VIDEO
    meta = normalize(upload, video, max_duration_s=max_duration_s, min_duration_s=min_duration_s)
    upload.unlink(missing_ok=True)  # the original is not kept once the normalised copy exists
    progress("Checking recording quality", 0.5)
    quality = quality_checks(video, meta)
    progress("Looking for players", 0.7)
    found = find_candidates(detector, video, meta)
    cv2.imwrite(str(session_dir / PICKER), found["frame"], [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {"meta": meta, "quality": quality, "candidates": {"frame_idx": found["frame_idx"], "people": found["candidates"]}}


def compose(track: PoseTrack, aux: dict, *, intake: dict, quality: dict, meta: dict, confirmations: dict | None, options: Options, strokes: list[Stroke] | None = None) -> dict:
    """Everything after tracking: strokes, ball contact, phases, measurements, movement, report. Shared by analyze and rebuild."""
    hand = intake.get("handedness", "right")
    confirmations = confirmations or {}
    tm = track.masked(confirmations)
    if strokes is None:
        strokes = detect_strokes(tm, hand, intake.get("backhand_style", "two_handed"))
    ball_tracks = ballmod.from_json(aux.get("ball_tracks")) if options.ball else []
    if options.ball:
        strokes = apply_ball(tm, strokes, ball_tracks, hand, recover_hidden=options.recover_hidden)
    else:
        strokes = clear_ball_side(strokes)
    measures = {s.id: measure_stroke(tm, s, hand, with_contact=options.ball) for s in strokes}
    extras: dict = {"ball_on": options.ball, "options": options.to_dict(), "phases": {s.id: phases(tm, s, hand) for s in strokes}}

    court = aux.get("court")
    if options.court and court:
        pos = np.asarray([[np.nan, np.nan] if p is None else p for p in court.get("pos") or []], float).reshape(-1, 2)
        info = {k: v for k, v in court.items() if k != "pos"}
        if info["status"] == "ok" and len(pos) == tm.n:
            pos[~tm.tracked] = np.nan
            extras["movement"] = {s.id: courtmod.stroke_movement(pos, tm, s.peak, strokes[k + 1].start if k + 1 < len(strokes) else None) for k, s in enumerate(strokes)}
            turns = courtmod.direction_changes(pos, tm.rate)
            step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
            extras["movement_summary"] = {
                "status": "ok", "validated": False, "distance_m": round(float(np.nansum(step[step < 1.5])), 1), "direction_changes": [round(float(tm.t[i]), 2) for i in turns],
                "note": "Position is the midpoint of the feet mapped onto the court plane from the detected lines. Unvalidated: no evaluation footage allowed the court to be located, so these numbers have not been checked.",
            }  # fmt: skip
        extras["court"] = info
    elif options.court:
        extras["court"] = {"status": "unable", "reason": "No frames were available to look for court lines."}
    hops = courtmod.find_hops(tm)
    hop_t = [round(float(tm.t[i]), 2) for i in hops]
    before = {}
    for s in strokes:
        prior = [t for t in hop_t if s.start - 1.2 <= t <= s.peak - 0.15]
        before[s.id] = round(s.peak - prior[-1], 2) if prior else None
    hop_info = {"times": hop_t, "before_stroke_s": before, "validated": False,
                "note": "Small two-footed hops found from the ankles and hips rising together. A split step is a hop timed to the opponent's hit; the opponent is not tracked and no footage has been labelled for this, so these are unvalidated observations."}  # fmt: skip

    segs = [dict(s) for s in (track.segments or [])]
    for s in segs:
        if s["status"] == "needs_confirmation":
            d = confirmations.get(str(s["id"]))
            s["decision"] = None if d is None else "confirmed" if d else "rejected"
    extras["tracking_events"] = {"cuts": [round(float(track.t[c]), 2) for c in aux.get("cuts") or [] if c < track.n], "segments": segs, "reacquire": options.reacquire}
    report = build_report(tm, strokes, measures, intake, quality, meta, extras)
    report["hops"] = hop_info
    report["pose_note"] = aux.get("pose_note")
    return report


def _overlay(track: PoseTrack, aux: dict) -> dict:
    out = track.overlay_payload()
    pts: dict[int, list[int]] = {}
    for t in aux.get("ball_tracks") or []:
        for i, (x, y) in zip(t["idx"], t["xy"]):
            pts[i] = [int(round(x / track.width * 1000)), int(round(y / track.height * 1000))]
    out["ball"] = [pts.get(i) for i in range(track.n)]
    out["seg"] = None if track.seg is None else [int(v) for v in track.seg]
    return out


def analyze(session_dir: Path, detector: Detector, *, meta: dict, quality: dict, candidates: dict, selected: int, intake: dict, max_fps: float, progress: Progress, options: Options | None = None,
            video: Path | None = None, confirmations: dict | None = None, object_model: Path | None = None) -> dict:  # fmt: skip
    options = options or Options()
    video = video or session_dir / VIDEO
    stride = max(1, int(round(meta["fps"] / max_fps)))
    progress("Estimating body pose", 0.02)
    frame_idx, detections = detector.detect_video(video, stride, lambda f: progress("Estimating body pose", 0.02 + 0.70 * f))
    vis = Visual()
    if options.ball or options.cuts or options.reacquire or options.court:
        progress("Looking for the ball, cuts and court lines", 0.74)
        vis = visual_pass(video, frame_idx, detections, want_ball=options.ball, want_appearance=options.reacquire, fps=meta["fps"])
    cuts = vis.cuts if options.cuts else []
    progress("Following the selected player", 0.82)
    person = candidates["people"][selected]
    track = track_player(frame_idx, detections, candidates["frame_idx"], tuple(person["centre"]), person["height_frac"], meta, cuts=cuts,
                         appearance=vis.appearance if options.reacquire else None, reacquire=options.reacquire)  # fmt: skip
    frac = float(track.masked(confirmations).tracked.mean())
    if frac < MIN_TRACKED and not any(s["status"] == "needs_confirmation" for s in track.segments or []):
        raise PipelineError(
            "tracking_poor",
            f"The selected player could only be followed in {frac:.0%} of the video, which is not enough to analyse.",
            "Use a clip where the player stays fully in view, is not blocked by others, and is reasonably large in the frame.",
        )
    pose_note = None
    if options.pose_mode == "cropped":
        if hasattr(detector, "detect_crops"):
            progress("Second pose pass on the player", 0.84)
            track, pose_note = refine_with_crops(detector, video, track, lambda f: progress("Second pose pass on the player", 0.84 + 0.06 * f))
        else:
            pose_note = "The cropped second pass is not available with this pose detector."
    aux: dict = {"cuts": [int(c) for c in cuts], "ball_tracks": [], "court": None, "pose_note": pose_note}
    if options.ball:
        aux["ball_tracks"] = ballmod.to_json(ballmod.link_tracks(vis.ball, meta["width"], meta["height"], set(cuts)))
    if options.court:
        progress("Looking for court lines", 0.91)
        c = courtmod.analyse(vis.samples, track, cuts)
        c["pos"] = [None if not np.isfinite(p).all() else [round(float(p[0]), 2), round(float(p[1]), 2)] for p in c["pos"]]
        aux["court"] = c
    track.save(session_dir / TRACK)
    (session_dir / AUX).write_text(json.dumps(aux, separators=(",", ":")))
    (session_dir / OVERLAY).write_text(json.dumps(_overlay(track, aux), separators=(",", ":")))
    progress("Finding strokes and writing the report", 0.95)
    return compose(track, aux, intake=intake, quality=quality, meta=meta, confirmations=confirmations, options=options)


def rebuild(load: Callable[[str], bytes | None], *, meta: dict, quality: dict, intake: dict, stroke_specs: list[dict], previous: list[dict],
            confirmations: dict | None = None, redetect: bool = False, options: Options | None = None) -> dict:  # fmt: skip
    """Recompute the report from saved pose data. ``load(name)`` returns a stored file's bytes.

    With ``redetect`` (after the user confirms or rejects a re-acquired stretch) strokes are detected again on the
    stretches now in play and the user's manual strokes in ``stroke_specs`` are added back. Otherwise ``stroke_specs``
    is the full edited stroke list.
    """
    options = options or Options()
    track = PoseTrack.load(io.BytesIO(load(TRACK)))
    raw = load(AUX)
    aux = json.loads(raw) if raw else {}
    tm = track.masked(confirmations)
    hand = intake.get("handedness", "right")
    prev = {s["id"]: s for s in previous}
    strokes: list[Stroke] = detect_strokes(tm, hand, intake.get("backhand_style", "two_handed")) if redetect else []
    for spec in stroke_specs:
        kind = spec.get("type")
        if kind not in (*STROKE_TYPES, "unclassified"):
            raise PipelineError("bad_stroke", f"Unknown stroke type: {kind}")
        start, end = float(spec["start"]), float(spec["end"])
        if not (0 <= start < end <= meta["duration_s"] + 0.5) or end - start > 6 or end - start < 0.2:
            raise PipelineError("bad_stroke", "A stroke must be between 0.2 and 6 seconds long and lie inside the video.")
        old = prev.get(spec.get("id") or "")
        if old and abs(old["start"] - start) < 0.01 and abs(old["end"] - end) < 0.01 and old["type"] == kind:
            strokes.append(Stroke(**{k: old.get(k, Stroke.__dataclass_fields__[k].default) for k in STROKE_FIELDS}))
        else:
            strokes.append(manual_stroke(tm, hand, start, end, kind, spec.get("id") if old else None))
    if redetect:  # a manual stroke wins over an automatic one that overlaps it
        manual = [s for s in strokes if s.source == "manual"]
        strokes = manual + [s for s in strokes if s.source != "manual" and not any(m.start <= s.peak <= m.end for m in manual)]
    strokes.sort(key=lambda s: s.start)
    return compose(track, aux, intake=intake, quality=quality, meta=meta, confirmations=confirmations, options=options, strokes=strokes)
