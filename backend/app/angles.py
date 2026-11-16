"""Joint angles over time, and comparison of the pose tracking with reference data.

Two kinds of reference are supported, and neither exists until a person supplies it:
  * hand-clicked joint positions on chosen frames (the in-app annotation tool), which
    check 2D landmark position and 2D image-plane angles;
  * a CSV of joint angles from a motion-capture system (format below), which checks the
    reported angles themselves, separately for the 2D image-plane method and the
    single-camera 3D estimate.
No accuracy figure is produced without one of these.

Motion-capture CSV: a header row, then one row per sample. Required column ``time_s``
(seconds from the start of the uploaded video; use --offset to shift). Any of:
knee_flexion_left, knee_flexion_right (degrees, 0 = straight), elbow_angle_left,
elbow_angle_right (degrees, 180 = straight), trunk_lean (degrees from vertical).
Lines starting with # are ignored.
"""
from __future__ import annotations

import csv
import io
import statistics

import numpy as np

from .pipeline.measure import knee_flexion, trunk_lean
from .pipeline.signals import LEFT, MIN_VIS, RIGHT, PoseTrack, angle_at

JOINTS = {"nose": 0, "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14, "left_wrist": 15, "right_wrist": 16,
          "left_hip": 23, "right_hip": 24, "left_knee": 25, "right_knee": 26, "left_ankle": 27, "right_ankle": 28}  # fmt: skip
SERIES = {
    "knee_flexion_left": ("Left knee bend", "0 = straight leg"), "knee_flexion_right": ("Right knee bend", "0 = straight leg"),
    "elbow_angle_left": ("Left elbow angle", "180 = straight arm"), "elbow_angle_right": ("Right elbow angle", "180 = straight arm"),
    "trunk_lean": ("Trunk tilt from vertical", "image plane only"),
}  # fmt: skip
METHODS = ("image_plane", "estimated_3d")


def _series(track: PoseTrack) -> dict[str, dict[str, np.ndarray]]:
    xy, w = track.xy_s, track.world_s
    out: dict[str, dict[str, np.ndarray]] = {}
    for name, side in (("left", LEFT), ("right", RIGHT)):
        k2, k3 = knee_flexion(track, side)
        out[f"knee_flexion_{name}"] = {"image_plane": k2, "estimated_3d": k3, "vis": track.vis[:, [side.hip, side.knee, side.ank]].min(axis=1)}
        out[f"elbow_angle_{name}"] = {"image_plane": angle_at(xy[:, side.sh], xy[:, side.el], xy[:, side.wr]), "estimated_3d": angle_at(w[:, side.sh], w[:, side.el], w[:, side.wr]),
                                      "vis": track.vis[:, [side.sh, side.el, side.wr]].min(axis=1)}  # fmt: skip
    out["trunk_lean"] = {"image_plane": trunk_lean(track), "vis": track.vis[:, [11, 12, 23, 24]].min(axis=1)}
    return out


def stroke_series(track: PoseTrack, start: float, end: float, pad: float = 0.3) -> dict:
    """Angle-over-time data for one stroke. A value is null (a gap in the plot) wherever a joint it needs was not tracked well enough."""
    sl = track.window(max(0.0, start - pad), end + pad)
    t = track.t[sl]
    r = lambda a: [None if not np.isfinite(v) else round(float(v), 1) for v in a]  # noqa: E731
    series = []
    for key, s in _series(track).items():
        vis = s["vis"][sl]
        entry = {"key": key, "label": SERIES[key][0], "note": SERIES[key][1], "unit": "deg", "visibility": [round(float(v), 2) for v in vis],
                 "tracked_frac": round(float((vis >= MIN_VIS).mean()), 2) if vis.size else 0.0, "methods": {}}  # fmt: skip
        for m in METHODS:
            if m in s:
                entry["methods"][m] = r(s[m][sl])
        series.append(entry)
    return {
        "t": [round(float(x), 3) for x in t], "min_visibility": MIN_VIS, "series": series,
        "method_notes": {"image_plane": "Angle measured in the 2D picture. Changes with camera position; a limb pointing at the camera looks straighter than it is.",
                         "estimated_3d": "From the pose model's single-camera 3D estimate. A model guess at depth, not motion capture; its error in tennis postures is unknown."},
        "gap_note": f"Gaps mean a joint needed for that angle was hidden, blurred or below the visibility cut-off ({MIN_VIS}); nothing is drawn rather than a guess.",
    }  # fmt: skip


# --- hand-clicked joint annotations ---------------------------------------------------------


def compare_annotations(track: PoseTrack, frames: list[dict]) -> dict:
    """frames: [{"frame": int, "joints": {name: [x, y] normalised 0-1}}]. Position error is in torso lengths and pixels."""
    per: dict[str, list[tuple[float, float]]] = {}
    angle_err: dict[str, list[float]] = {}
    scale = np.array([track.width, track.height], float)
    used = 0
    for f in frames:
        i = int(np.argmin(np.abs(track.frame_idx - int(f["frame"]))))
        if abs(int(track.frame_idx[i]) - int(f["frame"])) > track.stride or not track.tracked[i]:
            continue
        used += 1
        tl = track.torso_len[i]
        pts = {k: np.asarray(v, float) * scale for k, v in f["joints"].items() if k in JOINTS and v is not None}
        for name, p in pts.items():
            j = JOINTS[name]
            if track.vis[i, j] < MIN_VIS or not np.isfinite(tl):
                continue
            d = float(np.linalg.norm(track.xy[i, j] - p))
            per.setdefault(name, []).append((d, d / tl))
        for key, (a, b, c) in {"knee_flexion_left": ("left_hip", "left_knee", "left_ankle"), "knee_flexion_right": ("right_hip", "right_knee", "right_ankle"),
                               "elbow_angle_left": ("left_shoulder", "left_elbow", "left_wrist"), "elbow_angle_right": ("right_shoulder", "right_elbow", "right_wrist")}.items():  # fmt: skip
            if all(k in pts for k in (a, b, c)) and min(track.vis[i, JOINTS[k]] for k in (a, b, c)) >= MIN_VIS:
                ref = float(angle_at(pts[a], pts[b], pts[c]))
                est = float(angle_at(track.xy[i, JOINTS[a]], track.xy[i, JOINTS[b]], track.xy[i, JOINTS[c]]))
                if key.startswith("knee"):
                    ref, est = 180 - ref, 180 - est
                angle_err.setdefault(key, []).append(est - ref)
    med = statistics.median
    return {
        "frames_annotated": len(frames), "frames_compared": used,
        "position_error_by_joint": {k: {"n": len(v), "median_px": round(med(x[0] for x in v), 1), "median_torso_lengths": round(med(x[1] for x in v), 3),
                                        "max_torso_lengths": round(max(x[1] for x in v), 3)} for k, v in sorted(per.items())},
        "image_plane_angle_error": {k: {"n": len(v), "mean_signed_deg": round(statistics.fmean(v), 1), "median_abs_deg": round(med(abs(x) for x in v), 1)} for k, v in sorted(angle_err.items())},
        "note": "Reference = joint centres clicked by a person on the video frame. This checks where the model puts the joints in the picture and the 2D angles that follow; "
                "it cannot check the single-camera 3D estimates. Click accuracy and the annotator's judgement limit it.",
    }  # fmt: skip


# --- motion-capture reference ---------------------------------------------------------------


def parse_mocap_csv(text: str) -> dict[str, list[tuple[float, float]]]:
    rows = [r for r in csv.reader(io.StringIO(text)) if r and not r[0].lstrip().startswith("#")]
    if not rows:
        raise ValueError("The CSV is empty.")
    header = [h.strip().lower() for h in rows[0]]
    if "time_s" not in header:
        raise ValueError("The CSV needs a 'time_s' column.")
    cols = [h for h in header if h in SERIES]
    if not cols:
        raise ValueError(f"No recognised angle columns. Use any of: {', '.join(SERIES)}.")
    out: dict[str, list[tuple[float, float]]] = {c: [] for c in cols}
    ti = header.index("time_s")
    for n, r in enumerate(rows[1:], start=2):
        try:
            t = float(r[ti])
        except (ValueError, IndexError) as e:
            raise ValueError(f"Row {n}: time_s is not a number.") from e
        for c in cols:
            v = r[header.index(c)].strip() if header.index(c) < len(r) else ""
            if v:
                try:
                    out[c].append((t, float(v)))
                except ValueError as e:
                    raise ValueError(f"Row {n}: {c} is not a number.") from e
    return out


def compare_mocap(track: PoseTrack, ref: dict[str, list[tuple[float, float]]], offset_s: float = 0.0) -> dict:
    """Error of each angle against the reference, per method. Only samples where the joints were tracked are compared."""
    series = _series(track)
    out = {}
    for key, samples in ref.items():
        s = series[key]
        res = {}
        for m in METHODS:
            if m not in s:
                continue
            errs = []
            for t, v in samples:
                tt = t + offset_s
                if not track.t[0] <= tt <= track.t[-1]:
                    continue
                i = track.index_at(tt)
                if s["vis"][i] >= MIN_VIS and np.isfinite(s[m][i]):
                    errs.append(float(s[m][i]) - v)
            if errs:
                a = np.abs(errs)
                res[m] = {"n": len(errs), "mean_signed_deg": round(float(np.mean(errs)), 1), "mean_abs_deg": round(float(a.mean()), 1), "rmse_deg": round(float(np.sqrt(np.mean(np.square(errs)))), 1),
                          "p90_abs_deg": round(float(np.percentile(a, 90)), 1)}  # fmt: skip
            else:
                res[m] = {"n": 0}
        out[key] = {"reference_samples": len(samples), "methods": res}
    return {"offset_s": offset_s, "angles": out, "note": "Reference = motion-capture angles supplied as CSV. Errors apply to this camera position only; report them by camera angle before generalising."}
