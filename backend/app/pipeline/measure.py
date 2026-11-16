"""Per-stroke measurements.

Each measurement states its method:
  image_plane   angle or distance measured in the 2D picture; changes with camera angle
  estimated_3d  from MediaPipe single-camera "world" landmarks; a model estimate, not
                validated 3D motion capture
  relative      body-size-normalised image quantity; comparable within one recording
A measurement that cannot be supported by the tracking is returned with
assessable=False and a reason instead of a number.
"""
from __future__ import annotations

import numpy as np

from .segment import Stroke
from .signals import MIN_VIS, NOSE, PoseTrack, Side, angle_at, sides

MIN_TRACKED_FRAC = 0.8
FRONTAL_MAX_AZ = 30.0  # pelvis within 30 deg of facing the camera/away => frontal-plane checks allowed


def _m(key, label, value, unit, method, *, time=None, vis=None, reason="", **extra) -> dict:
    ok = value is not None and np.isfinite(value)
    return {
        "key": key,
        "label": label,
        "value": round(float(value), 2) if ok else None,
        "unit": unit,
        "method": method,
        "time": round(float(time), 3) if time is not None else None,
        "visibility": round(float(vis), 2) if vis is not None else None,
        "assessable": bool(ok),
        "reason": "" if ok else (reason or "The landmarks needed were not visible clearly enough."),
        "extra": extra,
    }


def _nanarg(a: np.ndarray, fn) -> int | None:
    return int(fn(a)) if np.isfinite(a).any() else None


def knee_flexion(track: PoseTrack, leg: Side) -> tuple[np.ndarray, np.ndarray]:
    """Knee flexion (0 = straight leg) per frame: (image-plane, estimated-3D)."""
    xy, w = track.xy_s, track.world_s
    return 180 - angle_at(xy[:, leg.hip], xy[:, leg.knee], xy[:, leg.ank]), 180 - angle_at(w[:, leg.hip], w[:, leg.knee], w[:, leg.ank])


def trunk_lean(track: PoseTrack) -> np.ndarray:
    """Image-plane tilt of the hip-to-shoulder line from vertical, degrees."""
    v = track.mid_sh - track.mid_hip
    return np.degrees(np.arctan2(np.abs(v[:, 0]), -v[:, 1]))


def _shoulder_lateral(track: PoseTrack, dom: Side, non: Side) -> np.ndarray:
    """Hitting hand along the shoulder line, in shoulder widths from mid-chest; + = hitting-arm side."""
    w = track.world_s
    axis = w[:, dom.sh] - w[:, non.sh]
    mid = (w[:, dom.sh] + w[:, non.sh]) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sum((w[:, dom.wr] - mid) * axis, axis=1) / np.sum(axis * axis, axis=1)


def _knee_alignment(track: PoseTrack, leg: Side, other: Side, i: int) -> float:
    """Sideways offset of the knee from the hip-ankle line / leg length; + = toward the other leg."""
    xy = track.xy_s
    H, K, A, O = xy[i, leg.hip], xy[i, leg.knee], xy[i, leg.ank], xy[i, other.hip]
    line = A - H
    L = np.linalg.norm(line)
    if not np.isfinite(L) or L < 1e-6 or not np.isfinite(K).all() or not np.isfinite(O).all():
        return float("nan")
    cross = lambda p: float(line[0] * (p - H)[1] - line[1] * (p - H)[0])  # noqa: E731
    dk, do = cross(K), cross(O)
    return abs(dk) / L * (1.0 if dk * do > 0 else -1.0)


def measure_stroke(track: PoseTrack, stroke: Stroke, handedness: str, with_contact: bool = False) -> dict:
    dom, non = sides(handedness)
    r = track.rate
    sl = track.window(stroke.start, stroke.end)
    p = track.index_at(stroke.pose_time if stroke.pose_time is not None else stroke.peak)  # pose measurements stay anchored to the pose-based timestamp
    is_serve = stroke.type == "serve"
    legs = [dom.hip, dom.knee, dom.ank, non.hip, non.knee, non.ank]
    arm = [dom.sh, dom.el, dom.wr]
    tracked_frac = float(track.tracked[sl].mean()) if track.tracked[sl].size else 0.0
    quality = {
        "tracked_frac": round(tracked_frac, 2),
        "arm_visibility": round(track.mean_vis(arm, sl), 2),
        "leg_visibility": round(track.mean_vis(legs, sl), 2),
        "pelvis_azimuth_median": None,
        "assessable": tracked_frac >= MIN_TRACKED_FRAC,
        "reason": "" if tracked_frac >= MIN_TRACKED_FRAC else f"The player was tracked in only {tracked_frac:.0%} of this stroke's frames.",
    }
    az = track.pelvis_azimuth
    if np.isfinite(az[sl]).any():
        quality["pelvis_azimuth_median"] = round(float(np.nanmedian(az[sl])), 0)
    out: dict[str, dict] = {}
    if not quality["assessable"]:
        return {"quality": quality, "values": out}

    def clip(i: int) -> int:
        return int(np.clip(i, 0, track.n - 1))

    def add(m: dict) -> None:
        out[m["key"]] = m

    # --- peak hitting-hand speed --------------------------------------------------------
    _, speed = track.hand_motion(dom.wr)
    # serves are timed at the highest hand position, where the hand is momentarily slow: take the fastest point around it
    near = slice(clip(p - int((0.4 if is_serve else 0.1) * r)), clip(p + int((0.4 if is_serve else 0.1) * r)) + 1)
    if np.isfinite(speed[near]).any():
        speed = speed.copy()
        speed[p] = np.nanmax(speed[near])
    add(_m("peak_hand_speed", "Peak hitting-hand speed (relative to the torso)", speed[p] if np.isfinite(speed[p]) else np.nanmax(speed[sl]) if np.isfinite(speed[sl]).any() else None,
           "torso lengths/s", "relative", time=track.t[p], vis=track.vis[p, dom.wr]))  # fmt: skip

    # --- knee bend while loading --------------------------------------------------------
    load = slice(clip(p - int(1.0 * r)), clip(p - int(0.1 * r)) + 1) if is_serve else slice(max(sl.start, clip(p - int(0.6 * r))), p + 1)
    for name, leg in (("dom", dom), ("non", non)):
        f2, f3 = knee_flexion(track, leg)
        a2, a3 = _nanarg(f2[load], np.nanargmax), _nanarg(f3[load], np.nanargmax)
        v2 = float(f2[load][a2]) if a2 is not None else None
        v3 = float(f3[load][a3]) if a3 is not None else None
        both = [v for v in (v2, v3) if v is not None]
        i_best = load.start + (a3 if a3 is not None else a2 or 0)
        label = "Deepest knee bend, " + ("hitting-arm side leg" if name == "dom" else "other leg")
        add(_m(f"knee_bend_{name}", label, max(both) if both else None, "deg", "image_plane+estimated_3d", time=track.t[i_best],
               vis=track.mean_vis([leg.hip, leg.knee, leg.ank], load), image_plane=None if v2 is None else round(v2), estimated_3d=None if v3 is None else round(v3),
               reason="The hip, knee or ankle of this leg was hidden or unclear during the loading phase."))  # fmt: skip

    # --- elbow angle at peak hand speed -------------------------------------------------
    pk = slice(clip(p - 1), clip(p + 1) + 1)
    e2 = angle_at(track.xy_s[pk, dom.sh], track.xy_s[pk, dom.el], track.xy_s[pk, dom.wr])
    e3 = angle_at(track.world_s[pk, dom.sh], track.world_s[pk, dom.el], track.world_s[pk, dom.wr])
    e2v = float(np.nanmean(e2)) if np.isfinite(e2).any() else None
    e3v = float(np.nanmean(e3)) if np.isfinite(e3).any() else None
    add(_m("elbow_angle_peak", "Elbow angle at peak hand speed (180 = straight)", e3v if e3v is not None else e2v, "deg", "estimated_3d", time=track.t[p],
           vis=track.mean_vis(arm, pk), image_plane=None if e2v is None else round(e2v), estimated_3d=None if e3v is None else round(e3v),
           reason="The hitting arm was hidden or blurred at peak hand speed."))  # fmt: skip

    # --- trunk tilt ---------------------------------------------------------------------
    lean = trunk_lean(track)
    tw = slice(clip(p - int(0.4 * r)), p + 1) if is_serve else pk
    li = _nanarg(lean[tw], np.nanargmax)
    if li is not None:
        i = tw.start + li
        add(_m("trunk_lean", "Trunk tilt from vertical" + (" (largest before peak)" if is_serve else " at peak hand speed"), lean[i], "deg", "image_plane",
               time=track.t[i], vis=track.mean_vis([dom.sh, non.sh, dom.hip, non.hip], tw), pelvis_azimuth=None if not np.isfinite(az[i]) else round(float(az[i]))))  # fmt: skip
    else:
        add(_m("trunk_lean", "Trunk tilt from vertical", None, "deg", "image_plane", reason="Shoulders or hips were not visible."))

    # --- serve reach --------------------------------------------------------------------
    if is_serve:
        rw = slice(clip(p - int(0.1 * r)), clip(p + int(0.1 * r)) + 1)
        with np.errstate(invalid="ignore"):
            reach = (track.xy_s[rw, NOSE, 1] - track.xy_s[rw, dom.wr, 1]) / track.torso_len[rw]
        ri = _nanarg(reach, np.nanargmax)
        add(_m("reach_height", "Hitting hand height above nose near peak speed", reach[ri] if ri is not None else None, "torso lengths", "relative",
               time=track.t[rw.start + (ri or 0)], vis=track.vis[p, dom.wr]))  # fmt: skip

    # --- follow-through (groundstrokes) -------------------------------------------------
    if stroke.type in ("forehand", "backhand"):
        ft = slice(p, sl.stop)
        lat = _shoulder_lateral(track, dom, non)[ft]
        fi = _nanarg(lat, np.nanargmin if stroke.type == "forehand" else np.nanargmax)
        enough = (sl.stop - p) >= int(0.2 * r)
        if fi is not None and enough:
            i = p + fi
            with np.errstate(invalid="ignore"):
                height = (track.xy_s[i, dom.sh, 1] - track.xy_s[i, dom.wr, 1]) / track.torso_len[i]
            # express as travel past mid-chest toward the finish side, in shoulder widths
            travel = -lat[fi] if stroke.type == "forehand" else lat[fi]
            add(_m("follow_through_travel", "How far the hitting hand finishes past mid-chest", travel, "shoulder widths", "estimated_3d", time=track.t[i],
                   vis=track.mean_vis(arm + [non.sh], ft), finish_height_vs_shoulder=None if not np.isfinite(height) else round(float(height), 2)))  # fmt: skip
        else:
            add(_m("follow_through_travel", "How far the hitting hand finishes past mid-chest", None, "shoulder widths", "estimated_3d",
                   reason="The hitting arm was not visible through the finish, or the clip ends too soon after the swing."))  # fmt: skip

    # --- knee position relative to the foot line (frontal view only) ---------------------
    aw = slice(clip(p + int(0.1 * r)), clip(p + int(0.6 * r)) + 1) if is_serve else load
    phase = "landing" if is_serve else "loading"
    for name, leg, other in (("dom", dom, non), ("non", non, dom)):
        key, label = f"knee_alignment_{name}", f"Knee offset from the hip-ankle line during {phase}, " + ("hitting-arm side leg" if name == "dom" else "other leg")
        _, f3 = knee_flexion(track, leg)
        ai = _nanarg(f3[aw], np.nanargmax)
        if ai is None:
            add(_m(key, label, None, "leg lengths", "image_plane", reason="This leg was not visible clearly enough."))
            continue
        i = aw.start + ai
        if not np.isfinite(az[i]) or az[i] > FRONTAL_MAX_AZ:
            add(_m(key, label, None, "leg lengths", "image_plane", reason="Needs a view from in front of or behind the player; at this moment the hips were turned too far from the camera."))
            continue
        if f3[i] < 20:
            add(_m(key, label, None, "leg lengths", "image_plane", reason="The knee stayed nearly straight, so sideways knee position is not meaningful here."))
            continue
        vis = float(track.vis[i, [leg.hip, leg.knee, leg.ank]].min())
        off = _knee_alignment(track, leg, other, i) if vis >= MIN_VIS else float("nan")
        add(_m(key, label, off, "leg lengths", "image_plane", time=track.t[i], vis=vis, pelvis_azimuth=round(float(az[i])), knee_flexion_estimated_3d=round(float(f3[i]))))

    # --- contact point relative to the body (needs ball contact) --------------------------
    if with_contact:
        no_ball = "The ball was not tracked well enough around this stroke to find contact, so the contact point is not reported."
        specs = (("contact_height", "Contact height above the hips", "torso lengths"), ("contact_distance", "Distance from the hips to the ball at contact", "torso lengths"),
                 ("contact_in_front", "Contact point ahead of the hips (side-on view)", "torso lengths"))  # fmt: skip
        ci = track.index_at(stroke.peak)
        hip, tl = track.mid_hip[ci], track.torso_len[ci]
        if stroke.contact is None or not np.isfinite(hip).all() or not np.isfinite(tl):
            for key, label, unit in specs:
                add(_m(key, label, None, unit, "image_plane", reason=no_ball if stroke.contact is None else "The hips were not visible at contact."))
        else:
            b = np.asarray(stroke.contact["xy"], float)
            v = min(float(stroke.contact["confidence"]), track.mean_vis([dom.hip, non.hip, dom.sh, non.sh], slice(ci, ci + 1)))
            add(_m("contact_height", specs[0][1], (hip[1] - b[1]) / tl, specs[0][2], "image_plane", time=stroke.peak, vis=v, basis=None))
            add(_m("contact_distance", specs[1][1], float(np.linalg.norm(b - hip)) / tl, specs[1][2], "image_plane", time=stroke.peak, vis=v))
            wv = track.xy_s[clip(ci + 1), dom.wr, 0] - track.xy_s[clip(ci - 1), dom.wr, 0]
            if not np.isfinite(az[ci]) or az[ci] < 55:
                add(_m("contact_in_front", specs[2][1], None, specs[2][2], "image_plane", reason="How far in front of the body contact was made can only be seen from the side; here the hips were not side-on to the camera."))
            elif not np.isfinite(wv) or abs(wv) < 1e-6:
                add(_m("contact_in_front", specs[2][1], None, specs[2][2], "image_plane", reason="The hitting hand was not visible at contact, so the hitting direction is unknown."))
            else:
                add(_m("contact_in_front", specs[2][1], (b[0] - hip[0]) * np.sign(wv) / tl, specs[2][2], "image_plane", time=stroke.peak, vis=v, pelvis_azimuth=round(float(az[ci]))))

    # --- shoulder turn range ------------------------------------------------------------
    sh = track.world_s[sl, dom.sh] - track.world_s[sl, non.sh]
    ang = np.arctan2(sh[:, 2], sh[:, 0])
    good = np.isfinite(ang)
    rng = float(np.degrees(np.ptp(np.unwrap(ang[good])))) if good.sum() >= 5 else None
    add(_m("shoulder_turn_range", "Shoulder-line rotation over the stroke", rng, "deg", "estimated_3d", vis=track.mean_vis([dom.sh, non.sh], sl),
           reason="Shoulders were not visible enough."))  # fmt: skip

    return {"quality": quality, "values": out}
