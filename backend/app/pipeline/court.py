"""Court lines -> camera-to-court mapping -> where the player is on the court.

Method (after Farin et al.'s line-based court calibration): find bright, thin line pixels;
group them into straight lines and into two direction families; try assigning pairs of
detected lines to pairs of real court lines; each assignment gives a homography, which is
scored by how many of ALL the court's painted lines then land on line pixels. A mapping is
accepted only when at least five distinct court lines are supported and the best
assignment clearly beats differently-placed alternatives. Two or three visible lines fit
many placements equally well, so that case is reported as "unable to assess" rather than
guessed.

All positions are on the court plane (2D, metres) and assume the feet are on the ground.
The cut-offs are developer heuristics. Court dimensions are the standard ITF ones.
"""
from __future__ import annotations

from itertools import combinations

import cv2
import numpy as np

from .signals import L_ANK, L_HEEL, R_ANK, R_HEEL, PoseTrack, rolling_nanmedian

W_DOUBLES, W_ALLEY, LENGTH, SERVICE = 10.97, 1.372, 23.77, 5.485
NET_Y = LENGTH / 2
ACROSS = [0.0, SERVICE, LENGTH - SERVICE, LENGTH]  # baselines and service lines (y = const)
ALONG = [0.0, W_ALLEY, W_DOUBLES / 2, W_DOUBLES - W_ALLEY, W_DOUBLES]  # sidelines and centre service line (x = const)
# painted segments of the full court: ((x0, y0), (x1, y1), name)
MODEL_LINES = (
    [((0, y), (W_DOUBLES, y), f"baseline_{k}") for k, y in (("near", 0.0), ("far", LENGTH))]
    + [((W_ALLEY, y), (W_DOUBLES - W_ALLEY, y), f"service_{k}") for k, y in (("near", SERVICE), ("far", LENGTH - SERVICE))]
    + [((x, 0), (x, LENGTH), f"sideline_{k}") for k, x in (("doubles_l", 0.0), ("singles_l", W_ALLEY), ("singles_r", W_DOUBLES - W_ALLEY), ("doubles_r", W_DOUBLES))]
    + [((W_DOUBLES / 2, SERVICE), (W_DOUBLES / 2, LENGTH - SERVICE), "centre_service")]
)
MIN_SUPPORTED_LINES = 5
MIN_LINE_SUPPORT = 0.6
MIN_MARGIN = 0.06
MAX_LINES_PER_FAMILY = 5


def line_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Pixels that look like painted court lines: brighter than their surroundings, thin and not strongly coloured."""
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    k = max(5, int(round(w * 0.02)) | 1)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    m = ((tophat > 35) & (gray > 140) & (hsv[..., 1] < 90)).astype(np.uint8) * 255
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def _normal_form(x0, y0, x1, y1) -> tuple[float, float, float]:
    a, b = y1 - y0, x0 - x1
    n = np.hypot(a, b)
    return a / n, b / n, -(a * x0 + b * y0) / n


def detect_lines(mask: np.ndarray) -> list[dict]:
    """Merge Hough segments into lines. Each: {"abc": normalised line, "angle": radians 0-pi, "length": supported px}."""
    h, w = mask.shape
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=int(0.06 * w), minLineLength=int(0.10 * w), maxLineGap=int(0.03 * w))
    if segs is None:
        return []
    lines: list[dict] = []
    for x0, y0, x1, y1 in sorted(np.asarray(segs).reshape(-1, 4).tolist(), key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1])):
        a, b, c = _normal_form(x0, y0, x1, y1)
        ang = np.arctan2(y1 - y0, x1 - x0) % np.pi
        length = float(np.hypot(x1 - x0, y1 - y0))
        for ln in lines:
            la, lb, lc = ln["abc"]
            d_ang = abs(ang - ln["angle"])
            d_ang = min(d_ang, np.pi - d_ang)
            if d_ang < np.radians(3) and max(abs(la * x0 + lb * y0 + lc), abs(la * x1 + lb * y1 + lc)) < 0.012 * w:
                ln["length"] += length
                ln["pts"] += [(x0, y0), (x1, y1)]
                break
        else:
            lines.append({"abc": (a, b, c), "angle": ang, "length": length, "pts": [(x0, y0), (x1, y1)]})
    for ln in lines:  # refit through all supporting segment ends
        pts = np.asarray(ln["pts"], float)
        vx, vy, px, py = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        ln["abc"] = _normal_form(px, py, px + vx, py + vy)
        ln["angle"] = float(np.arctan2(vy, vx) % np.pi)
    return sorted(lines, key=lambda ln: -ln["length"])


def _families(lines: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split lines into two direction groups (2-means on the doubled angle)."""
    if len(lines) < 4:
        return [], []
    v = np.asarray([[np.cos(2 * ln["angle"]), np.sin(2 * ln["angle"])] for ln in lines])
    wts = np.asarray([ln["length"] for ln in lines])
    c = np.stack([v[0], v[int(np.argmin(v @ v[0]))]])
    for _ in range(10):
        lab = np.argmax(v @ c.T, axis=1)
        for k in range(2):
            if (lab == k).any():
                m = (v[lab == k] * wts[lab == k, None]).sum(0)
                c[k] = m / max(np.linalg.norm(m), 1e-9)
    if c[0] @ c[1] > np.cos(np.radians(2 * 12)):  # the two groups must differ by at least ~12 degrees
        return [], []
    fam = [[ln for ln, k in zip(lines, lab) if k == j][:MAX_LINES_PER_FAMILY] for j in range(2)]
    return fam[0], fam[1]


def _intersect(l1, l2) -> np.ndarray | None:
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:
        return None
    return np.array([(b1 * c2 - b2 * c1) / det, (a2 * c1 - a1 * c2) / det])


def _model_samples(step: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    pts, ids = [], []
    for k, ((x0, y0), (x1, y1), _) in enumerate(MODEL_LINES):
        n = max(int(np.hypot(x1 - x0, y1 - y0) / step), 2)
        t = np.linspace(0, 1, n)
        pts.append(np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t], 1))
        ids.append(np.full(n, k))
    return np.concatenate(pts), np.concatenate(ids)


_SAMPLES = _model_samples()


def _project(Hm: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ Hm.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[:, :2] / p[:, 2:3]


def _score(Hm: np.ndarray, dist: np.ndarray) -> tuple[float, int, list[int]]:
    """(share of in-frame model-line samples that lie on line pixels, number of model lines supported, their indices)."""
    h, w = dist.shape
    pts, ids = _SAMPLES
    p = np.c_[pts, np.ones(len(pts))] @ Hm.T
    front = p[:, 2] > 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = p[:, :2] / p[:, 2:3]
    ok = front & (xy[:, 0] >= 2) & (xy[:, 0] < w - 2) & (xy[:, 1] >= 2) & (xy[:, 1] < h - 2)
    if ok.sum() < 40:
        return 0.0, 0, []
    xi = xy[ok].astype(int)
    hit = dist[xi[:, 1], xi[:, 0]] <= 2.5
    supported = []
    for k in range(len(MODEL_LINES)):
        sel = ids[ok] == k
        if sel.sum() >= 8 and hit[sel].mean() >= MIN_LINE_SUPPORT:
            supported.append(k)
    return float(hit.mean()), len(supported), supported


def fit_court(frame_bgr: np.ndarray) -> dict:
    """Try to find the court in one frame. Returns {"found": bool, "reason", "H" (court metres -> image px), "score", "lines_supported"}."""
    h, w = frame_bgr.shape[:2]
    mask = line_mask(frame_bgr)
    lines = detect_lines(mask)
    fa, fb = _families(lines)
    if len(fa) < 2 or len(fb) < 2:
        return {"found": False, "reason": "Fewer than two court lines were found in each direction, so the camera's position relative to the court cannot be fixed.", "n_lines": len(lines)}
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    best: tuple[float, int, np.ndarray, list[int]] | None = None
    runner = 0.0
    for across_f, along_f in ((fa, fb), (fb, fa)):
        for l1, l2 in combinations(across_f, 2):
            for l3, l4 in combinations(along_f, 2):
                corners = [_intersect(a["abc"], b["abc"]) for a in (l1, l2) for b in (l3, l4)]
                if any(c is None for c in corners):
                    continue
                img = np.float32(corners)
                img_area = cv2.contourArea(img[[0, 1, 3, 2]], oriented=True)
                if abs(img_area) < 0.005 * w * h:
                    continue
                for y1, y2 in combinations(ACROSS, 2):
                    for x1, x2 in combinations(ALONG, 2):
                        for ys in ((y1, y2), (y2, y1)):
                            for xs in ((x1, x2), (x2, x1)):
                                model = np.float32([(x, y) for y in ys for x in xs])
                                if cv2.contourArea(model[[0, 1, 3, 2]], oriented=True) * img_area > 0:
                                    continue  # a camera above the ground sees the court turned over relative to the plan (image y runs downward); the other sign is a mirror image
                                Hm = cv2.getPerspectiveTransform(model, img)
                                sc, n_sup, sup = _score(Hm, dist)
                                if n_sup < 4:
                                    continue
                                key = sc + 0.03 * n_sup
                                if best is None or key > best[0]:
                                    if best is not None and not _same_mapping(best[2], Hm, w, h):
                                        runner = max(runner, best[0])
                                    best = (key, n_sup, Hm, sup)
                                elif not _same_mapping(best[2], Hm, w, h):
                                    runner = max(runner, key)
    if best is None or best[1] < MIN_SUPPORTED_LINES:
        return {"found": False, "reason": f"Only {best[1] if best else len(lines)} court lines could be matched; at least {MIN_SUPPORTED_LINES} are needed to place the court without guessing.", "n_lines": len(lines)}
    if best[0] - runner < MIN_MARGIN:
        return {"found": False, "reason": "The visible lines fit more than one placement of the court equally well, so position on court is not reported.", "n_lines": len(lines)}
    Hm = best[2]
    under = _project(np.linalg.inv(Hm), np.float32([[w / 2, h * 0.95]]))[0]
    if np.isfinite(under).all() and under[1] > NET_Y:  # the court looks the same from either end: call the end nearest the bottom of the picture "near"
        Hm = Hm @ _HALF_TURN
    best = (best[0], best[1], Hm, best[3])
    return {"found": True, "reason": "", "H": best[2], "score": round(best[0] - 0.03 * best[1], 2), "lines_supported": [MODEL_LINES[k][2] for k in best[3]], "n_lines": len(lines)}


_HALF_TURN = np.array([[-1.0, 0, W_DOUBLES], [0, -1.0, LENGTH], [0, 0, 1.0]])


def _same_mapping(H1: np.ndarray, H2: np.ndarray, w: int, h: int) -> bool:
    """Do two homographies put the image centre and lower-centre at (nearly) the same court position?"""
    probes = np.float32([[w / 2, h / 2], [w / 2, h * 0.85], [w * 0.25, h * 0.7], [w * 0.75, h * 0.7]])
    try:
        a = _project(np.linalg.inv(H1), probes)
        b = _project(np.linalg.inv(H2), probes)
    except np.linalg.LinAlgError:
        return False
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return False
    turned = np.array([W_DOUBLES, LENGTH]) - b  # the court is unchanged by a half turn, so that counts as the same placement
    return bool(min(np.linalg.norm(a - b, axis=1).max(), np.linalg.norm(a - turned, axis=1).max()) < 1.0)


def camera_position(Hm: np.ndarray, w: int, h: int) -> np.ndarray | None:
    """Camera centre in court coordinates (x, y, height) from the homography, assuming square pixels and a centred principal point."""
    T = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1.0]])
    G = T @ Hm
    h1, h2, h3 = G[:, 0], G[:, 1], G[:, 2]
    den = h1[2] * h2[2]
    if abs(den) < 1e-12:
        return None
    f2 = -(h1[0] * h2[0] + h1[1] * h2[1]) / den
    if not np.isfinite(f2) or f2 <= 0:
        return None
    Kinv = np.diag([1 / np.sqrt(f2), 1 / np.sqrt(f2), 1.0])
    r1, r2, t = Kinv @ h1, Kinv @ h2, Kinv @ h3
    lam = (np.linalg.norm(r1) + np.linalg.norm(r2)) / 2
    r1, r2, t = r1 / lam, r2 / lam, t / lam
    R = np.stack([r1, r2, np.cross(r1, r2)], 1)
    if t[2] < 0:
        R, t = R * [-1, -1, 1], -t
    C = -R.T @ t
    C[2] = abs(C[2])
    return C if np.isfinite(C).all() else None


def view_from_court(cam: np.ndarray, player_xy: np.ndarray) -> str | None:
    """Where the camera is relative to a player who faces the net: behind / in front / to the side."""
    facing = np.array([0.0, 1.0 if player_xy[1] < NET_Y else -1.0])
    to_cam = cam[:2] - player_xy
    n = np.linalg.norm(to_cam)
    if n < 0.5:
        return None
    ang = np.degrees(np.arccos(np.clip(to_cam @ facing / n, -1, 1)))
    return "mostly from in front of or behind the player" if ang < 40 or ang > 140 else "mostly side-on" if 65 < ang < 115 else "at an angle, or the player turns a lot"


def analyse(samples: dict[int, np.ndarray], track: PoseTrack, cuts: list[int] | None = None) -> dict:
    """Fit the court on the sampled frames and place the player's feet on it.

    Returns {"status": "ok" | "unable", "reason", "fits" (per sample), "pos": (T, 2) court metres or NaN, "camera_view", ...}.
    A frame's position uses the nearest successful fit in the same shot within 0.75 s, so a moving camera is handled
    only as far as fits keep succeeding.
    """
    T = track.n
    pos = np.full((T, 2), np.nan)
    fits: dict[int, dict] = {}
    reasons: list[str] = []
    for i, frame in sorted(samples.items()):
        f = fit_court(frame)
        if f["found"]:
            fits[i] = f
        else:
            reasons.append(f["reason"])
    if not fits:
        reason = max(set(reasons), key=reasons.count) if reasons else "No frames were available to look for court lines."
        return {"status": "unable", "reason": reason, "frames_tried": len(samples), "frames_fitted": 0, "pos": pos, "camera_view": None}
    feet = np.nanmean(track.xy_s[:, [L_ANK, R_ANK, L_HEEL, R_HEEL]], axis=1)
    feet_ok = (track.vis[:, [L_ANK, R_ANK]] >= 0.5).all(axis=1)
    keys = np.asarray(sorted(fits))
    shot = np.searchsorted(np.asarray(sorted(cuts or [])), np.arange(T), side="right")
    max_gap = 0.75 * track.rate
    views = []
    for i in range(T):
        if not feet_ok[i] or not np.isfinite(feet[i]).all():
            continue
        k = keys[int(np.argmin(np.abs(keys - i)))]
        if abs(k - i) > max_gap or shot[k] != shot[i]:
            continue
        p = _project(np.linalg.inv(fits[k]["H"]), feet[i][None])[0]
        if np.isfinite(p).all() and -8 <= p[0] <= W_DOUBLES + 8 and -10 <= p[1] <= LENGTH + 10:
            pos[i] = p
    for k in keys:
        cam = camera_position(fits[k]["H"], track.width, track.height)
        near = pos[max(0, k - 3) : k + 4]
        if cam is not None and np.isfinite(near).any():
            v = view_from_court(cam, np.nanmean(near, axis=0))
            if v:
                views.append(v)
    placed = float(np.isfinite(pos[:, 0]).mean())
    if placed < 0.2:
        return {"status": "unable", "reason": "Court lines were found in a few frames only, or the player's feet were not visible, so position on court is not reported.",
                "frames_tried": len(samples), "frames_fitted": len(fits), "pos": np.full((T, 2), np.nan), "camera_view": None}  # fmt: skip
    for c in range(2):
        pos[:, c] = rolling_nanmedian(pos[:, c], int(0.2 * track.rate))
    return {"status": "ok", "reason": "", "frames_tried": len(samples), "frames_fitted": len(fits), "placed_frac": round(placed, 2), "pos": pos,
            "camera_view": max(set(views), key=views.count) if views else None, "lines_supported": sorted({n for f in fits.values() for n in f["lines_supported"]}),
            "fit_score": round(float(np.mean([f["score"] for f in fits.values()])), 2)}  # fmt: skip


# --- movement observations ----------------------------------------------------------------


def zone(p: np.ndarray) -> str:
    """Plain-language court zone for a position (near or far end is folded so both read the same)."""
    y = p[1] if p[1] <= NET_Y else LENGTH - p[1]
    x = p[0] if p[1] <= NET_Y else W_DOUBLES - p[0]
    depth = "well behind the baseline" if y < -1.5 else "around the baseline" if y < 1.5 else "inside the baseline" if y < SERVICE else "in the forecourt"
    side = "left of centre" if x < W_DOUBLES / 2 - 1.2 else "right of centre" if x > W_DOUBLES / 2 + 1.2 else "near the centre"
    return f"{depth}, {side}"


def stroke_movement(pos: np.ndarray, track: PoseTrack, contact_t: float, next_start: float | None) -> dict | None:
    """Court position at the stroke and where the player got to afterwards."""
    i = track.index_at(contact_t)
    w = max(int(0.1 * track.rate), 1)
    at = pos[max(0, i - w) : i + w + 1]
    if not np.isfinite(at).any():
        return None
    p0 = np.nanmean(at, axis=0)
    out = {"position": [round(float(p0[0]), 1), round(float(p0[1]), 1)], "zone": zone(p0), "recovery": None}
    t1 = contact_t + 1.5 if next_start is None else min(contact_t + 1.5, next_start)
    if t1 - contact_t >= 0.8:
        j = track.index_at(t1)
        late = pos[max(0, j - w) : j + w + 1]
        if np.isfinite(late).any():
            p1 = np.nanmean(late, axis=0)
            near = p0[1] <= NET_Y
            home = np.array([W_DOUBLES / 2, -0.5 if near else LENGTH + 0.5])
            out["recovery"] = {"after_s": round(t1 - contact_t, 1), "position": [round(float(p1[0]), 1), round(float(p1[1]), 1)], "zone": zone(p1), "moved_m": round(float(np.linalg.norm(p1 - p0)), 1),
                               "toward_centre_m": round(float(np.linalg.norm(p0 - home) - np.linalg.norm(p1 - home)), 1)}  # fmt: skip
    return out


def direction_changes(pos: np.ndarray, rate: float) -> list[float]:
    """Analysed-frame indices where the player's sideways travel reverses while moving at more than about 1 m/s either side."""
    x = pos[:, 0]
    k = max(int(0.25 * rate), 2)
    out: list[float] = []
    for i in range(k, len(x) - k):
        a, b, c = x[i - k], x[i], x[i + k]
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
            continue
        v1, v2 = (b - a) * rate / k, (c - b) * rate / k
        if v1 * v2 < 0 and min(abs(v1), abs(v2)) > 1.0 and (not out or i - out[-1] > 0.5 * rate):
            out.append(i)
    return out


def find_hops(track: PoseTrack) -> list[int]:
    """Analysed-frame indices of small two-footed hops (both ankles and the hips rise together, then come down).

    Pose only, no court needed. A split step is a hop timed to the opponent's hit; the opponent is not tracked, so this
    finds hops, not confirmed split steps.
    """
    y = track.xy_s[:, :, 1]
    tl = track.torso_len
    with np.errstate(invalid="ignore", divide="ignore"):
        la, ra, hip = y[:, L_ANK] / tl, y[:, R_ANK] / tl, track.mid_hip[:, 1] / tl
    k = max(int(0.2 * track.rate), 2)
    out: list[int] = []
    for i in range(k, track.n - k):
        ok = True
        for s in (la, ra, hip):
            a, b, c = s[i - k], s[i], s[i + k]
            if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)) or not (a - b > 0.035 and c - b > 0.035):  # image y grows downward: smaller = higher
                ok = False
                break
        if ok and (track.vis[i, [L_ANK, R_ANK]] >= 0.5).all() and (not out or i - out[-1] > 0.4 * track.rate):
            out.append(i)
    return out
