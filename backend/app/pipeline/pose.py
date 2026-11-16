"""Pose estimation (MediaPipe Pose Landmarker) and single-player tracking.

Why MediaPipe: 33 landmarks including heels and toes (needed for knee-over-foot checks),
a per-landmark visibility estimate (needed to suppress weak measurements), multi-person
output, CPU-only inference, and a permissive licence. Its limits matter and are surfaced
in the product: accuracy drops for small, distant or blurred players, its 3D "world"
output is a single-camera estimate, and it was not trained specifically on tennis.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np

from .errors import PipelineError
from .signals import L_HIP, L_SH, N_LANDMARKS, R_HIP, R_SH, PoseTrack
from .video import iter_frames, read_frame

MAX_PEOPLE = 4
Progress = Callable[[float], None]


@dataclass
class FramePoses:
    """All people found in one frame. lm: (P, 33, 4) = x, y (normalised 0-1), z, visibility. world: (P, 33, 3)."""

    lm: np.ndarray
    world: np.ndarray


class Detector(Protocol):
    def detect_video(self, path: Path, stride: int, progress: Progress) -> tuple[list[int], list[FramePoses]]: ...

    def detect_image(self, frame_bgr: np.ndarray) -> FramePoses: ...


def _to_frame_poses(result) -> FramePoses:
    P = len(result.pose_landmarks)
    lm = np.zeros((P, N_LANDMARKS, 4), dtype=np.float32)
    world = np.zeros((P, N_LANDMARKS, 3), dtype=np.float32)
    for p in range(P):
        for j, l in enumerate(result.pose_landmarks[p]):
            lm[p, j] = (l.x, l.y, l.z, l.visibility if l.visibility is not None else 0.0)
        for j, l in enumerate(result.pose_world_landmarks[p]):
            world[p, j] = (l.x, l.y, l.z)
    return FramePoses(lm, world)


class MediaPipeDetector:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise PipelineError(
                "model_missing",
                f"Pose model not found at {model_path}.",
                "Run `python scripts/download_models.py` in the backend folder.",
            )
        self.model_path = model_path

    def _make(self, mode: str, num_poses: int = MAX_PEOPLE):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        opts = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return mp, vision.PoseLandmarker.create_from_options(opts)

    def detect_video(self, path: Path, stride: int, progress: Progress) -> tuple[list[int], list[FramePoses]]:
        mp, landmarker = self._make("video")
        cap = cv2.VideoCapture(str(path))
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        idxs: list[int] = []
        out: list[FramePoses] = []
        try:
            for i, frame in iter_frames(path, stride):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = landmarker.detect_for_video(image, int(round(i * 1000.0 / fps)))
                idxs.append(i)
                out.append(_to_frame_poses(res))
                if len(idxs) % 15 == 0:
                    progress(min(i / total, 1.0))
        finally:
            landmarker.close()
        return idxs, out

    def detect_crops(self, path: Path, frame_idx: list[int], boxes: list, progress: Progress) -> list[FramePoses | None]:
        """Single-person pose on an enlarged crop of each frame. Landmarks are returned in whole-frame normalised coordinates."""
        mp, landmarker = self._make("image", num_poses=1)
        want = {f: k for k, f in enumerate(frame_idx)}
        out: list[FramePoses | None] = [None] * len(frame_idx)
        try:
            for i, frame in iter_frames(path, 1):
                k = want.get(i)
                if k is None or boxes[k] is None:
                    continue
                x0, y0, x1, y1 = boxes[k]
                if x1 - x0 < 16 or y1 - y0 < 16:
                    continue
                crop = frame[y0:y1, x0:x1]
                f = CROP_TARGET / max(crop.shape[:2])
                if f > 1.05:
                    crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
                fp = _to_frame_poses(landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))))
                if fp.lm.shape[0]:
                    H, W = frame.shape[:2]
                    fp.lm[:, :, 0] = (x0 + fp.lm[:, :, 0] * (x1 - x0)) / W
                    fp.lm[:, :, 1] = (y0 + fp.lm[:, :, 1] * (y1 - y0)) / H
                    out[k] = fp
                if k % 30 == 0:
                    progress(k / max(len(frame_idx), 1))
        finally:
            landmarker.close()
        return out

    def detect_image(self, frame_bgr: np.ndarray) -> FramePoses:
        mp, landmarker = self._make("image")
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return _to_frame_poses(landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)))
        finally:
            landmarker.close()


# --- player candidates ------------------------------------------------------------------


def _bbox(lm: np.ndarray) -> tuple[float, float, float, float] | None:
    """Normalised bounding box (x0, y0, x1, y1) of confidently visible landmarks."""
    ok = lm[:, 3] > 0.3
    if ok.sum() < 8:
        return None
    x, y = lm[ok, 0], lm[ok, 1]
    x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
    px, py = 0.08 * (x1 - x0), 0.08 * (y1 - y0)
    return (float(max(x0 - px, 0)), float(max(y0 - py, 0)), float(min(x1 + px, 1)), float(min(y1 + py, 1)))


def _torso_centre(lm: np.ndarray) -> np.ndarray:
    return lm[[L_SH, R_SH, L_HIP, R_HIP], :2].mean(axis=0)


def find_candidates(detector: Detector, video: Path, meta: dict, n_samples: int = 9) -> dict:
    """Look for people in sampled frames and return the frame showing the most of them.

    Returns {"frame_idx", "frame": BGR image, "candidates": [{"box": [x0,y0,x1,y1], "height_frac"}]}.
    """
    n = meta["n_frames"]
    best: tuple[int, int, np.ndarray, list[dict]] | None = None
    for idx in np.unique(np.linspace(n * 0.05, n * 0.95, n_samples).astype(int)):
        frame = read_frame(video, int(idx))
        if frame is None:
            continue
        poses = detector.detect_image(frame)
        cands = []
        for p in range(poses.lm.shape[0]):
            box = _bbox(poses.lm[p])
            if box is None or (box[3] - box[1]) < 0.08:
                continue
            c = _torso_centre(poses.lm[p])
            cands.append({"box": [round(v, 4) for v in box], "height_frac": round(box[3] - box[1], 3), "centre": [float(c[0]), float(c[1])]})
        cands.sort(key=lambda c: -c["height_frac"])
        score = len(cands)
        if cands and (best is None or score > best[0]):
            best = (score, int(idx), frame, cands)
    if best is None:
        raise PipelineError(
            "no_person",
            "We could not find a person in this video.",
            "Make sure the whole player is in the frame, reasonably large, and well lit.",
        )
    return {"frame_idx": best[1], "frame": best[2], "candidates": best[3]}


# --- tracking ---------------------------------------------------------------------------


REACQ_AUTO_SIM = 0.80  # appearance match needed to carry on automatically after a cut or re-entry
REACQ_ASK_SIM = 0.60  # below this the person is treated as someone else
REACQ_MARGIN = 0.08  # ...and must beat anyone else in view by this much
REACQ_MIN_S = 0.3  # a re-acquired stretch shorter than this is ignored


def _tracklets(detections: list[FramePoses], scale: np.ndarray, H: int, rate: float, cut_set: set[int]) -> list[np.ndarray]:
    """Give every detection a tracklet id by linking each person to the nearest one in the previous frames (never across a cut)."""
    T = len(detections)
    tid = [np.full(d.lm.shape[0], -1, dtype=int) for d in detections]
    live: dict[int, tuple[int, np.ndarray, float]] = {}  # id -> (last frame, centre px, height px)
    nxt = 0
    max_gap = max(int(0.5 * rate), 1)
    for i in range(T):
        if i in cut_set:
            live = {}
        live = {k: v for k, v in live.items() if i - v[0] <= max_gap}
        taken: set[int] = set()
        for p in range(detections[i].lm.shape[0]):
            c = _torso_centre(detections[i].lm[p]) * scale
            b = _bbox(detections[i].lm[p])
            h = (b[3] - b[1]) * H if b else 0.0
            best, best_d = -1, np.inf
            for k, (_, lc, lh) in live.items():
                d = float(np.linalg.norm(c - lc))
                if k not in taken and d < best_d and d < 0.35 * max(lh, 1.0) * 1.5 and (h == 0 or 0.5 < h / max(lh, 1.0) < 2.0):
                    best, best_d = k, d
            if best < 0:
                best, nxt = nxt, nxt + 1
            taken.add(best)
            tid[i][p] = best
            live[best] = (i, c, h if h > 0 else live.get(best, (0, c, 1.0))[2])
    return tid


def track_player(frame_idx: list[int], detections: list[FramePoses], seed_frame: int, seed_centre: tuple[float, float], seed_height: float, meta: dict, *,
                 cuts: list[int] | None = None, appearance: np.ndarray | None = None, reacquire: bool = False) -> PoseTrack:  # fmt: skip
    """Follow the chosen person through the video.

    Within a shot, association is by nearest torso centre, forwards and backwards from the frame the user picked the
    player in: a detection is accepted only inside a gate that scales with the player's apparent height, the gate widens
    slowly while the player is missing, and the track is abandoned in that direction after 1.5 s without a match or at
    a scene cut.

    With ``reacquire``, stretches after a cut or after the player left the frame are examined. A person can only be
    picked up again if (1) they are not someone who was already visible alongside the player, and (2) their clothing
    colours match the player's. A strong, unambiguous match continues automatically; a weaker one is kept but marked
    "needs_confirmation" and is not analysed until the user confirms it; anything else is left untracked. The track
    never moves to a different person silently.
    """
    T = len(frame_idx)
    W, H, fps = meta["width"], meta["height"], meta["fps"]
    stride = frame_idx[1] - frame_idx[0] if T > 1 else 1
    rate = fps / stride
    chosen = np.full(T, -1, dtype=int)
    seg = np.full(T, -1, dtype=int)
    scale = np.array([W, H], dtype=float)
    cut_set = set(cuts or [])

    def centre_px(i: int, p: int) -> np.ndarray:
        return _torso_centre(detections[i].lm[p]) * scale

    def height_px(i: int, p: int) -> float:
        b = _bbox(detections[i].lm[p])
        return (b[3] - b[1]) * H if b else 0.0

    seed_i = int(np.argmin(np.abs(np.asarray(frame_idx) - seed_frame)))
    target = np.asarray(seed_centre) * scale
    size = seed_height * H
    # find the seed detection, searching up to 1 s either side of the picked frame
    start = None
    for off in sorted(range(-int(rate), int(rate) + 1), key=abs):
        i = seed_i + off
        if not 0 <= i < T or detections[i].lm.shape[0] == 0:
            continue
        d = [np.linalg.norm(centre_px(i, p) - target) for p in range(detections[i].lm.shape[0])]
        p = int(np.argmin(d))
        if d[p] < 0.6 * size:
            start = (i, p)
            break
    if start is None:
        raise PipelineError(
            "tracking_lost",
            "We could not follow the selected player through the video.",
            "Choose a clip where the player stays in view, or pick the player again.",
        )
    chosen[start[0]] = start[1]
    seg[start[0]] = 0
    max_missing = int(1.5 * rate)

    def run(order: range) -> None:
        last = centre_px(*start)
        last_size = max(height_px(*start), 1.0)
        missing = 0
        step = 1 if order.step > 0 else -1
        for i in order:
            if (i if step > 0 else i + 1) in cut_set:
                break  # never follow across a scene cut
            P = detections[i].lm.shape[0]
            best_p, best_d = -1, np.inf
            for p in range(P):
                d = float(np.linalg.norm(centre_px(i, p) - last))
                if d < best_d:
                    best_p, best_d = p, d
            # the gate doubles at most (after 1 s missing) so the track cannot leap to a different person
            gate = 0.35 * last_size * (1 + min(missing / rate, 1.0))
            similar = best_p >= 0 and 0.5 < (height_px(i, best_p) or last_size) / last_size < 2.0
            if best_p >= 0 and best_d < gate and similar:
                chosen[i] = best_p
                seg[i] = 0
                last = centre_px(i, best_p)
                h = height_px(i, best_p)
                if h > 0:
                    last_size = 0.8 * last_size + 0.2 * h
                missing = 0
            else:
                missing += 1
                if missing > max_missing:
                    break

    run(range(start[0] + 1, T))
    run(range(start[0] - 1, -1, -1))

    def box_of(i: int, p: int) -> list[float] | None:
        b = _bbox(detections[i].lm[p])
        return [round(v, 4) for v in b] if b else None

    first = int(np.where(seg == 0)[0][0])
    segments = [{"id": 0, "kind": "seed", "status": "auto", "after": None, "confidence": 1.0, "similarity": None, "box": box_of(first, chosen[first]), "box_t": round(frame_idx[first] / fps, 3)}]
    if reacquire:
        tid = _tracklets(detections, scale, H, rate, cut_set)
        tracked0 = np.where(seg == 0)[0]
        mine = {int(tid[i][chosen[i]]) for i in tracked0}
        others = {int(t) for i in tracked0 for p, t in enumerate(tid[i]) if p != chosen[i]}  # people seen alongside the player: never candidates
        ref = None
        if appearance is not None:
            good = [appearance[i, chosen[i]] for i in tracked0 if np.isfinite(appearance[i, chosen[i]]).all()]
            if len(good) >= 5:
                ref = np.median(np.asarray(good), axis=0)
        from .visual import similarity

        members: dict[int, list[tuple[int, int]]] = {}
        for i in range(T):
            if seg[i] >= 0:
                continue
            for p, t in enumerate(tid[i]):
                members.setdefault(int(t), []).append((i, p))
        cut_list = sorted(cut_set)
        for t, frames in sorted(members.items(), key=lambda kv: kv[1][0][0]):
            frames = [(i, p) for i, p in frames if seg[i] < 0]
            if t in others or t in mine or len(frames) < max(int(REACQ_MIN_S * rate), 2):
                continue
            i0 = frames[0][0]
            prev = tracked0[tracked0 < i0]
            edge = int(prev[-1]) if prev.size else int(tracked0[0])
            lo, hi = min(edge, i0), max(edge, i0)
            after_cut = any(lo < c <= hi for c in cut_list)
            sims = [v for v in (similarity(ref, appearance[i, p]) for i, p in frames[:15]) if v is not None] if ref is not None else []
            sim = float(np.median(sims)) if sims else None
            rivals = []
            if ref is not None:
                for p in range(detections[i0].lm.shape[0]):
                    if tid[i0][p] != t and (v := similarity(ref, appearance[i0, p])) is not None:
                        rivals.append(v)
            alone = detections[i0].lm.shape[0] == 1
            unique = not rivals or sim is None or sim - max(rivals) >= REACQ_MARGIN
            status = None
            if sim is None:
                # nothing to compare clothing with: only offer it when there was no cut, the gap was short and nobody else is around
                near = np.linalg.norm(centre_px(*frames[0]) - centre_px(edge, chosen[edge])) < 1.5 * max(height_px(edge, chosen[edge]), 1.0)
                if not after_cut and alone and abs(i0 - edge) <= 3 * rate and near:
                    status, conf = "needs_confirmation", 0.4
            elif sim >= REACQ_AUTO_SIM and unique:
                status, conf = "auto", sim
            elif sim >= REACQ_ASK_SIM:
                status, conf = "needs_confirmation", round(sim * (0.9 if unique else 0.7), 2)
            if status is None:
                continue
            sid_ = len(segments)
            for i, p in frames:
                chosen[i], seg[i] = p, sid_
            segments.append({"id": sid_, "kind": "reacquired", "status": status, "after": "cut" if after_cut else "left_frame", "confidence": round(float(conf), 2),
                             "similarity": None if sim is None else round(sim, 2), "box": box_of(*frames[0]), "box_t": round(frame_idx[i0] / fps, 3)})  # fmt: skip
            if status == "auto":
                mine.add(t)
                others |= {int(x) for i, p in frames for q, x in enumerate(tid[i]) if q != p}

    xy = np.full((T, N_LANDMARKS, 2), np.nan)
    vis = np.zeros((T, N_LANDMARKS))
    world = np.full((T, N_LANDMARKS, 3), np.nan)
    for i in range(T):
        p = chosen[i]
        if p < 0:
            continue
        xy[i] = detections[i].lm[p, :, :2] * scale
        vis[i] = detections[i].lm[p, :, 3]
        world[i] = detections[i].world[p]
    t = np.asarray(frame_idx) / fps
    for sgm in segments:
        fr = np.where(seg == sgm["id"])[0]
        sgm |= {"start": round(float(t[fr[0]]), 3), "end": round(float(t[fr[-1]]), 3), "frames": int(fr.size)}
    return PoseTrack(fps=fps, width=W, height=H, frame_idx=np.asarray(frame_idx), xy=xy, vis=vis, world=world, seg=seg, segments=segments)


# --- optional second pass: pose on a crop around the selected player --------------------------

CROP_PAD = 0.25
CROP_TARGET = 512  # the crop is scaled so its longer side is this many pixels


def crop_boxes(track: PoseTrack) -> list[tuple[int, int, int, int] | None]:
    """A padded pixel box around the tracked player for each analysed frame (None where not tracked)."""
    out: list[tuple[int, int, int, int] | None] = []
    for i in range(track.n):
        ok = track.vis[i] > 0.3
        if not track.tracked[i] or ok.sum() < 8:
            out.append(None)
            continue
        x, y = track.xy[i, ok, 0], track.xy[i, ok, 1]
        w, h = x.max() - x.min(), y.max() - y.min()
        pad = CROP_PAD * max(w, h) + 4
        out.append((int(max(x.min() - pad, 0)), int(max(y.min() - pad, 0)), int(min(x.max() + pad, track.width)), int(min(y.max() + pad, track.height))))
    return out


def refine_with_crops(detector, video: Path, track: PoseTrack, progress: Progress) -> tuple[PoseTrack, str]:
    """Re-run the pose model on an enlarged crop of the player and use its landmarks where it found the same person.

    Evaluated against full-frame inference on the evaluation set (docs/VALIDATION.md); off unless POSE_MODE=cropped.
    """
    boxes = crop_boxes(track)
    crops = detector.detect_crops(video, [int(f) for f in track.frame_idx], boxes, progress)
    xy, vis, world = track.xy.copy(), track.vis.copy(), track.world.copy()
    used = 0
    scale = np.array([track.width, track.height], float)
    for i, fp in enumerate(crops):
        if fp is None or fp.lm.shape[0] == 0 or boxes[i] is None:
            continue
        new = fp.lm[0, :, :2] * scale
        old_c = np.nanmean(track.xy[i, [L_SH, R_SH, L_HIP, R_HIP]], axis=0)
        new_c = new[[L_SH, R_SH, L_HIP, R_HIP]].mean(axis=0)
        size = boxes[i][3] - boxes[i][1]
        if not np.isfinite(old_c).all() or np.linalg.norm(new_c - old_c) > 0.2 * size:
            continue  # the crop pass locked onto something else: keep the full-frame result
        xy[i], vis[i], world[i] = new, fp.lm[0, :, 3], fp.world[0]
        used += 1
    n = max(int(track.tracked.sum()), 1)
    return (PoseTrack(track.fps, track.width, track.height, track.frame_idx, xy, vis, world, track.seg, track.segments),
            f"Landmarks come from a second pose pass on an enlarged crop of the player in {used / n:.0%} of tracked frames; the rest use the full-frame result.")  # fmt: skip
