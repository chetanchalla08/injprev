"""Framing guidance for live capture: is the whole player in shot, big enough, and well lit?

Runs on single low-resolution snapshots sent by the capture page before recording starts.
Snapshots are checked in memory and never stored. Steadiness is judged in the browser from
consecutive frames. The cut-offs mirror the recording-quality warnings in video.py.
"""
from __future__ import annotations

import cv2
import numpy as np

from .pose import FramePoses
from .signals import L_ANK, NOSE, R_ANK
from .video import DARK_LUMA, SOFT_LAPLACIAN_VAR

MIN_HEIGHT, GOOD_HEIGHT, MAX_HEIGHT = 0.25, 0.33, 0.9
EDGE = 0.03


def assess(frame_bgr: np.ndarray, poses: FramePoses) -> dict:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (640, max(int(640 * gray.shape[0] / gray.shape[1]), 2)))
    luma, sharp = float(small.mean()), float(cv2.Laplacian(small, cv2.CV_64F).var())
    checks = []

    def add(key: str, ok: bool | None, message: str) -> None:
        checks.append({"key": key, "ok": ok, "message": message})

    add("lighting", luma >= DARK_LUMA, "Lighting looks fine." if luma >= DARK_LUMA else "Too dark. Move into daylight or turn the court lights on, with the light behind the camera.")
    add("focus", sharp >= SOFT_LAPLACIAN_VAR, "Picture is sharp." if sharp >= SOFT_LAPLACIAN_VAR else "The picture is soft. Clean the lens and tap to focus on the player.")
    people = []
    for p in range(poses.lm.shape[0]):
        lm = poses.lm[p]
        ok = lm[:, 3] > 0.3
        if ok.sum() >= 8:
            people.append((float(lm[ok, 1].max() - lm[ok, 1].min()), lm))
    people.sort(key=lambda t: -t[0])
    height = None
    if not people:
        add("person", False, "No player found yet. Stand where you will hit, facing any way, with your whole body in shot.")
        add("whole_body", None, "Waiting for a player.")
        add("size", None, "Waiting for a player.")
    else:
        height, lm = people[0]
        add("person", True, "Player found." if len(people) == 1 else f"{len(people)} people in shot. That works, but you will be asked which one to follow.")
        head = lm[NOSE, 3] >= 0.5 and lm[NOSE, 1] > EDGE
        feet = min(lm[L_ANK, 3], lm[R_ANK, 3]) >= 0.5 and max(lm[L_ANK, 1], lm[R_ANK, 1]) < 1 - EDGE
        vis = lm[lm[:, 3] > 0.5]
        sides = bool(len(vis) and vis[:, 0].min() > EDGE and vis[:, 0].max() < 1 - EDGE)
        if head and feet and sides:
            add("whole_body", True, "Whole body is in shot. Leave room for the racket above the head and out to the sides.")
        else:
            cut = [n for n, good in (("head", head), ("feet", feet), ("left or right edge", sides)) if not good]
            add("whole_body", False, f"Cut off at the {' and '.join(cut)}. Move the camera back or re-aim so the whole body is in shot.")
        if height < MIN_HEIGHT:
            add("size", False, f"The player fills only about {height:.0%} of the picture height. Move the camera closer: aim for at least a third.")
        elif height > MAX_HEIGHT:
            add("size", False, "The player is too close: a swing or serve reach will leave the picture. Move the camera back.")
        else:
            add("size", True, f"Player size is {'good' if height >= GOOD_HEIGHT else 'acceptable, closer would be better'} (about {height:.0%} of the picture height).")
    return {"ready": all(c["ok"] for c in checks), "checks": checks, "people": len(people), "height_frac": None if height is None else round(height, 2), "brightness": round(luma, 1), "sharpness": round(sharp, 1)}
