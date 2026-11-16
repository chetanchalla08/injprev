"""Switches for the optional analysis stages.

Every stage added after the first build can be turned off, so the evaluation harness can
score the pipeline with and without it (docs/VALIDATION.md records the comparisons).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Options:
    pose_mode: str = "full_frame"  # full_frame | cropped (second pass on a crop around the selected player)
    ball: bool = True  # ball tracking -> contact time, contact location, swing phases
    racket: bool = False  # pretrained racket detector (needs the optional detector weights)
    recover_hidden: bool = False  # propose strokes from ball contacts when the hitting arm is hidden. Off: on the tune split it added 37 false strokes and recovered none (docs/VALIDATION.md)
    court: bool = True  # court lines -> position on court, recovery, split-step
    cuts: bool = True  # scene-cut detection
    reacquire: bool = True  # re-acquire the selected player after a cut or after leaving frame

    def to_dict(self) -> dict:
        return asdict(self)


BASELINE = Options(ball=False, racket=False, recover_hidden=False, court=False, cuts=False, reacquire=False)  # the first build's behaviour
