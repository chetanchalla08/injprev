"""On-disk evaluation-set format (documented in evalset/README.md).

evalset/
  manifest.json          clips: where the footage came from, its licence, the fixed split
  clips/<id>/source.*    the footage itself (git-ignored; re-download from manifest sources)
  labels/<id>.json       human stroke labels for one clip
  annotations/<id>.json  hand-clicked joint positions on chosen frames (optional)
  mocap/<id>.csv         reference joint angles from motion capture (optional)
  runs/*.json            stored scoring runs, so changes can be compared
  cache/                 cached pose-model outputs (safe to delete; `score --clear-cache`)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

LABEL_FORMAT = "baseline-labels/1"
ANNOTATION_FORMAT = "baseline-joints/1"
STROKE_LABELS = ("forehand", "backhand", "serve", "other")  # "other" = a real stroke the app does not label (volley, slice, smash)
CAMERA_VIEWS = ("side", "behind", "front", "oblique", "far", "unknown")
SPLITS = ("tune", "held_out")
ROOT = Path(__file__).resolve().parents[3] / "evalset"


class FormatError(ValueError):
    pass


@dataclass
class LabelledStroke:
    type: str
    start: float
    contact: float  # ball contact when the ball is visible, otherwise the labeller's estimate of the hit (see contact_kind)
    end: float
    contact_kind: str = "contact"  # contact | peak (shadow swing or ball not visible: fastest point of the swing)
    handedness: str = "right"
    backhand_style: str | None = None
    camera_view: str = "unknown"
    notes: str = ""


@dataclass
class Labels:
    clip: str
    labeller: str
    expert: bool
    strokes: list[LabelledStroke]
    labeller_role: str = ""
    method: str = ""
    labelled_at: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {"format": LABEL_FORMAT, "clip": self.clip, "labeller": self.labeller, "labeller_role": self.labeller_role, "expert": self.expert, "method": self.method,
                "labelled_at": self.labelled_at or time.strftime("%Y-%m-%d"), "notes": self.notes, "strokes": [vars(s) for s in self.strokes]}  # fmt: skip


def parse_labels(d: dict, defaults: dict | None = None) -> Labels:
    """Validate a label file. Per-stroke handedness / backhand style / camera view fall back to the clip defaults."""
    defaults = defaults or {}
    if d.get("format") != LABEL_FORMAT:
        raise FormatError(f"labels: expected format {LABEL_FORMAT!r}")
    if not str(d.get("labeller") or "").strip():
        raise FormatError("labels: 'labeller' is required (who marked these strokes)")
    if not isinstance(d.get("expert"), bool):
        raise FormatError("labels: 'expert' must be true or false (is the labeller a qualified coach?)")
    strokes = []
    for i, s in enumerate(d.get("strokes") or []):
        try:
            st = LabelledStroke(
                type=s["type"], start=float(s["start"]), contact=float(s["contact"]), end=float(s["end"]), contact_kind=s.get("contact_kind", "contact"),
                handedness=s.get("handedness") or defaults.get("handedness", "right"), backhand_style=s.get("backhand_style") or defaults.get("backhand_style"),
                camera_view=s.get("camera_view") or defaults.get("camera_view", "unknown"), notes=str(s.get("notes") or ""),
            )  # fmt: skip
        except (KeyError, TypeError, ValueError) as e:
            raise FormatError(f"labels: stroke {i} is missing or has a bad field: {e}") from e
        if st.type not in STROKE_LABELS:
            raise FormatError(f"labels: stroke {i} type {st.type!r} not in {STROKE_LABELS}")
        if not st.start <= st.contact <= st.end:
            raise FormatError(f"labels: stroke {i} needs start <= contact <= end")
        if st.contact_kind not in ("contact", "peak") or st.handedness not in ("right", "left") or st.camera_view not in CAMERA_VIEWS:
            raise FormatError(f"labels: stroke {i} has an invalid contact_kind, handedness or camera_view")
        strokes.append(st)
    strokes.sort(key=lambda s: s.contact)
    return Labels(clip=str(d.get("clip") or ""), labeller=str(d["labeller"]), expert=d["expert"], strokes=strokes, labeller_role=str(d.get("labeller_role") or ""),
                  method=str(d.get("method") or ""), labelled_at=str(d.get("labelled_at") or ""), notes=str(d.get("notes") or ""))  # fmt: skip


@dataclass
class Clip:
    id: str
    file: str
    split: str
    handedness: str = "right"
    backhand_style: str = "two_handed"
    camera_view: str = "unknown"
    t0: float | None = None  # optional trim of the source file, seconds
    t1: float | None = None
    player_hint: list[float] | None = None  # normalised (x, y) of the player to follow when several people are visible
    source: dict = field(default_factory=dict)  # title, page, author, licence, licence_url
    cuts: list[float] | None = None  # known scene-cut times (seconds in the clip), for the cut-detection check
    notes: str = ""

    @property
    def defaults(self) -> dict:
        return {"handedness": self.handedness, "backhand_style": self.backhand_style, "camera_view": self.camera_view}


def load_manifest(root: Path = ROOT) -> list[Clip]:
    p = root / "manifest.json"
    if not p.exists():
        return []
    clips = []
    for c in json.loads(p.read_text())["clips"]:
        clip = Clip(**c)
        if clip.split not in SPLITS:
            raise FormatError(f"manifest: clip {clip.id} split must be one of {SPLITS}")
        if not (clip.source.get("licence") and clip.source.get("page")):
            raise FormatError(f"manifest: clip {clip.id} must state its source page and licence")
        clips.append(clip)
    if len({c.id for c in clips}) != len(clips):
        raise FormatError("manifest: duplicate clip ids")
    return clips


def save_manifest(clips: list[Clip], root: Path = ROOT) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({"version": 1, "clips": [vars(c) for c in clips]}, indent=1, ensure_ascii=False) + "\n")


def load_labels(clip: Clip, root: Path = ROOT) -> Labels | None:
    p = root / "labels" / f"{clip.id}.json"
    return parse_labels(json.loads(p.read_text()), clip.defaults) if p.exists() else None
