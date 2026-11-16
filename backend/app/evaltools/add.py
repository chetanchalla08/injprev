"""Add a clip (and optionally a label file exported from the app's Label tab) to the evaluation set.

    .venv/bin/python -m app.evaltools.add --id club_serves_01 --video ~/clips/serves.mp4 --split held_out \
        --labels ~/Downloads/labels-1a2b3c4d.json --licence "Player consent on file, internal use" --page "local recording 2026-11-20" \
        --author "Coach name" --hand right --backhand two_handed --view side

The split is fixed when the clip is added; decide it BEFORE looking at how the app does on the clip.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .format import CAMERA_VIEWS, ROOT, SPLITS, Clip, load_manifest, parse_labels, save_manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--id", required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--licence", required=True, help="licence or consent basis for using this footage")
    ap.add_argument("--page", required=True, help="where the footage came from (URL or description)")
    ap.add_argument("--author", default="")
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--hand", choices=("right", "left"), default="right")
    ap.add_argument("--backhand", choices=("one_handed", "two_handed"), default="two_handed")
    ap.add_argument("--view", choices=CAMERA_VIEWS, default="unknown")
    ap.add_argument("--notes", default="")
    a = ap.parse_args(argv)
    clips = load_manifest(a.root)
    if any(c.id == a.id for c in clips):
        raise SystemExit(f"A clip with id {a.id!r} already exists.")
    if not a.video.exists():
        raise SystemExit(f"{a.video} not found.")
    clip = Clip(id=a.id, file=f"clips/{a.id}/source{a.video.suffix.lower()}", split=a.split, handedness=a.hand, backhand_style=a.backhand, camera_view=a.view,
                source={"title": a.video.name, "page": a.page, "author": a.author, "licence": a.licence}, notes=a.notes)  # fmt: skip
    if a.labels:
        doc = json.loads(a.labels.read_text())
        d = doc.get("clip_defaults") or {}
        clip.handedness, clip.backhand_style = d.get("handedness", clip.handedness), d.get("backhand_style") or clip.backhand_style
        clip.camera_view = d.get("camera_view", clip.camera_view) if a.view == "unknown" else clip.camera_view
        doc["clip"] = a.id
        doc.pop("clip_defaults", None)
        labels = parse_labels(doc, clip.defaults)  # refuses a malformed file before anything is copied
        (a.root / "labels").mkdir(parents=True, exist_ok=True)
        (a.root / "labels" / f"{a.id}.json").write_text(json.dumps(labels.to_dict(), indent=1) + "\n")
        print(f"Labels: {len(labels.strokes)} strokes by {labels.labeller} ({'expert' if labels.expert else 'non-expert'}).")
    (a.root / "clips" / a.id).mkdir(parents=True, exist_ok=True)
    shutil.copy(a.video, a.root / clip.file)
    save_manifest([*clips, clip], a.root)
    print(f"Added {a.id} to the {a.split} split. Score with: python -m app.evaltools.score --name <run-name>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
