"""Report pose and angle error for evaluation clips that have reference data.

    .venv/bin/python -m app.evaltools.angles            # every clip with evalset/annotations/<id>.json or evalset/mocap/<id>.csv
    .venv/bin/python -m app.evaltools.angles --offset club_serves_01=0.42

Reference data comes from people: joint clicks exported from the app's Annotate tab, or a
motion-capture CSV (format in app/angles.py). With none present this prints that there is
nothing to report; it never estimates accuracy on its own. Run the scoring command first so
each clip's pose track exists in evalset/cache/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import angles
from ..pipeline.signals import PoseTrack
from .format import ROOT, load_manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--offset", action="append", default=[], metavar="CLIP=SECONDS", help="seconds to add to a clip's motion-capture times")
    a = ap.parse_args(argv)
    offsets = {k: float(v) for k, v in (o.split("=") for o in a.offset)}
    out, by_view = {}, {}
    for clip in load_manifest(a.root):
        ann, moc, trk = a.root / "annotations" / f"{clip.id}.json", a.root / "mocap" / f"{clip.id}.csv", a.root / "cache" / "work" / clip.id / "track.npz"
        if not ann.exists() and not moc.exists():
            continue
        if not trk.exists():
            print(f"{clip.id}: reference data found but no pose track; run `python -m app.evaltools.score` first.")
            continue
        track = PoseTrack.load(trk)
        res = {"camera_view": clip.camera_view, "split": clip.split}
        if ann.exists():
            doc = json.loads(ann.read_text())
            res["annotations"] = {"annotator": doc.get("annotator"), **angles.compare_annotations(track, doc["frames"])}
        if moc.exists():
            res["mocap"] = angles.compare_mocap(track, angles.parse_mocap_csv(moc.read_text()), offsets.get(clip.id, 0.0))
        out[clip.id] = res
        by_view.setdefault(clip.camera_view, []).append(clip.id)
        print(f"\n{clip.id}  (camera view: {clip.camera_view}, split: {clip.split})")
        for joint, v in (res.get("annotations") or {}).get("position_error_by_joint", {}).items():
            print(f"  position  {joint:<16} n={v['n']:<3} median {v['median_px']} px = {v['median_torso_lengths']} torso lengths")
        for k, v in (res.get("annotations") or {}).get("image_plane_angle_error", {}).items():
            print(f"  2D angle  {k:<20} n={v['n']:<3} median |err| {v['median_abs_deg']} deg, mean signed {v['mean_signed_deg']} deg")
        for k, v in (res.get("mocap") or {}).get("angles", {}).items():
            for m, e in v["methods"].items():
                print(f"  mocap     {k:<20} {m:<13} n={e['n']:<4}" + (f" mean |err| {e['mean_abs_deg']} deg, bias {e['mean_signed_deg']} deg, RMSE {e['rmse_deg']} deg" if e["n"] else " no overlapping tracked samples"))
    if not out:
        print("No reference data found (evalset/annotations/*.json or evalset/mocap/*.csv). Angle accuracy is unknown; nothing is reported.")
        return 1
    (a.root / "runs").mkdir(exist_ok=True)
    (a.root / "runs" / "angles_latest.json").write_text(json.dumps({"by_camera_view": by_view, "clips": out}, indent=1))
    print("\nClips by camera view: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in by_view.items()) + ". Do not pool errors across views.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
