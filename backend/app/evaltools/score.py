"""Score the real pipeline against the labelled evaluation set.

    cd backend && .venv/bin/python -m app.evaltools.score --name baseline
    .venv/bin/python -m app.evaltools.score --compare baseline with_ball
    .venv/bin/python -m app.evaltools.score --list

Reports stroke-detection precision and recall, label confusion and timing error, on the
fixed tune and held-out splits, and split by camera view, handedness and stroke type.
Each run is stored in evalset/runs/ so changes can be compared. A clip with no label file
is run and listed, but contributes nothing to the scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np

from ..config import load_settings
from ..pipeline import run
from ..pipeline.errors import PipelineError
from ..pipeline.options import BASELINE, Options
from ..pipeline.pose import Detector, FramePoses
from ..pipeline.video import ffmpeg_exe
from .format import ROOT, SPLITS, Clip, Labels, load_labels, load_manifest

MATCH_TOLERANCE_S = 0.5  # a detection counts for a labelled stroke if its timestamp is within this of the labelled contact
PRED_TYPES = ("forehand", "backhand", "serve", "unclassified")


# --- matching and metrics (pure functions; tested on synthetic inputs) --------------------


def match(labels: list[dict], preds: list[dict], tol: float = MATCH_TOLERANCE_S) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy one-to-one matching by time, closest pairs first. Returns (pairs, unmatched label idx, unmatched pred idx)."""
    cand = sorted((abs(l["contact"] - p["time"]), i, j) for i, l in enumerate(labels) for j, p in enumerate(preds) if abs(l["contact"] - p["time"]) <= tol)
    used_l, used_p, pairs = set(), set(), []
    for _, i, j in cand:
        if i not in used_l and j not in used_p:
            used_l.add(i), used_p.add(j), pairs.append((i, j))
    return sorted(pairs), [i for i in range(len(labels)) if i not in used_l], [j for j in range(len(preds)) if j not in used_p]


def _timing(errs: list[float]) -> dict | None:
    if not errs:
        return None
    a = sorted(abs(e) for e in errs)
    return {"n": len(errs), "mean_signed_s": round(statistics.fmean(errs), 3), "median_abs_s": round(statistics.median(a), 3), "p90_abs_s": round(a[min(len(a) - 1, int(0.9 * len(a)))], 3)}


def summarise(items: list[dict]) -> dict:
    """items: one dict per labelled stroke {"true", "pred" (type or "missed"), "err", "contact_kind"} plus false detections {"true": None, "pred"}.

    Strokes labelled "other" (volley, slice, smash...) are real strokes the app does not claim to label: finding one is
    not a false detection and missing one is not a miss, so they are excluded from precision and recall.
    """
    scored = [x for x in items if x["true"] in ("forehand", "backhand", "serve")]
    found = [x for x in scored if x["pred"] != "missed"]
    false = [x for x in items if x["true"] is None]
    correct = [x for x in found if x["pred"] == x["true"]]
    wrong = [x for x in found if x["pred"] not in (x["true"], "unclassified")]
    conf = {t: {p: 0 for p in (*PRED_TYPES, "missed")} for t in ("forehand", "backhand", "serve")}
    for x in scored:
        conf[x["true"]][x["pred"]] += 1
    r = lambda a, b: round(a / b, 3) if b else None  # noqa: E731
    return {
        "labelled": len(scored), "detected": len(found), "false_detections": len(false), "correct_label": len(correct), "wrong_label": len(wrong),
        "left_unlabelled": len(found) - len(correct) - len(wrong), "missed": len(scored) - len(found),
        "detection_recall": r(len(found), len(scored)), "detection_precision": r(len(found), len(found) + len(false)),
        "label_accuracy_of_detected": r(len(correct), len(found)), "correct_end_to_end": r(len(correct), len(scored)),
        "confusion": conf, "false_by_predicted_type": {p: sum(x["pred"] == p for x in false) for p in PRED_TYPES},
        "timing_vs_contact": _timing([x["err"] for x in found if x["contact_kind"] == "contact"]),
        "timing_vs_swing_peak": _timing([x["err"] for x in found if x["contact_kind"] == "peak"]),
        "timing_sources": {k: sum(x.get("timing_source") == k for x in found) for k in sorted({x.get("timing_source") or "?" for x in found})},
    }  # fmt: skip


def score_clip(labels: Labels, preds: list[dict]) -> list[dict]:
    lab = [vars(s) for s in labels.strokes]
    pairs, miss, extra = match(lab, preds)
    items = []
    for i, j in pairs:
        items.append({"true": lab[i]["type"], "pred": preds[j]["type"], "err": round(preds[j]["time"] - lab[i]["contact"], 3), "contact_kind": lab[i]["contact_kind"],
                      "timing_source": preds[j].get("timing_source"), "t": lab[i]["contact"], "handedness": lab[i]["handedness"], "camera_view": lab[i]["camera_view"]})  # fmt: skip
    for i in miss:
        items.append({"true": lab[i]["type"], "pred": "missed", "err": None, "contact_kind": lab[i]["contact_kind"], "t": lab[i]["contact"], "handedness": lab[i]["handedness"], "camera_view": lab[i]["camera_view"]})
    for j in extra:
        items.append({"true": None, "pred": preds[j]["type"], "t": preds[j]["time"]})
    return sorted(items, key=lambda x: x["t"])


def aggregate(per_clip: list[dict]) -> dict:
    """per_clip: [{"clip", "split", "items": [...]}] -> metrics for all / each split, each further broken down."""
    out = {}
    for split in ("all", *SPLITS):
        clips = [c for c in per_clip if split == "all" or c["split"] == split]
        items = [x for c in clips for x in c["items"]]
        block = {"clips": len(clips), "overall": summarise(items)}
        for key in ("camera_view", "handedness", "true"):
            groups = sorted({x[key] for x in items if x.get(key) and x["true"] in ("forehand", "backhand", "serve")})
            block["by_" + ("stroke_type" if key == "true" else key)] = {g: summarise([x for x in items if x.get(key) == g]) for g in groups}
        out[split] = block
    return out


# --- running the pipeline on a clip -------------------------------------------------------


class CachingDetector:
    """Wraps the real detector and stores its raw per-frame output, so re-scoring after a logic change does not re-run the pose model."""

    def __init__(self, inner: Detector, cache_dir: Path, key: str) -> None:
        self.inner, self.dir, self.key = inner, cache_dir, key
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _load(self, p: Path) -> tuple[list[int], list[FramePoses]]:
        d = np.load(p)
        counts, lm, world = d["counts"], d["lm"], d["world"]
        off = np.concatenate([[0], np.cumsum(counts)])
        return d["idx"].tolist(), [FramePoses(lm[off[i] : off[i + 1]], world[off[i] : off[i + 1]]) for i in range(len(counts))]

    def _save(self, p: Path, idx: list[int], dets: list[FramePoses]) -> None:
        lm = np.concatenate([d.lm for d in dets]) if dets else np.zeros((0, 33, 4), np.float32)
        world = np.concatenate([d.world for d in dets]) if dets else np.zeros((0, 33, 3), np.float32)
        np.savez_compressed(p, idx=np.asarray(idx), counts=np.asarray([d.lm.shape[0] for d in dets]), lm=lm, world=world)

    def detect_video(self, path: Path, stride: int, progress, **kw):
        tag = hashlib.sha256(json.dumps(kw, sort_keys=True, default=str).encode()).hexdigest()[:8] if kw else "full"
        p = self.dir / f"{self.key}-s{stride}-{tag}.npz"
        if p.exists():
            return self._load(p)
        idx, dets = self.inner.detect_video(path, stride, progress, **kw)
        self._save(p, idx, dets)
        return idx, dets

    def detect_image(self, frame):
        return self.inner.detect_image(frame)

    def detect_crops(self, path: Path, frame_idx, boxes, progress):
        tag = hashlib.sha256(json.dumps([frame_idx, boxes]).encode()).hexdigest()[:10]
        p = self.dir / f"{self.key}-crops-{tag}.npz"
        if p.exists():
            _, dets = self._load(p)
            return [d if d.lm.shape[0] else None for d in dets]
        out = self.inner.detect_crops(path, frame_idx, boxes, progress)
        empty = FramePoses(np.zeros((0, 33, 4), np.float32), np.zeros((0, 33, 3), np.float32))
        self._save(p, list(frame_idx), [d if d is not None else empty for d in out])
        return out


def code_version() -> str:
    h = hashlib.sha256()
    for f in sorted((Path(__file__).resolve().parents[1] / "pipeline").glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def run_clip(clip: Clip, root: Path, detector_factory, options: Options, settings, log=print) -> dict:
    """Normalise, pick the player and analyse one clip with the same functions the web app's worker calls."""
    src = root / clip.file
    if not src.exists():
        return {"status": "missing_file", "message": f"{clip.file} not found; download it from {clip.source.get('page')}"}
    work = root / "cache" / "work" / clip.id
    key = hashlib.sha256(f"{clip.id}|{clip.t0}|{clip.t1}|{src.stat().st_size}|{settings.pose_model}".encode()).hexdigest()[:12]
    meta_p = work / f"prepared-{key}.json"
    try:
        if not meta_p.exists():
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            upload = work / f"upload{src.suffix}"
            if clip.t0 is not None or clip.t1 is not None:
                upload = work / "upload.mkv"
                cmd = [ffmpeg_exe(), "-loglevel", "error", "-y", "-i", str(src), "-ss", str(clip.t0 or 0)] + (["-to", str(clip.t1)] if clip.t1 is not None else []) + ["-an", "-c:v", "libx264", "-crf", "12", "-preset", "veryfast", str(upload)]
                subprocess.run(cmd, check=True, capture_output=True)
            else:
                shutil.copy(src, upload)
            prep = run.prepare(work, upload, detector_factory(), max_duration_s=600, min_duration_s=0.5, progress=lambda *_: None)
            meta_p.write_text(json.dumps(prep))
        prep = json.loads(meta_p.read_text())
        people = prep["candidates"]["people"]
        sel = 0
        if clip.player_hint and len(people) > 1:
            sel = int(np.argmin([np.hypot(p["centre"][0] - clip.player_hint[0], p["centre"][1] - clip.player_hint[1]) for p in people]))
        det = CachingDetector(detector_factory(), root / "cache" / "pose", key)
        t0 = time.time()
        report = run.analyze(work, det, meta=prep["meta"], quality=prep["quality"], candidates=prep["candidates"], selected=sel,
                             intake={"handedness": clip.handedness, "backhand_style": clip.backhand_style}, max_fps=settings.analysis_max_fps, progress=lambda *_: None, options=options)  # fmt: skip
    except PipelineError as e:
        return {"status": "failed", "code": e.code, "message": e.message}
    preds = [{"type": s["type"], "time": s["peak"], "start": s["start"], "end": s["end"], "timing_source": (s.get("timing") or {}).get("source", "peak_hand_speed"),
              "source": s["source"], "label_confidence": s["label_confidence"]} for s in report["strokes"]]  # fmt: skip
    tq = report["quality"]["tracking"]
    return {"status": "ok", "duration_s": prep["meta"]["duration_s"], "fps": prep["meta"]["fps"], "resolution": f"{prep['meta']['width']}x{prep['meta']['height']}", "people": len(people),
            "tracked_frac": tq["tracked_frac"], "camera_view_detected": tq["camera_view"], "preds": preds, "analysis_s": round(time.time() - t0, 1),
            "extras": {k: report.get(k) for k in ("ball", "court", "tracking_events") if report.get(k) is not None}}  # fmt: skip


def run_all(root: Path, options: Options, name: str, only: list[str] | None = None, detector_factory=None, log=print) -> dict:
    settings = load_settings()
    if detector_factory is None:
        from ..pipeline.pose import MediaPipeDetector

        detector_factory = lambda: MediaPipeDetector(settings.model_path)  # noqa: E731
    per_clip = []
    for clip in load_manifest(root):
        if only and clip.id not in only:
            continue
        res = run_clip(clip, root, detector_factory, options, settings, log)
        labels = load_labels(clip, root)
        entry = {"clip": clip.id, "split": clip.split, "status": res["status"], "labelled": labels is not None, "labeller": labels and labels.labeller, "expert_labels": bool(labels and labels.expert),
                 "result": {k: v for k, v in res.items() if k != "preds"}, "preds": res.get("preds", []), "items": []}  # fmt: skip
        if labels is not None:
            entry["items"] = score_clip(labels, res.get("preds", []))  # a failed clip scores every labelled stroke as missed
        s = summarise(entry["items"])
        log(f"{clip.id:<16} {clip.split:<9} {res['status']:<7} labelled={s['labelled']:<3} detected={s['detected']:<3} correct={s['correct_label']:<3} false={s['false_detections']:<3} {res.get('message', '')}")
        per_clip.append(entry)
    labelled = [c for c in per_clip if c["labelled"]]
    out = {
        "name": name, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "options": options.to_dict(), "code_version": code_version(), "pose_model": settings.pose_model,
        "match_tolerance_s": MATCH_TOLERANCE_S, "set": {"clips": len(per_clip), "labelled_clips": len(labelled), "expert_labelled_clips": sum(c["expert_labels"] for c in labelled),
        "footage_s": round(sum(c["result"].get("duration_s") or 0 for c in per_clip), 1)},
        "metrics": aggregate(labelled), "per_clip": per_clip,
    }  # fmt: skip
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}_{name}.json").write_text(json.dumps(out, indent=1))
    return out


# --- reporting ----------------------------------------------------------------------------


def _fmt(m: dict) -> str:
    pct = lambda v: "  n/a" if v is None else f"{v:5.0%}"  # noqa: E731
    t = m["timing_vs_contact"]
    return (f"n={m['labelled']:<3} recall {pct(m['detection_recall'])}  precision {pct(m['detection_precision'])}  correct {pct(m['correct_end_to_end'])}  "
            f"wrong-label {m['wrong_label']:<2} unlabelled {m['left_unlabelled']:<2} missed {m['missed']:<2} false {m['false_detections']:<2} "
            + (f" |t-contact| med {t['median_abs_s']:.2f}s (n={t['n']})" if t else ""))  # fmt: skip


def print_run(r: dict, log=print) -> None:
    s = r["set"]
    log(f"\nRun '{r['name']}'  {r['created_at']}  code {r['code_version']}  options {json.dumps(r['options'])}")
    log(f"Set: {s['clips']} clips, {s['labelled_clips']} labelled ({s['expert_labelled_clips']} by an expert), {s['footage_s']} s of footage. Tolerance ±{r['match_tolerance_s']} s.")
    if s["labelled_clips"] and not s["expert_labelled_clips"]:
        log("NOTE: every label in this set was made by a non-expert. Treat these numbers as a development aid, not as validated accuracy.")
    for split in SPLITS:
        b = r["metrics"][split]
        log(f"\n[{split}] {b['clips']} clips\n  overall            {_fmt(b['overall'])}")
        for dim in ("by_stroke_type", "by_camera_view", "by_handedness"):
            for g, m in b[dim].items():
                log(f"  {dim[3:]:<12}{g:<9} {_fmt(m)}")
        log("  confusion (rows = labelled, cols = " + ", ".join((*PRED_TYPES, "missed")) + ")")
        for t, row in b["overall"]["confusion"].items():
            log(f"    {t:<9} " + " ".join(f"{row[p]:>4}" for p in (*PRED_TYPES, "missed")))


def load_run(root: Path, name: str) -> dict:
    files = sorted((root / "runs").glob(f"*_{name}.json"))
    if not files:
        raise SystemExit(f"No stored run named {name!r} in {root / 'runs'}")
    return json.loads(files[-1].read_text())


def compare(a: dict, b: dict, log=print) -> None:
    log(f"\nBefore: '{a['name']}' ({a['created_at']})   After: '{b['name']}' ({b['created_at']})")
    for split in SPLITS:
        ma, mb = a["metrics"][split]["overall"], b["metrics"][split]["overall"]
        log(f"[{split}] labelled strokes: {ma['labelled']} -> {mb['labelled']}")
        for k in ("detected", "correct_label", "wrong_label", "left_unlabelled", "missed", "false_detections", "detection_recall", "detection_precision", "correct_end_to_end"):
            log(f"  {k:<22} {ma[k]} -> {mb[k]}")
        for k in ("timing_vs_contact", "timing_vs_swing_peak"):
            ta, tb = ma[k], mb[k]
            log(f"  {k:<22} " + " -> ".join("n/a" if t is None else f"median |err| {t['median_abs_s']}s, mean signed {t['mean_signed_s']}s (n={t['n']})" for t in (ta, tb)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--name", default="run")
    ap.add_argument("--clips", nargs="*", help="only these clip ids")
    ap.add_argument("--baseline", action="store_true", help="first-build behaviour: every later analysis stage off")
    for f in fields(Options):
        ap.add_argument(f"--{f.name.replace('_', '-')}", default=None, help=f"override option (default {f.default})")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    ap.add_argument("--list", action="store_true", help="list stored runs")
    ap.add_argument("--show", metavar="NAME")
    ap.add_argument("--clear-cache", action="store_true", help="delete cached pose-model outputs and normalised clips")
    a = ap.parse_args(argv)
    if a.clear_cache:
        shutil.rmtree(a.root / "cache", ignore_errors=True)
        print("Cache cleared.")
        return 0
    if a.list:
        for f in sorted((a.root / "runs").glob("*.json")):
            r = json.loads(f.read_text())
            print(f"{f.name:<48} " + "  ".join(f"{sp}: {_fmt(r['metrics'][sp]['overall'])}" for sp in SPLITS))
        return 0
    if a.show:
        print_run(load_run(a.root, a.show))
        return 0
    if a.compare:
        compare(load_run(a.root, a.compare[0]), load_run(a.root, a.compare[1]))
        return 0
    base = BASELINE if a.baseline else Options()
    over = {}
    for f in fields(Options):
        v = getattr(a, f.name)
        if v is not None:
            over[f.name] = v if f.type == "str" else v.lower() in ("1", "true", "yes", "on")
    opts = Options(**{**base.to_dict(), **over})
    if not load_manifest(a.root):
        print(f"No clips in {a.root / 'manifest.json'}. See evalset/README.md for how to add footage and labels.")
        return 1
    print_run(run_all(a.root, opts, a.name, a.clips))
    return 0


if __name__ == "__main__":
    sys.exit(main())
