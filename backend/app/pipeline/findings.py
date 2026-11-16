"""Turn per-stroke measurements into findings and a report.

All wording here is fixed text filled with measured values. No language model is involved,
so nothing can be reported that the rules below did not compute.

The numeric cut-offs are COACHING HEURISTICS chosen by the developers to decide when a
pattern is worth mentioning. They are not safety thresholds and have not been validated.
"""
from __future__ import annotations

from datetime import datetime, timezone
from statistics import median

import numpy as np

from .drills import for_rule
from .evidence import refs
from .segment import STROKE_TYPES, TIMING_LABEL, Stroke
from .signals import L_ANK, L_KNEE, R_ANK, R_KNEE, PoseTrack

REPORT_VERSION = 1

DISCLAIMER = (
    "This is a technique-analysis and education tool. It does not diagnose injuries, measure joint forces, or predict "
    "whether you will be injured. It has not been clinically validated. Measurements come from a single ordinary video "
    "and are approximate."
)

PAIN_GUIDANCE = (
    "You told us you currently have pain or discomfort. This app cannot assess pain or tell whether it is related to "
    "anything seen in the video, and nothing here is a rehabilitation plan. Don't push through pain to practise a "
    "technique change. If the pain is persistent or getting worse, started with a specific incident, affects sleep or "
    "daily activities, or comes with swelling, numbness, tingling or weakness, see a doctor or physiotherapist before "
    "you continue playing."
)

# Things this version does not assess, shown explicitly so their absence is not read as "fine".
# docs/VALIDATION.md records why each one is (still) here after ball, court and re-acquisition tracking were added.
NOT_ASSESSED = [
    {"topic": "Wrist position and grip", "reason": "Revisited with player-cropped pose inference: hand landmarks are still a few pixels across and usually hidden by the racket at normal filming distances, "
     "and there is no reference data to check them against. Grip pressure cannot be seen on video at all."},
    {"topic": "Sequencing of legs, hips, trunk and arm", "reason": "Revisited: the time gaps between body segments (a few hundredths of a second) are about one frame at 25-30 fps, and single-camera 3D estimates "
     "are too noisy to order them. It would need high-frame-rate footage and reference measurements to validate."},
    {"topic": "Footwork: split step, recovery position and changes of direction", "reason": "Built but not validated. Position on court needs at least five court lines in view, which none of the evaluation clips had, "
     "and no footage has been labelled for split steps. Where a recording does allow it, these are shown as unvalidated observations, not findings."},
    {"topic": "Racket path and face angle", "reason": "A general object detector can box the racket in many frames, but not its orientation; nothing is concluded from it."},
    {"topic": "Ankle and foot mechanics", "reason": "Foot landmarks are not accurate enough at this scale."},
    {"topic": "Volleys, overheads, slices and returns as separate strokes", "reason": "Only forehands, backhands and serves are labelled. An overhead may be labelled as a serve."},
    {"topic": "Joint loads, muscle activity, pain or tissue damage", "reason": "None of these can be inferred from video."},
]

CONTACT_OFF = "Ball tracking was switched off. Timestamps mark peak hitting-hand speed, which is near contact but is not contact."
CONTACT_ON = ("Built but not validated, so still listed here. The ball is tracked and, where its path clearly changes at the racket, the stroke is timed at contact and a 2D contact point is shown as an unvalidated "
              "observation. On the footage held back for testing this was found for only 2 of 14 strokes and did not make timing more accurate than peak hand speed; in swings with no ball it was wrongly claimed "
              "for 1 of 5. No rule or coaching cue uses it.")

GOAL_STROKES = {"serve": {"serve"}, "forehand": {"forehand"}, "backhand": {"backhand"}}


# --- detection confidence ---------------------------------------------------------------


def confidence(visibility: float, tracked: float, view: float, n_support: int, method: str) -> dict:
    """Detection confidence = the weakest of the tracking factors, scaled by how many strokes agree.

    This describes how well the pattern could be SEEN. It is not a probability of injury
    or of anything else about the player.
    """
    method_cap = 0.75 if "estimated_3d" in method else 1.0
    support = {1: 0.6, 2: 0.8}.get(n_support, 1.0)
    base = min(visibility, tracked, view, method_cap)
    score = round(base * (0.7 + 0.3 * support), 2)
    level = "high" if score >= 0.75 else "moderate" if score >= 0.55 else "low" if score >= 0.4 else "insufficient"
    factors = [
        {"name": "Landmark visibility", "value": round(visibility, 2), "note": "Average of the pose model's own visibility estimate for the joints used."},
        {"name": "Tracking continuity", "value": round(tracked, 2), "note": "Share of frames in these strokes where the player was tracked."},
        {"name": "Camera view suitability", "value": round(view, 2), "note": "How well the body orientation suited this measurement."},
        {"name": "Measurement method", "value": method_cap, "note": "Capped when the value relies on single-camera 3D estimates."},
        {"name": "Supporting strokes", "value": n_support, "note": "One stroke is weaker support than a repeated pattern."},
    ]
    return {
        "level": level,
        "score": score,
        "factors": factors,
        "explanation": "Set by the weakest of visibility, tracking, view and method, then reduced when few strokes show the pattern. "
        "It reflects how clearly the pattern was seen, not how likely an injury is.",
    }


# --- rules ------------------------------------------------------------------------------


def _val(m: dict, key: str):
    v = m["values"].get(key)
    return v if v and v["assessable"] else None


def _r5(x: float) -> int:
    return int(round(x / 5.0) * 5)


def _knee_bend(threshold: float):
    def check(m: dict):
        kd, kn = _val(m, "knee_bend_dom"), _val(m, "knee_bend_non")
        if not kd or not kn:
            return None
        v = max(kd["value"], kn["value"])
        side = _side_view_factor(m)
        return {"fired": v < threshold, "value": v, "time": kd["time"], "vis": min(kd["visibility"], kn["visibility"]), "view": side}

    return check


def _side_view_factor(m: dict) -> float:
    az = m["quality"].get("pelvis_azimuth_median")
    return 0.8 if az is None else float(np.clip(0.6 + az / 150.0, 0.6, 1.0))


def _follow_through(m: dict, stroke_type: str):
    ft = _val(m, "follow_through_travel")
    if not ft:
        return None
    limit = 0.15 if stroke_type == "forehand" else 0.3
    return {"fired": ft["value"] < limit, "value": ft["value"], "time": ft["time"], "vis": ft["visibility"], "view": 0.85}


def _serve_lean(m: dict):
    tl = _val(m, "trunk_lean")
    if not tl or tl["extra"].get("pelvis_azimuth") is None or tl["extra"]["pelvis_azimuth"] > 35:
        return None
    return {"fired": tl["value"] > 35, "value": tl["value"], "time": tl["time"], "vis": tl["visibility"], "view": 1.0 - tl["extra"]["pelvis_azimuth"] / 100.0}


def _ground_lean(m: dict):
    tl = _val(m, "trunk_lean")
    if not tl:
        return None
    return {"fired": tl["value"] > 30, "value": tl["value"], "time": tl["time"], "vis": tl["visibility"], "view": 0.8}


def _knee_align(m: dict):
    best = None
    for k in ("knee_alignment_dom", "knee_alignment_non"):
        v = _val(m, k)
        if v and (best is None or v["value"] > best["value"]):
            best = v
    if not best:
        return None
    return {"fired": best["value"] > 0.08, "value": best["value"], "time": best["time"], "vis": best["visibility"], "view": 1.0 - best["extra"]["pelvis_azimuth"] / 100.0}


RULES: list[dict] = [
    {
        "id": "serve_knee_bend",
        "strokes": ("serve",),
        "check": lambda m, s: _knee_bend(30)(m),
        "method": "image_plane+estimated_3d",
        "region": "knee",
        "category": "health_related",
        "title": "Limited knee bend before the serve",
        "observe": lambda med, k, n: f"In {k} of {n} assessable serves, neither knee bent more than about {_r5(med)}° while loading (0° is a straight leg).",
        "clear": lambda med, n: f"Knee bend while loading was about {_r5(med)}° across {n} assessable serves, above the level at which this tool flags it.",
        "interpretation": "A shallow knee bend usually means less upward drive from the legs, so more of the serve's speed has to come from the trunk and arm. "
        "Whether that matters for you cannot be told from video.",
        "evidence": {"strength": "Limited - one group-level study in skilled players", "refs": ("elliott2003",)},
        "cue": "Let both knees bend as the toss goes up, then drive up to the ball.",
        "cue_beginner": "As you toss, sit down a little, then push up tall to hit.",
        "recording": "Film from the side at hip height so the knee angle is seen square-on.",
        "limits": [
            "Knee angle is taken as the larger of the 2D image angle and a single-camera 3D estimate; a leg pointing toward the camera looks straighter than it is.",
            "The 30° cut-off is a coaching heuristic, not a safety threshold.",
        ],
    },
    {
        "id": "serve_trunk_lean",
        "strokes": ("serve",),
        "check": lambda m, s: _serve_lean(m),
        "method": "image_plane",
        "region": "lower_back",
        "category": "health_related",
        "title": "Pronounced sideways trunk tilt in the serve",
        "unable": "Sideways trunk tilt needs a view from in front of or behind the player; in this recording the hips were turned too far from the camera.",
        "observe": lambda med, k, n: f"In {k} of {n} serves filmed from in front or behind, the trunk tilted about {_r5(med)}° from vertical before peak hand speed.",
        "clear": lambda med, n: f"Trunk tilt before peak hand speed was about {_r5(med)}° across {n} serves filmed from in front or behind.",
        "interpretation": "Sideways lean is a normal part of serving and often helps reach. Large lean is one of several things a coach or clinician may look at "
        "when a player reports lower-back discomfort, but tilt in a 2D video is not a measure of load on the back.",
        "evidence": {"strength": "Limited - one small study of elite adolescents; does not show cause", "refs": ("campbell2013",)},
        "cue": "Check the toss: one that drifts far to the non-hitting side pulls the trunk over. Aim to toss slightly in front and reach up.",
        "cue_beginner": "Toss the ball a little in front of you and reach up tall.",
        "recording": "Film from directly behind the baseline so sideways lean is seen square-on.",
        "limits": [
            "Measured in the image plane; forward lean and sideways lean cannot be separated from one camera.",
            "The 35° cut-off is a heuristic for when to mention it, not a limit of what is acceptable.",
        ],
    },
    {
        "id": "knee_alignment",
        "strokes": STROKE_TYPES,
        "check": lambda m, s: _knee_align(m),
        "method": "image_plane",
        "region": "knee",
        "category": "health_related",
        "title": "Knee tracks inward of the foot line",
        "observe": lambda med, k, n: f"In {k} of {n} strokes seen from in front or behind, the bent knee sat inward of the line from hip to ankle by about {med * 100:.0f}% of leg length.",
        "clear": lambda med, n: f"In {n} strokes seen from in front or behind, the bent knee stayed close to the hip-ankle line.",
        "interpretation": "A knee that appears to drop inward can simply be the hip rotating, which is normal in tennis, or it can reflect how the leg is controlling load. "
        "A 2D video cannot tell these apart.",
        "evidence": {"strength": "Mixed - from jump-landing tests in other sports; not established for tennis or for 2D video", "refs": ("hewett2005", "krosshaug2016")},
        "cue": "As you load, feel the knee point the same way as the toes.",
        "cue_beginner": "Bend your knee toward your toes, not toward the other knee.",
        "recording": "Film from directly in front or behind, with feet in frame, to check this again.",
        "limits": [
            "Only assessed when the hips face the camera or directly away; even then hip rotation can mimic inward knee movement.",
            "The 8% cut-off is a heuristic.",
        ],
    },
    {
        "id": "ground_knee_bend",
        "strokes": ("forehand", "backhand"),
        "check": lambda m, s: _knee_bend(25)(m),
        "method": "image_plane+estimated_3d",
        "region": "legs",
        "category": "technique",
        "title": "Upright legs while loading",
        "observe": lambda med, k, n: f"In {k} of {n} assessable strokes, neither knee bent more than about {_r5(med)}° before the swing.",
        "clear": lambda med, n: f"Knee bend before the swing was about {_r5(med)}° across {n} assessable strokes.",
        "cue": "Get lower before the ball arrives: bend the knees as you turn, then push up and through the shot.",
        "cue_beginner": "Bend your knees as you turn to the ball.",
        "recording": "A side-on view at hip height shows leg loading best.",
        "limits": ["Ball height matters: a high ball needs less knee bend. The ball is not tracked, so this is not accounted for.", "The 25° cut-off is a coaching heuristic."],
    },
    {
        "id": "short_follow_through",
        "strokes": ("forehand", "backhand"),
        "check": lambda m, s: _follow_through(m, s),
        "method": "estimated_3d",
        "region": "arm",
        "category": "technique",
        "title": "Short follow-through",
        "observe": lambda med, k, n: f"In {k} of {n} assessable strokes, the hitting hand stopped before travelling clearly past the middle of the chest.",
        "clear": lambda med, n: f"The hitting hand finished well across the body in {n} assessable strokes.",
        "cue": "Let the swing run its course: finish with the hand over the opposite shoulder (forehand) or out and up on the hitting side (backhand).",
        "cue_beginner": "Keep swinging after you hit - finish high.",
        "recording": "Keep the whole arm in frame through the finish.",
        "limits": ["Uses single-camera 3D estimates of hand and shoulder position.", "Slices, blocks and defensive shots have short finishes by design; shot intent is not known."],
    },
    {
        "id": "ground_trunk_lean",
        "strokes": ("forehand", "backhand"),
        "check": lambda m, s: _ground_lean(m),
        "method": "image_plane",
        "region": "trunk",
        "category": "technique",
        "title": "Upper body tilted well off vertical at the hit",
        "observe": lambda med, k, n: f"In {k} of {n} assessable strokes, the trunk was tilted about {_r5(med)}° from vertical at peak hand speed.",
        "clear": lambda med, n: f"The trunk stayed within about {_r5(med)}° of vertical at peak hand speed across {n} strokes.",
        "cue": "Move the feet to the ball so you can hit with your chest up, rather than reaching with the upper body.",
        "cue_beginner": "Take small steps to get close to the ball and stay tall.",
        "recording": "",
        "limits": ["Image-plane tilt; wide or low balls legitimately need more lean.", "The 30° cut-off is a coaching heuristic."],
    },
]


def _evaluate_rule(rule: dict, strokes: list[Stroke], measures: dict[str, dict], stroke_type: str, intake: dict) -> dict | None:
    group = [s for s in strokes if s.type == stroke_type]
    if not group:
        return None
    results = []
    for s in group:
        m = measures[s.id]
        r = rule["check"](m, stroke_type) if m["quality"]["assessable"] else None
        if r:
            results.append((s, m, r))
    beginner = intake.get("skill_level") == "beginner"
    base = {
        "id": f"{rule['id']}:{stroke_type}",
        "rule": rule["id"],
        "stroke_type": stroke_type,
        "body_region": rule["region"],
        "category": rule["category"],
        "title": rule["title"],
        "interpretation": None,
        "evidence": None,
        "action": {"cue": None, "recording": rule.get("recording") or None, "professional": None},
        "limitations": rule["limits"],
        "occurrences": [],
        "support": {"fired": 0, "assessed": len(results), "total": len(group)},
        "recurring": False,
        "confidence": None,
        "user_context": None,
        "drills": [],
    }
    if not results:
        reasons = sorted({v["reason"] for s in group for v in measures[s.id]["values"].values() if not v["assessable"] and v["key"].split("_")[0] in rule["id"]})
        return {**base, "status": "unable_to_assess",
                "observation": "Could not be assessed in this recording. " + (reasons[0] if reasons else rule.get("unable", "The landmarks or camera view needed were not available.")),
                "action": {"cue": None, "recording": rule.get("recording") or "Re-record with the whole body in frame and good light.", "professional": None}}  # fmt: skip
    fired = [(s, m, r) for s, m, r in results if r["fired"]]
    used = fired or results
    conf = confidence(
        visibility=float(np.mean([r["vis"] for _, _, r in used])),
        tracked=float(np.mean([m["quality"]["tracked_frac"] for _, m, _ in used])),
        view=float(np.mean([r["view"] for _, _, r in used])),
        n_support=len(used),
        method=rule["method"],
    )
    med = median(r["value"] for _, _, r in used)
    base["confidence"] = conf
    base["support"]["fired"] = len(fired)
    if not fired:
        text = rule["clear"](med, len(results))
        if len(results) == 1:
            text = text.replace("1 assessable serves", "1 assessable serve").replace("1 assessable strokes", "1 assessable stroke").replace("1 strokes", "1 stroke").replace("1 serves", "1 serve")
        return {**base, "status": "not_observed", "observation": text}
    if conf["level"] == "insufficient":
        return {**base, "status": "unable_to_assess", "confidence": conf,
                "observation": "A possible pattern was seen but tracking was too weak to report it. See the confidence factors for what limited it."}  # fmt: skip
    base["occurrences"] = [{"stroke_id": s.id, "start": round(s.start, 2), "end": round(s.end, 2), "time": r["time"], "value": round(r["value"], 2)} for s, _, r in fired]
    base["recurring"] = len(fired) >= 3 and len(fired) / len(results) >= 0.6
    base["action"]["cue"] = rule["cue_beginner"] if beginner else rule["cue"]
    base["drills"] = for_rule(rule["id"])
    if rule["category"] == "health_related":
        base["interpretation"] = rule["interpretation"]
        base["evidence"] = {"strength": rule["evidence"]["strength"], "refs": refs(*rule["evidence"]["refs"])}
        base["action"]["professional"] = "If you want this looked at properly, a coach can review it on court; a physiotherapist can assess it if you have symptoms."
        regions = set((intake.get("discomfort") or {}).get("regions") or [])
        related = {"knee": {"knee"}, "lower_back": {"lower_back"}}.get(rule["region"], set())
        if rule["id"] == "serve_knee_bend":
            related = {"shoulder", "elbow", "knee"}
        if (intake.get("discomfort") or {}).get("current") and regions & related:
            base["user_context"] = "You reported discomfort in a related area. The video cannot show whether this pattern has anything to do with it; mention both to whoever assesses you."
    return {**base, "status": "observed", "observation": rule["observe"](med, len(fired), len(results))}


def _trend_findings(strokes: list[Stroke], measures: dict[str, dict]) -> list[dict]:
    out = []
    for st in STROKE_TYPES:
        vals = [(s, _val(measures[s.id], "peak_hand_speed")) for s in strokes if s.type == st]
        vals = [(s, v["value"]) for s, v in vals if v]
        if len(vals) < 6:
            continue
        third = len(vals) // 3
        first, last = median(v for _, v in vals[:third]), median(v for _, v in vals[-third:])
        change = (last - first) / first if first else 0.0
        if abs(change) < 0.15:
            continue
        out.append({
            "id": f"speed_change:{st}", "rule": "speed_change", "status": "observed", "stroke_type": st, "body_region": "whole_body", "category": "technique",
            "title": f"Hitting-hand speed {'dropped' if change < 0 else 'rose'} over the recording",
            "observation": f"Median peak hand speed on {st}s was about {abs(change):.0%} {'lower' if change < 0 else 'higher'} in the last third of the recording than in the first third ({len(vals)} strokes).",
            "interpretation": "This could reflect tiredness, a change of drill or intent, easier or harder balls, or the player moving nearer to or further from the camera. The video cannot tell which.",
            "evidence": None,
            "action": {"cue": None, "recording": "Keep the camera fixed and note what the drill was, so a change can be put in context.", "professional": None},
            "limitations": ["Hand speed is an image-plane quantity scaled by body size; it is not racket or ball speed."],
            "occurrences": [{"stroke_id": s.id, "start": round(s.start, 2), "end": round(s.end, 2), "time": round(s.peak, 2), "value": round(v, 1)} for s, v in vals[-third:]],
            "support": {"fired": third, "assessed": len(vals), "total": len(vals)}, "recurring": False,
            "confidence": confidence(0.8, 1.0, 0.8, len(vals), "relative"), "user_context": None,
        })  # fmt: skip
    return out


def _stroke_summary(strokes: list[Stroke], measures: dict[str, dict]) -> dict:
    keys = ["peak_hand_speed", "knee_bend_dom", "knee_bend_non", "elbow_angle_peak", "trunk_lean", "follow_through_travel", "reach_height", "shoulder_turn_range",
            "contact_height", "contact_distance", "contact_in_front"]
    out = {}
    for st in (*STROKE_TYPES, "unclassified"):
        group = [s for s in strokes if s.type == st]
        if not group:
            continue
        metrics = {}
        for k in keys:
            vs = [v["value"] for s in group if (v := _val(measures[s.id], k))]
            if vs:
                any_m = next(v for s in group if (v := _val(measures[s.id], k)))
                metrics[k] = {"label": any_m["label"], "unit": any_m["unit"], "method": any_m["method"], "n": len(vs),
                              "median": round(median(vs), 1), "min": round(min(vs), 1), "max": round(max(vs), 1)}  # fmt: skip
        out[st] = {"count": len(group), "assessable": sum(measures[s.id]["quality"]["assessable"] for s in group), "metrics": metrics}
    return out


def tracking_quality(track: PoseTrack) -> dict:
    tracked = float(track.tracked.mean())
    ys = track.xy[track.tracked][:, :, 1]
    height_frac = float(np.nanmedian((np.nanmax(ys, axis=1) - np.nanmin(ys, axis=1)) / track.height)) if ys.size else 0.0
    legs_vis = float(track.vis[track.tracked][:, [L_KNEE, R_KNEE, L_ANK, R_ANK]].mean()) if ys.size else 0.0
    az = track.pelvis_azimuth
    az_med = float(np.nanmedian(az)) if np.isfinite(az).any() else None
    view = "unknown" if az_med is None else "mostly from in front of or behind the player" if az_med < 30 else "mostly side-on" if az_med > 60 else "at an angle, or the player turns a lot"
    issues = []
    if tracked < 0.9:
        issues.append({"code": "tracking_gaps", "message": f"The player was tracked in {tracked:.0%} of frames. Strokes that fall in the gaps are not assessed.",
                       "suggestion": "Keep the player in frame and unobstructed for the whole clip."})  # fmt: skip
    if height_frac < 0.25:
        issues.append({"code": "player_small", "message": f"The player fills only about {height_frac:.0%} of the frame height, which limits joint accuracy.",
                       "suggestion": "Move the camera closer or zoom optically so the player fills at least a third of the frame."})  # fmt: skip
    if legs_vis < 0.6:
        issues.append({"code": "legs_hidden", "message": "The knees and ankles were often hidden or out of frame, so leg findings are limited.",
                       "suggestion": "Frame the shot so the feet stay visible."})  # fmt: skip
    return {"tracked_frac": round(tracked, 2), "player_height_frac": round(height_frac, 2), "leg_visibility": round(legs_vis, 2),
            "pelvis_azimuth_median": None if az_med is None else round(az_med), "camera_view": view, "issues": issues}  # fmt: skip


def _contact_summary(strokes: list[Stroke], ball_on: bool) -> dict:
    hits = [s for s in strokes]
    found = [s for s in hits if s.timing_source == "ball_contact"]
    if not ball_on:
        return {"status": "off", "found": 0, "strokes": len(hits), "note": "Ball tracking was switched off for this analysis. Timestamps mark peak hitting-hand speed (serves: highest hand position), which is near contact but is not contact."}
    if not hits:
        return {"status": "none", "found": 0, "strokes": 0, "note": ""}
    if not found:
        return {"status": "unable", "found": 0, "strokes": len(hits),
                "note": "No ball was tracked clearly enough near any stroke, so contact time and contact point could not be assessed. Timestamps mark peak hitting-hand speed (serves: highest hand position) instead. "
                        "A small, fast or blurred ball, a ball the same colour as the background, or a camera that moves a lot all cause this."}  # fmt: skip
    return {"status": "ok" if len(found) == len(hits) else "partial", "found": len(found), "strokes": len(hits),
            "note": f"Ball contact was found for {len(found)} of {len(hits)} strokes; those are timed at contact. The rest are timed at peak hitting-hand speed (serves: highest hand position), which is near contact but is not contact. "
                    "Contact is located in the 2D picture from the ball's path; it has been checked against hand-marked contact frames on a small evaluation set only (docs/VALIDATION.md)."}  # fmt: skip


def build_report(track: PoseTrack, strokes: list[Stroke], measures: dict[str, dict], intake: dict, video_quality: dict, meta: dict, extras: dict | None = None) -> dict:
    extras = extras or {}
    findings: list[dict] = []
    for st in STROKE_TYPES:
        for rule in RULES:
            if st in rule["strokes"]:
                f = _evaluate_rule(rule, strokes, measures, st, intake)
                if f:
                    findings.append(f)
    findings += _trend_findings(strokes, measures)

    goals = set(intake.get("goals") or [])
    goal_strokes = set().union(*(GOAL_STROKES.get(g, set()) for g in goals)) if goals else set()
    observed = [f for f in findings if f["status"] == "observed" and f["action"]["cue"]]
    observed.sort(key=lambda f: (-int(f["recurring"]), -int(f["stroke_type"] in goal_strokes), -f["confidence"]["score"]))
    priorities = [{"finding_id": f["id"], "stroke_type": f["stroke_type"], "title": f["title"], "cue": f["action"]["cue"]} for f in observed[:3]]

    context_notes = []
    disc = intake.get("discomfort") or {}
    if disc.get("current"):
        context_notes.append({"kind": "pain", "title": "About the discomfort you reported", "body": PAIN_GUIDANCE, "refs": [],
                              "review_note": "General safety wording written by the developers; not yet reviewed by a clinician."})  # fmt: skip
    if intake.get("workload_change") == "increased":
        context_notes.append({
            "kind": "workload", "title": "You said your playing load went up recently",
            "body": "Sudden jumps in how much you play are widely discussed as a contributor to overuse problems, although how to measure that and how strong the link is are disputed. "
            "The video analysis does not use this information; it is noted here so a coach or clinician reading the report sees it.",
            "refs": refs("gabbett2016", "impellizzeri2020"), "review_note": "Not yet reviewed by a clinician.",
        })  # fmt: skip

    tq = tracking_quality(track)
    court = extras.get("court") or {"status": "off", "reason": "Court detection was switched off for this analysis."}
    if court.get("camera_view"):
        tq["camera_view_basis"] = "court lines"
        tq["camera_view_from_pose"], tq["camera_view"] = tq["camera_view"], court["camera_view"]
    else:
        tq["camera_view_basis"] = "body orientation (court lines were not found)" if court["status"] != "off" else "body orientation"
    events = extras.get("tracking_events")
    if events:
        waiting = [s for s in events["segments"] if s["status"] == "needs_confirmation" and s.get("decision") is None]
        if waiting:
            tq["issues"].append({"code": "confirm_player", "message": f"After a cut or a gap, someone who may be the selected player was found in {len(waiting)} stretch{'es' if len(waiting) != 1 else ''} of the video. "
                                 "Those parts are not analysed until you confirm it is the same player.", "suggestion": "Use “Check the player” above the report to confirm or reject each one."})  # fmt: skip
        if events["cuts"]:
            tq["issues"].append({"code": "scene_cuts", "message": f"This video contains {len(events['cuts'])} scene cut{'s' if len(events['cuts']) != 1 else ''}. The player is only followed across a cut when their appearance clearly matches.",
                                 "suggestion": "For the most reliable result, upload one continuous take."})  # fmt: skip
    counts = {st: sum(s.type == st for s in strokes) for st in (*STROKE_TYPES, "unclassified")}
    parts = [f"{counts[st]} {st}{'s' if counts[st] != 1 else ''}" for st in STROKE_TYPES if counts[st]]
    if counts["unclassified"]:
        parts.append(f"{counts['unclassified']} swing{'s' if counts['unclassified'] != 1 else ''} that could not be labelled")
    if strokes:
        headline = f"Found {len(strokes)} swing{'s' if len(strokes) != 1 else ''}: " + ", ".join(parts) + "."
    else:
        headline = "No forehands, backhands or serves were detected in this recording."
    n_obs = sum(f["status"] == "observed" for f in findings)
    n_un = sum(f["status"] == "unable_to_assess" for f in findings)
    summary = {
        "headline": headline,
        "detail": (f"{n_obs} pattern{'s' if n_obs != 1 else ''} worth a look, {n_un} check{'s' if n_un != 1 else ''} that could not be made from this recording. " if strokes else
                   "If the video does contain strokes, mark them yourself with “Add stroke” and they will be measured. ")
        + f"The camera view was {tq['camera_view']}.",
        "counts": counts,
        "no_strokes": not strokes,
    }  # fmt: skip

    return {
        "version": REPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER,
        "meta": meta,
        "handedness": intake.get("handedness", "right"),
        "quality": {"video": video_quality, "tracking": tq, "issues": video_quality.get("issues", []) + tq["issues"]},
        "summary": summary,
        "priorities": priorities,
        "findings": findings,
        "strokes": [{**s.to_dict(), "quality": measures[s.id]["quality"], "measurements": list(measures[s.id]["values"].values()),
                     "timing": {"source": s.timing_source, "label": TIMING_LABEL.get(s.timing_source, s.timing_source), "pose_time": None if s.pose_time is None else round(s.pose_time, 3),
                                "confidence": (s.contact or {}).get("confidence"), "basis": (s.contact or {}).get("basis")},
                     "phases": (extras.get("phases") or {}).get(s.id), "movement": (extras.get("movement") or {}).get(s.id)} for s in strokes],
        "stroke_summary": _stroke_summary(strokes, measures),
        "ball": _contact_summary(strokes, bool(extras.get("ball_on"))),
        "court": court,
        "movement": extras.get("movement_summary"),
        "tracking_events": events,
        "options": extras.get("options"),
        "not_assessed": [{"topic": "Contact point and timing", "reason": CONTACT_ON if extras.get("ball_on") else CONTACT_OFF}] + NOT_ASSESSED,
        "context_notes": context_notes,
        "supported_strokes": list(STROKE_TYPES),
        "validation_status": "Stroke detection, measurements and thresholds are unvalidated heuristics. Evidence summaries have not been reviewed by a clinician.",
    }
