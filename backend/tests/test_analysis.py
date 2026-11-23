"""Stroke detection, measurement and findings logic on synthetic poses with known ground truth."""
import json

import numpy as np
import pytest

from app.pipeline.findings import build_report, confidence
from app.pipeline.measure import measure_stroke
from app.pipeline.pose import FramePoses, track_player
from app.pipeline.segment import detect_strokes, manual_stroke
from app.pipeline.signals import L_ANK, L_HIP, L_KNEE, L_WR, R_ANK, R_HIP, R_KNEE, R_EL, R_WR, fill_short_gaps, smooth

from .synth import FakeDetector, make_track

EVENTS = [(2.0, "forehand"), (4.5, "backhand"), (7.5, "serve"), (10.0, "forehand"), (12.5, "backhand"), (15.5, "serve")]
Q = {"metrics": {}, "issues": []}
META = {"fps": 30.0, "width": 1280, "height": 720, "duration_s": 17.0, "n_frames": 510}


def report_for(track, intake=None, hand="right"):
    intake = {"handedness": hand, **(intake or {})}
    strokes = detect_strokes(track, hand, intake.get("backhand_style", "one_handed"))
    measures = {s.id: measure_stroke(track, s, hand) for s in strokes}
    return strokes, build_report(track, strokes, measures, intake, Q, META)


@pytest.mark.parametrize("yaw", [0, 45, 90, 135, 180, 225])
@pytest.mark.parametrize("left", [False, True])
@pytest.mark.parametrize("fps", [30.0, 60.0])
@pytest.mark.parametrize("two", [False, True])
def test_strokes_detected_and_labelled(yaw, left, fps, two):
    track = make_track(EVENTS, yaw=yaw, left_handed=left, fps=fps, noise=1.0, two_handed=two)
    strokes = detect_strokes(track, "left" if left else "right", "two_handed" if two else "one_handed")
    assert [s.type for s in strokes] == [k for _, k in EVENTS]
    for s, (t, _) in zip(strokes, EVENTS):
        assert abs(s.peak - t) < 0.3  # scripted keyframes ease in and out, so the speed peak sits between them
        assert s.start < s.peak < s.end


@pytest.mark.parametrize("yaw", [0, 45, 90, 135, 180, 225])
@pytest.mark.parametrize("left", [False, True])
def test_one_handed_backhand_with_high_finish_is_not_a_serve(yaw, left):
    # Held-out footage had every one-handed backhand labelled as a serve because the hand finished above the head.
    events = [(2.0, "forehand"), (4.5, "backhand_high_finish"), (7.5, "serve"), (10.0, "backhand_high_finish")]
    track = make_track(events, yaw=yaw, left_handed=left, noise=1.0)
    strokes = detect_strokes(track, "left" if left else "right", "one_handed")
    assert [s.type for s in strokes] == ["forehand", "backhand", "serve", "backhand"]
    for s, (t, _) in zip(strokes, events):
        assert abs(s.peak - t) < 0.3


@pytest.mark.parametrize("yaw", [0, 45, 90, 135, 180, 225])
@pytest.mark.parametrize("left", [False, True])
def test_shoulder_high_backhand_with_held_finish_is_one_backhand(yaw, left):
    # Tune clip p3_bh1_shadow: hand fastest near shoulder height, finish held above the head, then brought down.
    # Neither the high finish (not a serve) nor the arm coming back down (not a new stroke) is a stroke of its own.
    events = [(2.0, "forehand"), (4.5, "backhand_held_high_finish"), (8.0, "serve")]
    track = make_track(events, yaw=yaw, left_handed=left, noise=1.0)
    strokes = detect_strokes(track, "left" if left else "right", "one_handed")
    assert [s.type for s in strokes] == ["forehand", "backhand", "serve"]
    for s, (t, _) in zip(strokes, events):
        assert abs(s.peak - t) < 0.3


@pytest.mark.parametrize("yaw", [0, 45, 90, 135, 180, 225])
def test_looped_forehand_from_above_the_head_is_still_a_stroke(yaw):
    # rola_c: the take-back goes above the head, but the hand is rising again at the hit, so it is not a "return".
    events = [(2.0, "forehand_high_loop"), (4.5, "forehand")]
    strokes = detect_strokes(make_track(events, yaw=yaw, noise=1.0), "right", "one_handed")
    assert [s.type for s in strokes] == ["forehand", "forehand"]
    assert abs(strokes[0].peak - 2.0) < 0.3


@pytest.mark.parametrize("yaw", [0, 45, 135, 180])
def test_backhand_with_backswing_hidden_behind_the_body_is_still_a_backhand(yaw):
    # Tune clip p3_bh1_shadow: seen from the front, the hand is out of sight behind the body for most of the lead-up.
    track = make_track([(2.0, "backhand_early_backswing")], yaw=yaw, noise=1.0)
    hide = (track.t >= 2.0 - 1.05) & (track.t <= 2.0 - 0.4)
    track.vis[hide, R_WR] = 0.1
    s = detect_strokes(track, "right", "one_handed")
    assert [x.type for x in s] == ["backhand"] and s[0].label_confidence <= 0.45


@pytest.mark.parametrize("yaw", [0, 45, 135, 180])
def test_backhand_is_labelled_even_when_the_hidden_hand_is_misplaced(yaw):
    # Tune clip p12_rally_front: with a small player the pose model's low-visibility guess for the hidden hand
    # is often on the wrong side of the body, so it cannot be what decides; the swing seen before the hit does.
    track = make_track([(2.0, "backhand_early_backswing")], yaw=yaw, noise=1.0)
    hide = (track.t >= 2.0 - 1.05) & (track.t <= 2.0 - 0.4)
    track.vis[hide, R_WR] = 0.1
    w = track.world
    w[hide, R_WR] = w[hide, R_HIP] + (w[hide, R_HIP] - w[hide, L_HIP])  # one hip width out on the hitting-arm side
    s = detect_strokes(track, "right", "one_handed")
    assert [x.type for x in s] == ["backhand"] and s[0].label_confidence <= 0.45


@pytest.mark.parametrize("yaw", [0, 45, 135, 180])
def test_hidden_backswing_stays_unlabelled_if_the_hand_is_not_seen_again_before_the_hit(yaw):
    # The backhand exception decides from the swing seen just before the hit; with that hidden too there is nothing to go on.
    track = make_track([(2.0, "backhand_early_backswing")], yaw=yaw, noise=1.0)
    hide = (track.t >= 2.0 - 1.05) & (track.t <= 2.0 - 0.1)
    track.vis[hide, R_WR] = 0.1
    assert all(x.type != "backhand" for x in detect_strokes(track, "right", "one_handed"))


@pytest.mark.parametrize("yaw", [0, 45, 135, 180])
def test_left_handed_backhand_is_labelled_even_when_the_hidden_hand_is_misplaced(yaw):
    track = make_track([(2.0, "backhand_early_backswing")], yaw=yaw, left_handed=True, noise=1.0)
    hide = (track.t >= 2.0 - 1.05) & (track.t <= 2.0 - 0.4)
    track.vis[hide, L_WR] = 0.1
    w = track.world
    w[hide, L_WR] = w[hide, L_HIP] + (w[hide, L_HIP] - w[hide, R_HIP])  # one hip width out on the hitting-arm side
    s = detect_strokes(track, "left", "one_handed")
    assert [x.type for x in s] == ["backhand"] and s[0].label_confidence <= 0.45


@pytest.mark.parametrize("yaw", [0, 45, 135, 180])
def test_forehand_with_hidden_backswing_stays_unclassified(yaw):
    # Nothing seen before the hit travels backhand-wise, so the hidden-arm guard still applies from any camera angle.
    track = make_track([(2.0, "forehand")], yaw=yaw, noise=1.0)
    hide = (track.t >= 2.0 - 1.05) & (track.t <= 2.0 - 0.4)
    track.vis[hide, R_WR] = 0.1
    assert [x.type for x in detect_strokes(track, "right", "one_handed")] == ["unclassified"]


def test_wrong_handedness_is_not_silently_right():
    """Analysing a right-hander as left-handed must not reproduce the same labels with confidence."""
    track = make_track(EVENTS)
    right = detect_strokes(track, "right")
    left = detect_strokes(track, "left")
    assert [s.type for s in left] != [s.type for s in right] or len(left) != len(right)


def test_no_tennis_movement_gives_empty_report():
    track = make_track([], duration=6.0, noise=1.0)
    strokes, rep = report_for(track)
    assert strokes == [] and rep["summary"]["no_strokes"]
    assert rep["findings"] == [] and rep["priorities"] == []
    assert len(rep["not_assessed"]) >= 5


def test_landmark_jitter_alone_is_not_a_stroke():
    track = make_track([], duration=6.0)
    track.xy[60, R_WR] += 200  # a single-frame glitch
    assert detect_strokes(track, "right") == []


@pytest.mark.parametrize("flex", [20.0, 45.0, 70.0])
def test_knee_bend_measured_side_on(flex):
    track = make_track([(2.0, "serve")], yaw=90, knee_flex=flex)
    s = detect_strokes(track, "right")[0]
    m = measure_stroke(track, s, "right")["values"]
    assert abs(m["knee_bend_dom"]["value"] - flex) < 6
    assert abs(m["knee_bend_dom"]["extra"]["image_plane"] - flex) < 6


def test_frontal_view_knee_bend_uses_3d_estimate_not_foreshortened_2d():
    track = make_track([(2.0, "serve")], yaw=180, knee_flex=60)
    m = measure_stroke(track, detect_strokes(track, "right")[0], "right")["values"]["knee_bend_dom"]
    assert m["extra"]["image_plane"] < 30  # 2D angle is foreshortened head-on
    assert m["value"] > 50  # so the reported value does not falsely read as a straight leg


def test_serve_knee_rule_fires_only_for_shallow_bend():
    serves = [(2.0 + 3 * i, "serve") for i in range(4)]
    _, shallow = report_for(make_track(serves, yaw=90, knee_flex=15))
    _, deep = report_for(make_track(serves, yaw=90, knee_flex=65))
    f1 = next(f for f in shallow["findings"] if f["rule"] == "serve_knee_bend")
    f2 = next(f for f in deep["findings"] if f["rule"] == "serve_knee_bend")
    assert f1["status"] == "observed" and f1["recurring"] and len(f1["occurrences"]) == 4
    assert f1["evidence"]["refs"][0]["id"] == "elliott2003" and f1["interpretation"]
    assert f2["status"] == "not_observed" and f2["evidence"] is None and f2["interpretation"] is None
    assert shallow["priorities"] and shallow["priorities"][0]["finding_id"] == f1["id"]


def test_frontal_checks_refused_from_side_view():
    _, rep = report_for(make_track([(2.0, "serve"), (5.0, "serve")], yaw=90, knee_flex=50))
    by_rule = {f["rule"]: f for f in rep["findings"]}
    assert by_rule["knee_alignment"]["status"] == "unable_to_assess"
    assert by_rule["serve_trunk_lean"]["status"] == "unable_to_assess"
    assert "in front of or behind" in by_rule["serve_trunk_lean"]["observation"]


def test_hidden_legs_suppress_leg_findings():
    track = make_track(EVENTS, yaw=90, knee_flex=10)
    track.vis[:, [L_KNEE, R_KNEE, L_ANK, R_ANK]] = 0.1
    _, rep = report_for(track)
    for f in rep["findings"]:
        if f["rule"] in ("serve_knee_bend", "ground_knee_bend", "knee_alignment"):
            assert f["status"] == "unable_to_assess", f["id"]
    assert any(i["code"] == "legs_hidden" for i in rep["quality"]["issues"])


def test_hidden_hitting_arm_means_no_stroke_claimed():
    track = make_track([(2.0, "forehand")])
    track.vis[:, [R_WR, R_EL]] = 0.1
    assert detect_strokes(track, "right") == []


def test_tracking_gap_makes_stroke_unassessable():
    track = make_track([(2.0, "forehand"), (5.0, "forehand")])
    s = detect_strokes(track, "right")
    assert len(s) == 2
    sl = track.window(s[1].start, s[1].end)
    half = slice(sl.start, sl.start + (sl.stop - sl.start) // 2)
    track.vis[half] = 0
    track.xy[half] = np.nan
    track.world[half] = np.nan
    for attr in ("xy_s", "world_s", "tracked", "mid_hip", "mid_sh", "torso_len", "pelvis_azimuth"):
        track.__dict__.pop(attr, None)
    m = measure_stroke(track, s[1], "right")
    assert not m["quality"]["assessable"] and m["values"] == {}
    assert measure_stroke(track, s[0], "right")["quality"]["assessable"]


def test_manual_stroke_is_measured():
    track = make_track([(2.0, "forehand")])
    s = manual_stroke(track, "right", 1.4, 2.6, "forehand")
    assert s.source == "manual" and abs(s.peak - 2.0) < 0.15
    assert measure_stroke(track, s, "right")["values"]["peak_hand_speed"]["assessable"]


def test_multi_player_tracking_follows_the_selected_person():
    det = FakeDetector([(2.0, "forehand")], people_x=(0.25, 0.7))
    import cv2  # noqa: F401

    meta = {"fps": 30.0, "width": 1280, "height": 720}
    frames = list(range(120))
    dets = [det._poses(i / 30, 1280, 720) for i in frames]
    # shuffle person order per frame so tracking cannot rely on list position
    rng = np.random.default_rng(1)
    for d in dets:
        if rng.random() < 0.5:
            d.lm, d.world = d.lm[::-1].copy(), d.world[::-1].copy()
    seed = det._poses(0, 1280, 720)
    for p, x in enumerate((0.25, 0.7)):
        c = seed.lm[p][[11, 12, 23, 24], :2].mean(axis=0)
        track = track_player(frames, dets, 0, (float(c[0]), float(c[1])), 0.6, meta)
        assert track.tracked.all()
        assert abs(np.nanmedian(track.mid_hip[:, 0]) / 1280 - x) < 0.05
        assert len(detect_strokes(track, "right")) == (1 if p == 1 else 0)


def test_tracking_stops_when_player_leaves_and_does_not_jump_to_another_person():
    det = FakeDetector([], people_x=(0.25, 0.7))
    meta = {"fps": 30.0, "width": 1280, "height": 720}
    dets = [det._poses(i / 30, 1280, 720) for i in range(150)]
    for d in dets[40:]:  # person 0 walks out of frame
        d.lm, d.world = d.lm[1:], d.world[1:]
    c = det._poses(0, 1280, 720).lm[0][[11, 12, 23, 24], :2].mean(axis=0)
    track = track_player(list(range(150)), dets, 0, (float(c[0]), float(c[1])), 0.6, meta)
    assert track.tracked[:40].all() and not track.tracked[40:].any()


def test_confidence_is_bounded_by_weakest_factor_and_explained():
    hi = confidence(0.95, 1.0, 1.0, 4, "image_plane")
    lo = confidence(0.45, 1.0, 1.0, 4, "image_plane")
    one = confidence(0.95, 1.0, 1.0, 1, "image_plane")
    est = confidence(0.95, 1.0, 1.0, 4, "estimated_3d")
    assert hi["level"] == "high" and lo["score"] < hi["score"] and one["score"] < hi["score"] and est["score"] <= 0.75
    assert "not how likely an injury" in hi["explanation"] and len(hi["factors"]) == 5


def test_report_makes_no_diagnostic_or_risk_claims():
    intake = {"discomfort": {"current": True, "regions": ["shoulder"], "notes": ""}, "workload_change": "increased", "skill_level": "beginner"}
    _, rep = report_for(make_track([(2.0 + 3 * i, "serve") for i in range(4)], yaw=90, knee_flex=15), intake)
    text = json.dumps(rep["findings"] + rep["priorities"]).lower()
    for banned in ("risk score", "% risk", "% chance", "you have a", "you may have", "diagnos", "tendin", "tear", "will cause", "safe range", "injury risk of"):
        assert banned not in text, banned
    for f in rep["findings"]:
        if f["status"] == "observed" and f["category"] == "health_related":
            assert f["evidence"]["refs"] and all(r["doi"] and r["review_status"] for r in f["evidence"]["refs"])
            assert f["limitations"] and f["confidence"]["factors"]
        if f["category"] == "technique" and f["rule"] != "speed_change":
            assert f["interpretation"] is None and f["evidence"] is None
    kinds = {n["kind"] for n in rep["context_notes"]}
    assert kinds == {"pain", "workload"}
    pain = next(n for n in rep["context_notes"] if n["kind"] == "pain")["body"].lower()
    assert "don't push through pain" in pain and "exercise" not in pain
    knee = next(f for f in rep["findings"] if f["rule"] == "serve_knee_bend")
    assert knee["user_context"] and "cannot show" in knee["user_context"]
    assert knee["action"]["cue"].startswith("As you toss")  # beginner wording


def test_speed_change_reported_without_blaming_fatigue():
    track = make_track([(2.0 + 2.5 * i, "forehand") for i in range(9)])
    strokes = detect_strokes(track, "right")
    measures = {s.id: measure_stroke(track, s, "right") for s in strokes}
    for s in strokes[-3:]:
        measures[s.id]["values"]["peak_hand_speed"]["value"] *= 0.7
    rep = build_report(track, strokes, measures, {"handedness": "right"}, Q, META)
    f = next(f for f in rep["findings"] if f["rule"] == "speed_change")
    assert "lower" in f["observation"] and "cannot tell which" in f["interpretation"]


def test_signal_helpers():
    a = np.array([1.0, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0])
    assert np.allclose(fill_short_gaps(a, 1)[:3], [1, 2, 3]) and np.isnan(fill_short_gaps(a, 1)[4])
    assert np.allclose(smooth(np.arange(10.0), 5)[2:-2], np.arange(10.0)[2:-2])


def test_empty_detections_handled():
    from app.pipeline.errors import PipelineError

    empty = FramePoses(np.zeros((0, 33, 4), np.float32), np.zeros((0, 33, 3), np.float32))
    with pytest.raises(PipelineError) as e:
        track_player(list(range(30)), [empty] * 30, 0, (0.5, 0.5), 0.5, {"fps": 30.0, "width": 640, "height": 360})
    assert e.value.code == "tracking_lost"
