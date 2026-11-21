"""Stage 1: label format, scoring maths, labelling / annotation / review APIs."""
import json

import numpy as np
import pytest

from app import angles, review
from app.evaltools import score
from app.evaltools.format import LABEL_FORMAT, FormatError, parse_labels

from .synth import FakeDetector, make_track, make_video
from .test_video_and_api import EVENTS, drain, make_client, upload


def L(strokes, **kw):
    return {"format": LABEL_FORMAT, "clip": "c", "labeller": "A Coach", "expert": False, "strokes": strokes, **kw}


def test_label_format_validation():
    ok = parse_labels(L([{"type": "serve", "start": 1, "contact": 2, "end": 3}]), {"handedness": "left", "camera_view": "side"})
    assert ok.strokes[0].handedness == "left" and ok.strokes[0].camera_view == "side" and ok.to_dict()["format"] == LABEL_FORMAT
    for bad in (L([], labeller=""), L([], expert="yes"), L([{"type": "smash", "start": 1, "contact": 2, "end": 3}]), L([{"type": "serve", "start": 3, "contact": 2, "end": 1}]),
                L([{"type": "serve", "start": 1, "end": 3}]), {**L([]), "format": "other"}):  # fmt: skip
        with pytest.raises(FormatError):
            parse_labels(bad)


def test_matching_is_one_to_one_and_closest_first():
    labels = [{"contact": 1.0}, {"contact": 1.6}, {"contact": 5.0}]
    preds = [{"time": 1.5}, {"time": 1.05}, {"time": 9.0}]
    pairs, miss, extra = score.match(labels, preds)
    assert pairs == [(0, 1), (1, 0)] and miss == [2] and extra == [2]


def test_scores_precision_recall_confusion_and_timing():
    labels = parse_labels(L([
        {"type": "forehand", "start": 0.5, "contact": 1.0, "end": 1.5}, {"type": "backhand", "start": 2.5, "contact": 3.0, "end": 3.5},
        {"type": "serve", "start": 4.5, "contact": 6.0, "end": 6.5, "contact_kind": "peak"}, {"type": "forehand", "start": 8.5, "contact": 9.0, "end": 9.5},
        {"type": "other", "start": 11, "contact": 11.5, "end": 12},
    ]), {"camera_view": "side"})  # fmt: skip
    preds = [{"type": "forehand", "time": 1.1}, {"type": "forehand", "time": 3.2}, {"type": "unclassified", "time": 6.1}, {"type": "serve", "time": 11.4}, {"type": "backhand", "time": 20.0}]
    items = score.score_clip(labels, preds)
    m = score.summarise(items)
    assert (m["labelled"], m["detected"], m["missed"], m["false_detections"]) == (4, 3, 1, 1)  # the volley is neither a miss nor a false detection
    assert m["detection_recall"] == 0.75 and m["detection_precision"] == 0.75 and m["correct_label"] == 1 and m["wrong_label"] == 1 and m["left_unlabelled"] == 1
    assert m["confusion"]["backhand"]["forehand"] == 1 and m["confusion"]["forehand"]["missed"] == 1 and m["false_by_predicted_type"]["backhand"] == 1
    assert m["timing_vs_contact"]["n"] == 2 and m["timing_vs_contact"]["median_abs_s"] == pytest.approx(0.15) and m["timing_vs_swing_peak"]["n"] == 1
    agg = score.aggregate([{"clip": "a", "split": "tune", "items": items}, {"clip": "b", "split": "held_out", "items": []}])
    assert agg["tune"]["overall"]["labelled"] == 4 and agg["held_out"]["overall"]["detection_recall"] is None
    assert set(agg["tune"]["by_stroke_type"]) == {"forehand", "backhand", "serve"} and "side" in agg["tune"]["by_camera_view"] and "right" in agg["tune"]["by_handedness"]


def test_other_strokes_take_a_subtype_only_from_the_known_list():
    ok = parse_labels(L([{"type": "other", "subtype": "smash", "start": 1, "contact": 2, "end": 3}, {"type": "other", "start": 4, "contact": 5, "end": 6}]))
    assert [s.subtype for s in ok.strokes] == ["smash", None] and ok.to_dict()["strokes"][0]["subtype"] == "smash"
    for bad in (L([{"type": "other", "subtype": "tweener", "start": 1, "contact": 2, "end": 3}]), L([{"type": "serve", "subtype": "smash", "start": 1, "contact": 2, "end": 3}])):
        with pytest.raises(FormatError):
            parse_labels(bad)


def test_report_shows_what_smashes_and_slices_were_taken_for_and_splits_by_backhand_style_and_player():
    labels = parse_labels(L([
        {"type": "other", "subtype": "smash", "start": 0.5, "contact": 1.0, "end": 1.5}, {"type": "other", "subtype": "smash", "start": 2.5, "contact": 3.0, "end": 3.5},
        {"type": "other", "subtype": "slice", "start": 4.5, "contact": 5.0, "end": 5.5}, {"type": "other", "start": 6.5, "contact": 7.0, "end": 7.5},
        {"type": "backhand", "start": 8.5, "contact": 9.0, "end": 9.5},
    ]), {"backhand_style": "one_handed"})  # fmt: skip
    preds = [{"type": "serve", "time": 1.0}, {"type": "backhand", "time": 5.1}, {"type": "serve", "time": 9.0}]
    items = score.score_clip(labels, preds, player="p7")
    m = score.summarise(items)
    assert m["other_by_subtype"] == {"smash": {"serve": 1, "missed": 1}, "slice": {"backhand": 1}, "unspecified": {"missed": 1}}
    assert m["labelled"] == 1 and m["false_detections"] == 0  # "other" strokes still neither count as misses nor as false detections
    agg = score.aggregate([{"clip": "a", "split": "tune", "items": items}])["tune"]
    assert agg["by_backhand_style"]["one_handed"]["confusion"]["backhand"]["serve"] == 1 and agg["by_player"]["p7"]["labelled"] == 1


def _add(root, video, cid, split, player):
    from app.evaltools import add

    return add.main(["--root", str(root), "--id", cid, "--video", str(video), "--split", split, "--licence", "test", "--page", "test", "--player", player])


def test_add_records_the_player_and_warns_when_a_player_would_be_in_both_splits(tmp_path, capsys):
    from app.evaltools.format import load_manifest

    video = make_video(tmp_path / "v.mp4", duration=1)
    root = tmp_path / "evalset"
    assert _add(root, video, "a", "tune", "p1") == 0 and _add(root, video, "b", "held_out", "p2") == 0
    assert "WARNING" not in capsys.readouterr().out
    assert _add(root, video, "c", "held_out", "p1") == 0
    out = capsys.readouterr().out
    assert "WARNING" in out and "p1" in out and "tune" in out
    assert {c.id: c.player for c in load_manifest(root)} == {"a": "p1", "b": "p2", "c": "p1"}


def test_empty_set_reports_blank_not_zero(tmp_path, capsys):
    assert score.main(["--root", str(tmp_path)]) == 1 and "No clips" in capsys.readouterr().out
    assert score.summarise([])["detection_recall"] is None and score.summarise([])["timing_vs_contact"] is None


@pytest.fixture
def done(tmp_path):
    client, worker, s = make_client(tmp_path, FakeDetector(EVENTS), reviewer_code="let-me-review")
    sid = upload(client, make_video(tmp_path / "v.mp4", duration=8)).json()["id"]
    drain(worker)
    return client, worker, sid


def test_labelling_mode_save_and_export(done):
    client, _, sid = done
    assert client.get(f"/api/sessions/{sid}/labels").json()["strokes"] == []
    assert client.get(f"/api/sessions/{sid}/labels/export").status_code == 404
    body = {"labeller": "Sam Coach", "labeller_role": "LTA level 3", "expert": True, "camera_view": "side", "strokes": [{"type": "forehand", "start": 1.0, "contact": 1.5, "end": 2.0}]}
    assert client.put(f"/api/sessions/{sid}/labels", json={**body, "labeller": ""}).status_code == 400
    assert client.put(f"/api/sessions/{sid}/labels", json={**body, "strokes": [{"type": "forehand", "start": 1.0, "contact": 1.5, "end": 99}]}).status_code == 400
    assert client.put(f"/api/sessions/{sid}/labels", json=body).status_code == 200
    body["strokes"].append({"type": "serve", "start": 5.0, "contact": 6.0, "end": 6.5, "contact_kind": "peak"})
    assert client.put(f"/api/sessions/{sid}/labels", json=body).status_code == 200  # replaces, does not duplicate
    exp = client.get(f"/api/sessions/{sid}/labels/export")
    doc = exp.json()
    assert "attachment" in exp.headers["content-disposition"] and doc["expert"] is True and len(doc["strokes"]) == 2
    parsed = parse_labels(doc, doc["clip_defaults"])  # the export is a valid label file for the scoring harness
    assert parsed.strokes[0].camera_view == "side" and parsed.labeller == "Sam Coach"


def test_joint_annotation_and_error_report(done):
    client, worker, sid = done
    assert client.get(f"/api/sessions/{sid}/annotations/report").json()["available"] is False  # no reference -> no accuracy claim
    from app.pipeline.signals import PoseTrack

    track = PoseTrack.load(worker.session_dir(sid) / "track.npz")
    frames = []
    for i in (10, 40, 70):
        j = {name: [float(track.xy[i, idx, 0] / track.width) + 0.01, float(track.xy[i, idx, 1] / track.height)] for name, idx in angles.JOINTS.items()}
        frames.append({"frame": i, "joints": j})
    assert client.put(f"/api/sessions/{sid}/annotations", json={"annotator": "", "frames": frames}).status_code == 400
    assert client.put(f"/api/sessions/{sid}/annotations", json={"annotator": "A N", "frames": [{"frame": 10**7, "joints": {"nose": [0.5, 0.5]}}]}).status_code == 400
    assert client.put(f"/api/sessions/{sid}/annotations", json={"annotator": "A N", "frames": frames}).status_code == 200
    rep = client.get(f"/api/sessions/{sid}/annotations/report").json()
    assert rep["available"] and rep["frames_compared"] == 3 and rep["camera_view"]
    knee = rep["position_error_by_joint"]["left_knee"]
    assert knee["n"] == 3 and abs(knee["median_px"] - 0.01 * track.width) < 1.0  # recovers the 1 % offset we introduced
    assert rep["image_plane_angle_error"]["knee_flexion_left"]["median_abs_deg"] < 1.0
    assert client.get(f"/api/sessions/{sid}/annotations/export").json()["frames"][0]["frame"] == 10


def test_mocap_csv_comparison_and_bad_files(done):
    client, worker, sid = done
    from app.pipeline.signals import PoseTrack

    track = PoseTrack.load(worker.session_dir(sid) / "track.npz")
    series = angles._series(track)
    rows = ["# vicon export", "time_s,knee_flexion_right,elbow_angle_right,ignored"]
    for i in range(20, 200, 10):
        rows.append(f"{track.t[i]:.3f},{series['knee_flexion_right']['image_plane'][i] + 5:.2f},{series['elbow_angle_right']['estimated_3d'][i]:.2f},x")
    r = client.post(f"/api/sessions/{sid}/mocap", content="\n".join(rows))
    assert r.status_code == 200
    a = r.json()["angles"]
    assert a["knee_flexion_right"]["methods"]["image_plane"]["mean_signed_deg"] == pytest.approx(-5, abs=0.3)
    assert a["elbow_angle_right"]["methods"]["estimated_3d"]["mean_abs_deg"] < 0.2 and "image_plane" in a["elbow_angle_right"]["methods"]
    assert client.post(f"/api/sessions/{sid}/mocap", content="a,b\n1,2").status_code == 400
    assert client.post(f"/api/sessions/{sid}/mocap", content="time_s,knee_flexion_left\nabc,1").status_code == 400


def test_stroke_angle_series_has_gaps_where_tracking_is_poor():
    track = make_track([(2.0, "forehand")], yaw=90)
    track.vis[55:62, 26] = 0.1  # right knee hidden for a few frames
    out = angles.stroke_series(track, 1.4, 2.6)
    knee = next(s for s in out["series"] if s["key"] == "knee_flexion_right")
    i = out["t"].index(round(58 / 30, 3))
    assert knee["methods"]["image_plane"][i] is None and knee["methods"]["estimated_3d"][i] is None and knee["visibility"][i] < 0.5
    assert knee["methods"]["image_plane"][0] is not None and knee["tracked_frac"] < 1
    other = next(s for s in out["series"] if s["key"] == "knee_flexion_left")
    assert None not in other["methods"]["image_plane"] and set(out["method_notes"]) == {"image_plane", "estimated_3d"}
    assert "estimated_3d" not in next(s for s in out["series"] if s["key"] == "trunk_lean")["methods"]


def test_nothing_is_reviewed_until_a_reviewer_signs_off(done):
    client, _, sid = done
    items = client.get("/api/review/items").json()
    assert items["reviewer_mode"] and all(i["status"]["state"] == "unreviewed" for i in items["items"])
    ev = next(i for i in items["items"] if i["id"] == "evidence:elliott2003")
    base = {"target": ev["id"], "decision": "approved", "reviewer": "Dr Pat Physio", "role": "physiotherapist", "credentials": "HCPC PH000", "hash": ev["hash"]}
    assert client.post("/api/review/signoff", json={**base, "code": "wrong"}).status_code == 403
    assert client.post("/api/review/signoff", json=base).status_code == 403
    assert client.post("/api/review/signoff", json={**base, "code": "let-me-review", "hash": "stale"}).status_code == 409
    assert client.post("/api/review/signoff", json={**base, "code": "let-me-review", "reviewer": "x"}).status_code == 400
    assert client.post("/api/review/signoff", json={**base, "code": "let-me-review", "decision": "changes_requested"}).status_code == 400  # needs a comment
    # a coach cannot approve a clinical item
    r = client.post("/api/review/signoff", json={**base, "code": "let-me-review", "reviewer": "Casey Coach", "role": "coach"})
    assert r.status_code == 201 and r.json()["status"]["state"] == "unreviewed"
    r = client.post("/api/review/signoff", json={**base, "code": "let-me-review"})
    assert r.status_code == 201 and r.json()["status"]["state"] == "approved" and "Dr Pat Physio" in r.json()["status"]["label"]
    # a later "changes requested" from another clinician wins until resolved
    r = client.post("/api/review/signoff", json={**base, "code": "let-me-review", "reviewer": "Dr Sam Sports", "role": "sports_physician", "decision": "changes_requested", "comment": "Summary overstates it."})
    assert r.json()["status"]["state"] == "changes_requested"
    so = next(s for s in r.json()["status"]["signoffs"] if s["reviewer"] == "Dr Sam Sports")
    assert client.post(f"/api/review/signoff/{so['id']}/withdraw", json={"code": "let-me-review"}).status_code == 200
    st = next(i for i in client.get("/api/review/items").json()["items"] if i["id"] == ev["id"])["status"]
    assert st["state"] == "approved"


def test_signoff_goes_stale_when_wording_changes(monkeypatch):
    items = review.reviewables()
    so = [{"id": "1", "target": "evidence:elliott2003", "content_hash": items["evidence:elliott2003"]["hash"], "decision": "approved", "reviewer": "Dr P", "role": "physiotherapist", "credentials": "", "comment": "", "created_at": 1.0}]
    assert review.status_of("evidence:elliott2003", so)["state"] == "approved"
    from app.pipeline import evidence

    monkeypatch.setitem(evidence.EVIDENCE, "elliott2003", {**evidence.EVIDENCE["elliott2003"], "summary": "Edited summary."})
    assert review.status_of("evidence:elliott2003", so)["state"] == "stale"


def test_signoff_updates_review_status_in_reports_and_is_off_without_a_code(tmp_path):
    det = FakeDetector([(2.0 + 3 * i, "serve") for i in range(4)], yaw=90, knee_flex=15)
    client, worker, _ = make_client(tmp_path, det, reviewer_code="code-1")
    sid = upload(client, make_video(tmp_path / "v.mp4", duration=14)).json()["id"]
    drain(worker)
    f = next(f for f in client.get(f"/api/sessions/{sid}").json()["report"]["findings"] if f["rule"] == "serve_knee_bend")
    assert f["evidence"]["refs"][0]["review_status"] == "Not yet reviewed by a sports clinician" and f["drills"][0]["review_status"] == "Not yet coach-reviewed"
    ev = next(i for i in client.get("/api/review/items").json()["items"] if i["id"] == "evidence:elliott2003")
    client.post("/api/review/signoff", json={"code": "code-1", "target": ev["id"], "decision": "approved", "reviewer": "Dr Pat Physio", "role": "physiotherapist", "hash": ev["hash"]})
    f = next(f for f in client.get(f"/api/sessions/{sid}").json()["report"]["findings"] if f["rule"] == "serve_knee_bend")
    assert f["evidence"]["refs"][0]["review_status"].startswith("Reviewed and approved by Dr Pat Physio")
    assert "unvalidated heuristics" in client.get(f"/api/sessions/{sid}").json()["report"]["validation_status"]
    # per-clip rule review
    occ = f["occurrences"][0]["stroke_id"]
    rv = {"finding_id": f["id"], "stroke_id": occ, "verdict": "correct", "cue_ok": True, "comment": "", "reviewer": "Casey Coach", "role": "coach"}
    assert client.put(f"/api/sessions/{sid}/reviews", json={**rv, "reviewer": ""}).status_code == 400
    assert client.put(f"/api/sessions/{sid}/reviews", json={**rv, "stroke_id": "nope"}).status_code == 400
    assert client.put(f"/api/sessions/{sid}/reviews", json=rv).status_code == 200
    assert client.put(f"/api/sessions/{sid}/reviews", json={**rv, "verdict": "incorrect"}).status_code == 200  # same reviewer, same clip: updated
    client.put(f"/api/sessions/{sid}/reviews", json={**rv, "reviewer": "Other Coach"})
    summ = client.get("/api/review/export").json()["summary"]["serve_knee_bend"]
    assert summ["clips_reviewed"] == 1 and summ["reviews"] == 2 and summ["pair_agreement"] == 0.0 and summ["judged_correct"] == 1
    # another server without a reviewer code cannot record sign-offs at all
    c2, _, _ = make_client(tmp_path / "b", FakeDetector(EVENTS))
    assert c2.post("/api/review/signoff", json={"code": "", "target": ev["id"], "decision": "approved", "reviewer": "Dr Pat Physio", "role": "physiotherapist", "hash": ev["hash"]}).status_code == 403
    assert json.dumps(c2.get("/api/review/items").json()).count('"approved"') == 0
