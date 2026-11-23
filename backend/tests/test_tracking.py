"""Stage 2: ball contact, court lines, scene cuts and re-acquisition, including the cases where each must give up."""
import subprocess

import cv2
import numpy as np
import pytest

from app.pipeline import ball, court, run
from app.pipeline.options import BASELINE, Options
from app.pipeline.pose import FramePoses, track_player
from app.pipeline.segment import apply_ball, detect_strokes, phases
from app.pipeline.video import ffmpeg_exe
from app.pipeline.visual import APPEARANCE_DIM, describe, find_cuts, similarity, visual_pass

from .synth import FakeDetector, make_track, make_video
from .test_video_and_api import drain, make_client, upload

Q = {"metrics": {}, "issues": []}
META = {"fps": 30.0, "width": 1280, "height": 720, "duration_s": 6.0, "n_frames": 180}
BALL_BGR = (60, 230, 200)  # tennis-ball yellow-green


def ball_frames(path, n=40, turn=20, size=(640, 360)):
    """A ball flies in from the left, reverses at frame ``turn`` and flies back out faster."""
    out = []
    rng = np.random.default_rng(0)
    bg = np.clip(np.array([40, 90, 160]) + cv2.resize(rng.integers(-25, 25, (36, 64, 1)).astype(np.float32), size, interpolation=cv2.INTER_CUBIC)[..., None], 0, 255).astype(np.uint8)
    for i in range(n):
        img = bg.copy()
        x = 100 + 14 * i if i <= turn else 100 + 14 * turn - 22 * (i - turn)
        y = 200 - 2 * i if i <= turn else 200 - 2 * turn - 3 * (i - turn)
        cv2.circle(img, (int(x), int(y)), 4, BALL_BGR, -1, cv2.LINE_AA)
        out.append(img)
    return out, (100 + 14 * turn, 200 - 2 * turn)


def test_ball_is_tracked_and_contact_found_where_its_path_reverses():
    frames, hit = ball_frames(None)
    per = [[]] + [ball.find_candidates(frames[i - 1], frames[i], frames[i + 1]) for i in range(1, len(frames) - 1)] + [[]]
    assert sum(bool(p) for p in per) >= 30
    tracks = ball.link_tracks(per, 640, 360)
    assert tracks and max(len(t.idx) for t in tracks) >= 15
    wrist = np.tile(np.array(hit, float) + [10, 5], (len(frames), 1))
    est = ball.estimate_contact(tracks, wrist, np.full(len(frames), 60.0), peak_i=22, rate=30.0)
    assert est is not None and abs(est["index"] - 20) <= 1.0 and np.hypot(est["xy"][0] - hit[0], est["xy"][1] - hit[1]) < 25
    # the same ball far from the player's hand is not a contact
    assert ball.estimate_contact(tracks, wrist + [0, 400], np.full(len(frames), 60.0), 22, 30.0) is None
    # a blob that only ever moves with the hand (the racket) is not the ball
    hand = np.asarray([t_.xy for t_ in tracks][0])
    glued = ball.BallTrack(list(range(len(hand))), [(x + 30, y) for x, y in hand], [1.0] * len(hand))
    assert ball.estimate_contact([glued], np.pad(hand, ((0, len(frames) - len(hand)), (0, 0)), mode="edge"), np.full(len(frames), 60.0), 20, 30.0) is None
    # a contact too far in time from the pose-based timestamp is not believed
    assert ball.estimate_contact(tracks, wrist, np.full(len(frames), 60.0), peak_i=36, rate=30.0) is None


def test_no_ball_found_keeps_pose_timestamp_and_says_so():
    still = [np.full((360, 640, 3), (40, 90, 160), np.uint8)] * 5
    assert ball.find_candidates(still[0], still[1], still[2]) == [] and ball.link_tracks([[] for _ in range(30)], 640, 360) == []
    track = make_track([(2.0, "forehand"), (4.5, "serve")])
    strokes = apply_ball(track, detect_strokes(track, "right", "one_handed"), [], "right")
    assert [s.timing_source for s in strokes] == ["peak_hand_speed", "highest_hand"] and all(s.contact is None and s.peak == s.pose_time for s in strokes)
    rep = run.compose(track, {"ball_tracks": [], "cuts": []}, intake={"handedness": "right", "backhand_style": "one_handed"}, quality=Q, meta=META, confirmations={}, options=Options(court=False))
    assert rep["ball"]["status"] == "unable" and "could not be assessed" in rep["ball"]["note"]
    s = rep["strokes"][0]
    assert s["timing"]["source"] == "peak_hand_speed" and "not contact" in s["timing"]["label"]
    contact = [m for m in s["measurements"] if m["key"].startswith("contact_")]
    assert len(contact) == 3 and all(not m["assessable"] and "not tracked well enough" in m["reason"] for m in contact)
    assert s["phases"]["boundary"].startswith("pose-based")


def test_ball_contact_replaces_timestamp_and_gives_contact_point():
    track = make_track([(2.0, "forehand")], yaw=90)
    s0 = detect_strokes(track, "right", "one_handed")[0]
    i = track.index_at(s0.peak) + 2  # put contact two frames after peak hand speed
    w = track.xy[i, 16]
    idx_in, idx_out = list(range(i - 8, i + 1)), list(range(i + 1, i + 8))
    tr_in = ball.BallTrack(idx_in, [(w[0] + 20 - 45 * (i - k), w[1] - 30) for k in idx_in], [1.0] * len(idx_in))
    tr_out = ball.BallTrack(idx_out, [(w[0] + 20 - 60 * (k - i), w[1] - 30 - 4 * (k - i)) for k in idx_out], [1.0] * len(idx_out))
    rep = run.compose(track, {"ball_tracks": ball.to_json([tr_in, tr_out]), "cuts": []}, intake={"handedness": "right", "backhand_style": "one_handed"}, quality=Q, meta=META, confirmations={}, options=Options(court=False))
    s = rep["strokes"][0]
    assert s["timing"]["source"] == "ball_contact" and abs(s["peak"] - track.t[i]) <= 0.05 and s["timing"]["pose_time"] == pytest.approx(s0.peak, abs=0.01)
    m = {x["key"]: x for x in s["measurements"]}
    assert m["contact_height"]["assessable"] and m["contact_height"]["method"] == "image_plane" and m["contact_distance"]["value"] > 0
    assert m["contact_in_front"]["assessable"]  # side-on view
    assert rep["ball"]["status"] == "ok" and s["phases"]["forward_swing"][1] == s["peak"] and s["phases"]["boundary"] == "contact"
    # from the front the same contact gives no "in front" distance
    front = run.compose(make_track([(2.0, "forehand")], yaw=180), {"ball_tracks": ball.to_json([tr_in, tr_out]), "cuts": []}, intake={"handedness": "right", "backhand_style": "one_handed"},
                        quality=Q, meta=META, confirmations={}, options=Options(court=False))  # fmt: skip
    cf = next((x for st in front["strokes"] for x in st["measurements"] if x["key"] == "contact_in_front"), None)
    assert cf is None or not cf["assessable"]
    # ball tracking switched off: listed as not assessed, timing unchanged
    off = run.compose(track, {}, intake={"handedness": "right", "backhand_style": "one_handed"}, quality=Q, meta=META, confirmations={}, options=BASELINE)
    assert off["ball"]["status"] == "off" and "switched off" in off["not_assessed"][0]["reason"] and "not validated" in rep["not_assessed"][0]["reason"] and off["strokes"][0]["timing"]["source"] == "peak_hand_speed"


def test_hidden_arm_recovery_is_off_by_default_and_never_fires_when_the_arm_is_visible():
    assert Options().recover_hidden is False
    track = make_track([], duration=4.0)
    i, c = 60, track.mid_sh[60]
    tr_in = ball.BallTrack(list(range(i - 6, i + 1)), [(c[0] + 70 * (i - k), c[1]) for k in range(i - 6, i + 1)], [1.0] * 7)
    tr_out = ball.BallTrack(list(range(i + 1, i + 7)), [(c[0] + 80 * (k - i), c[1] - 5) for k in range(i + 1, i + 7)], [1.0] * 6)
    assert apply_ball(track, [], [tr_in, tr_out], "right", recover_hidden=True) == []  # arm visible, no swing: a bounce, not a stroke
    track.vis[:, [14, 16]] = 0.1
    got = apply_ball(track, [], [tr_in, tr_out], "right", recover_hidden=True)
    assert len(got) == 1 and got[0].type == "unclassified" and got[0].timing_source == "ball_contact" and "hidden" in got[0].label_basis


def synthetic_court(w=960, h=540, lines=None):
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [10.97, 0], [10.97, 23.77], [0, 23.77]]), np.float32([[120, 500], [840, 500], [640, 150], [320, 150]]))
    img = np.full((h, w, 3), (90, 120, 60), np.uint8)
    for a, b, name in court.MODEL_LINES:
        if lines is None or name in lines:
            p = court._project(Hm, np.float32([a, b]))
            cv2.line(img, tuple(map(int, p[0])), tuple(map(int, p[1])), (245, 245, 245), 3, cv2.LINE_AA)
    return img, Hm


def test_court_mapping_recovered_from_lines():
    img, Hm = synthetic_court()
    f = court.fit_court(img)
    assert f["found"] and len(f["lines_supported"]) >= 7
    pts = np.float32([[5.5, 2.0], [2.0, 10.0], [9.0, 20.0]])
    assert np.abs(court._project(np.linalg.inv(f["H"]), court._project(Hm, pts)) - pts).max() < 0.3
    assert court.zone(np.array([5.4, -2.5])) == "well behind the baseline, near the centre" and "forecourt" in court.zone(np.array([2.0, 9.0]))


def test_no_court_lines_or_too_few_is_unable_to_assess_not_a_guess():
    blank = np.full((540, 960, 3), (60, 90, 170), np.uint8)
    f = court.fit_court(blank)
    assert not f["found"] and "cannot be fixed" in f["reason"]
    corner, _ = synthetic_court(lines={"baseline_near", "sideline_doubles_l", "sideline_singles_l", "service_near"})  # a corner fits many placements
    assert not court.fit_court(corner)["found"]
    track = make_track([(2.0, "forehand")])
    res = court.analyse({0: blank, 30: blank}, track)
    assert res["status"] == "unable" and res["camera_view"] is None and not np.isfinite(res["pos"]).any()
    res["pos"] = [None] * track.n
    rep = run.compose(track, {"court": res, "ball_tracks": [], "cuts": []}, intake={"handedness": "right", "backhand_style": "one_handed"}, quality=Q, meta=META, confirmations={}, options=Options())
    assert rep["court"]["status"] == "unable" and rep["movement"] is None and rep["strokes"][0]["movement"] is None
    assert rep["quality"]["tracking"]["camera_view_basis"].startswith("body orientation") and rep["hops"]["validated"] is False


def test_movement_reported_only_as_unvalidated_when_court_found():
    track = make_track([(2.0, "forehand"), (4.5, "forehand")])
    pos = [[3.0 + 3.0 * abs((i % 60) - 30) / 30, -0.5] for i in range(track.n)]  # side to side at 3 m/s
    rep = run.compose(track, {"court": {"status": "ok", "reason": "", "camera_view": "mostly side-on", "pos": pos}, "ball_tracks": [], "cuts": []}, intake={"handedness": "right", "backhand_style": "one_handed"},
                      quality=Q, meta=META, confirmations={}, options=Options())  # fmt: skip
    mv = rep["strokes"][0]["movement"]
    assert "baseline" in mv["zone"] and mv["recovery"]["after_s"] > 0 and rep["movement"]["validated"] is False and rep["movement"]["direction_changes"]
    assert rep["quality"]["tracking"]["camera_view"] == "mostly side-on" and rep["quality"]["tracking"]["camera_view_basis"] == "court lines"
    assert any("Footwork" in n["topic"] for n in rep["not_assessed"])  # still listed as not validated


def test_scene_cut_detected_in_real_video_and_not_in_a_continuous_one(tmp_path):
    a, b = make_video(tmp_path / "a.mp4", duration=2), make_video(tmp_path / "b.mp4", duration=2, source="smptebars")
    (tmp_path / "list.txt").write_text(f"file '{a}'\nfile '{b}'\n")
    subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(tmp_path / "list.txt"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp_path / "cut.mp4")], check=True)
    empty = FramePoses(np.zeros((0, 33, 4), np.float32), np.zeros((0, 33, 3), np.float32))
    v = visual_pass(tmp_path / "cut.mp4", list(range(120)), [empty] * 120, want_ball=False, fps=30)
    assert v.cuts == [60]
    assert visual_pass(a, list(range(60)), [empty] * 60, want_ball=False, fps=30).cuts == []
    dark = np.zeros(40, bool)
    dark[10:20] = True
    assert find_cuts(np.ones(40), np.zeros(40), dark) == [20]  # coming out of a fade to black starts a new shot


def people(xs, n, w=1280, h=720):
    det = FakeDetector([], people_x=xs)
    return [det._poses(i / 30, w, h) for i in range(n)]


def appearance(dets, colours):
    """Fake clothing descriptors: one distinct histogram per 'colour' id, per detection."""
    app = np.full((len(dets), 4, APPEARANCE_DIM), np.nan, np.float32)
    for i, d in enumerate(dets):
        for p in range(d.lm.shape[0]):
            v = np.zeros(APPEARANCE_DIM)
            mix = colours(i, p)
            for k, wgt in mix.items():
                v[k], v[39 + k] = wgt, wgt
            app[i, p] = v
    return app


M = {"fps": 30.0, "width": 1280, "height": 720}


def seed(dets):
    c = dets[0].lm[0][[11, 12, 23, 24], :2].mean(axis=0)
    return (float(c[0]), float(c[1]))


def test_cut_mid_stroke_stops_tracking_and_no_stroke_is_claimed_across_it():
    det = FakeDetector([(2.0, "forehand")])
    dets = [det._poses(i / 30, 1280, 720) for i in range(120)]
    track = track_player(list(range(120)), dets, 0, seed(dets), 0.6, M, cuts=[60], reacquire=False)  # the cut lands at the hit
    assert track.tracked[:60].all() and not track.tracked[60:].any()
    for s in detect_strokes(track, "right", "one_handed"):
        assert s.end <= 2.0  # nothing extends over the cut
    whole = track_player(list(range(120)), dets, 0, seed(dets), 0.6, M)
    assert len(detect_strokes(whole, "right", "one_handed")) == 1


def test_after_a_cut_a_different_looking_person_is_never_picked_up():
    dets = people((0.5,), 120)
    app = appearance(dets, lambda i, p: {3: 1.0} if i < 60 else {20: 1.0})  # same place, different clothes after the cut
    track = track_player(list(range(120)), dets, 0, seed(dets), 0.6, M, cuts=[60], appearance=app, reacquire=True)
    assert not track.tracked[60:].any() and len(track.segments) == 1


def test_after_a_cut_the_same_looking_player_is_picked_up_and_a_doubtful_one_needs_confirmation():
    dets = people((0.5,), 120)
    same = track_player(list(range(120)), dets, 0, seed(dets), 0.6, M, cuts=[60], appearance=appearance(dets, lambda i, p: {3: 1.0}), reacquire=True)
    assert same.tracked.all() and same.segments[1]["status"] == "auto" and same.segments[1]["after"] == "cut"
    maybe = appearance(dets, lambda i, p: {3: 1.0} if i < 60 else {3: 0.7, 20: 0.3})
    track = track_player(list(range(120)), dets, 0, seed(dets), 0.6, M, cuts=[60], appearance=maybe, reacquire=True)
    sg = track.segments[1]
    assert sg["status"] == "needs_confirmation" and 0.6 <= sg["similarity"] < 0.8 and sg["box"] and track.tracked.all()
    assert not track.masked({}).tracked[60:].any()  # not analysed until confirmed
    assert track.masked({"1": True}).tracked.all() and not track.masked({"1": False}).tracked[60:].any()
    # without re-acquisition switched on nothing after the cut is followed at all
    assert not track_player(list(range(120)), dets, 0, seed(dets), 0.6, M, cuts=[60], appearance=maybe).tracked[60:].any()


def test_a_bystander_seen_alongside_the_player_is_never_reacquired_even_if_dressed_alike():
    dets = people((0.25, 0.7), 150)
    for d in dets[40:]:  # the selected player (first person) walks out; the bystander stays
        d.lm, d.world = d.lm[1:], d.world[1:]
    app = appearance(dets, lambda i, p: {3: 1.0})  # identical clothing
    track = track_player(list(range(150)), dets, 0, seed(dets), 0.6, M, appearance=app, reacquire=True)
    assert track.tracked[:40].all() and not track.tracked[40:].any()


def test_player_who_leaves_and_returns_alone_is_reacquired_by_appearance():
    dets = people((0.3,), 200)
    empty = FramePoses(np.zeros((0, 33, 4), np.float32), np.zeros((0, 33, 3), np.float32))
    for i in range(60, 140):
        dets[i] = empty
    app = appearance(dets, lambda i, p: {3: 1.0})
    track = track_player(list(range(200)), dets, 0, seed(dets), 0.6, M, appearance=app, reacquire=True)
    assert track.tracked[140:].all() and track.segments[1]["after"] == "left_frame" and track.segments[1]["status"] == "auto"
    # with nothing to compare clothing with, it is only ever offered for confirmation
    blind = track_player(list(range(200)), dets, 0, seed(dets), 0.6, M, appearance=None, reacquire=True)
    assert all(s["status"] == "needs_confirmation" for s in blind.segments[1:])


def test_appearance_descriptor_separates_colours():
    hsv = np.zeros((200, 200, 3), np.uint8)
    hsv[..., 0], hsv[..., 1], hsv[..., 2] = 0, 200, 200
    lm = np.zeros((33, 4), np.float32)
    lm[:, 3] = 0.9
    for j, (x, y) in {11: (0.4, 0.2), 12: (0.6, 0.2), 23: (0.42, 0.5), 24: (0.58, 0.5), 25: (0.42, 0.75), 26: (0.58, 0.75)}.items():
        lm[j, :2] = (x, y)
    red = describe(hsv, lm)
    hsv[..., 0] = 110
    blue = describe(hsv, lm)
    assert similarity(red, red) == pytest.approx(1.0) and similarity(red, blue) < 0.1
    lm[:, 3] = 0.1
    assert similarity(red, describe(hsv, lm)) is None


def test_confirming_a_reacquired_stretch_through_the_api(tmp_path, monkeypatch):
    import app.pipeline.pose as pose

    monkeypatch.setattr(pose, "REACQ_AUTO_SIM", 1.01)  # force every re-acquisition to ask

    class Gap(FakeDetector):
        def _poses(self, t, w, h):
            fp = super()._poses(t, w, h)
            return FramePoses(fp.lm[:0], fp.world[:0]) if 2.6 <= t < 4.6 else fp

    client, worker, _ = make_client(tmp_path, Gap([(1.5, "forehand"), (6.0, "serve")]))
    sid = upload(client, make_video(tmp_path / "v.mp4", duration=8, source="color=c=gray")).json()["id"]
    drain(worker)
    body = client.get(f"/api/sessions/{sid}").json()
    assert body["status"] == "complete"
    rep = body["report"]
    seg = next(s for s in rep["tracking_events"]["segments"] if s["status"] == "needs_confirmation")
    assert seg["decision"] is None and [s["type"] for s in rep["strokes"]] == ["forehand"]  # the serve after the gap is not analysed yet
    assert any(i["code"] == "confirm_player" for i in rep["quality"]["issues"])
    assert client.post(f"/api/sessions/{sid}/segments/0", json={"decision": "confirm"}).status_code == 400
    r = client.post(f"/api/sessions/{sid}/segments/{seg['id']}", json={"decision": "confirm"})
    assert r.status_code == 200 and [s["type"] for s in r.json()["report"]["strokes"]] == ["forehand", "serve"]
    r = client.post(f"/api/sessions/{sid}/segments/{seg['id']}", json={"decision": "reject"})
    assert [s["type"] for s in r.json()["report"]["strokes"]] == ["forehand"] and r.json()["report"]["tracking_events"]["segments"][1]["decision"] == "rejected"


def test_swing_phases_are_ordered():
    track = make_track([(2.0, "forehand")])
    s = detect_strokes(track, "right", "one_handed")[0]
    p = phases(track, s, "right")
    assert p["preparation"][0] < p["forward_swing"][0] < p["forward_swing"][1] == p["follow_through"][0] < p["follow_through"][1]


def _unlabelled_forehand(yaw, hand="right"):
    """A forehand whose lead-up is hidden, so the pose alone leaves it unlabelled (see test_analysis)."""
    track = make_track([(2.0, "forehand")], yaw=yaw, noise=1.0, left_handed=hand == "left")
    track.vis[(track.t >= 2.0 - 1.05) & (track.t <= 2.0 - 0.4), 16 if hand == "right" else 15] = 0.1
    strokes = detect_strokes(track, hand, "one_handed")
    assert [s.type for s in strokes] == ["unclassified"]
    return track, strokes


def _ball_at(track, i, side_tl, hand="right"):
    """Ball paths in and out meeting at frame i, side_tl torso lengths from the hip centre toward the hitting-arm shoulder."""
    dom, non = (12, 11) if hand == "right" else (11, 12)
    across = np.sign(track.xy_s[i, dom, 0] - track.xy_s[i, non, 0])
    c = np.array([track.mid_hip[i, 0] + across * side_tl * track.torso_len[i], track.mid_sh[i, 1]])
    away = 1.0 if c[0] >= track.mid_hip[i, 0] else -1.0  # the ball comes in from outside, not across the body, and goes back out
    tr_in = ball.BallTrack(list(range(i - 8, i + 1)), [(c[0] + away * 45 * (i - k), c[1] + 6 * (i - k)) for k in range(i - 8, i + 1)], [1.0] * 9)
    tr_out = ball.BallTrack(list(range(i + 1, i + 8)), [(c[0] + away * 60 * (k - i), c[1] - 4 * (k - i)) for k in range(i + 1, i + 8)], [1.0] * 7)
    return [tr_in, tr_out]


@pytest.mark.parametrize("yaw", [0, 180])
@pytest.mark.parametrize("hand", ["right", "left"])
@pytest.mark.parametrize("side_tl, expected", [(1.2, "forehand"), (-1.2, "backhand"), (0.3, "unclassified")])
def test_unlabelled_stroke_is_labelled_from_the_ball_side_at_contact(yaw, hand, side_tl, expected):
    # p12_rally_front: player ~90 px tall, wrist landmarks flicker, so the pose leaves rally forehands and two-handed
    # backhands unlabelled; the ball at contact, well to one side of the hips, still says which side the hit was on.
    track, strokes = _unlabelled_forehand(yaw, hand)
    i = track.index_at(strokes[0].peak)
    got = apply_ball(track, strokes, _ball_at(track, i, side_tl, hand), hand, recover_hidden=False)
    assert got[0].timing_source == "ball_contact"
    assert got[0].type == expected
    if expected != "unclassified":
        assert got[0].label_confidence <= 0.5 and "ball" in got[0].label_basis


def test_ball_side_is_not_used_when_the_player_is_side_on():
    # Side-on the shoulders overlap in the image, so left and right of the hips says nothing about forehand or backhand.
    track, strokes = _unlabelled_forehand(90)
    i = track.index_at(strokes[0].peak)
    got = apply_ball(track, strokes, _ball_at(track, i, 1.2), "right", recover_hidden=False)
    assert got[0].type == "unclassified"


def test_ball_side_never_changes_a_stroke_the_user_labelled():
    track, strokes = _unlabelled_forehand(180)
    strokes[0].source = "manual"
    i = track.index_at(strokes[0].peak)
    got = apply_ball(track, strokes, _ball_at(track, i, 1.2), "right", recover_hidden=False)
    assert got[0].type == "unclassified"


def test_ball_side_label_is_dropped_when_the_ball_no_longer_supports_it():
    # A rebuild keeps saved labels; one that came from the ball must not outlive the ball (no contact now, or ball off).
    track, strokes = _unlabelled_forehand(180)
    i = track.index_at(strokes[0].peak)
    labelled = apply_ball(track, strokes, _ball_at(track, i, 1.2), "right", recover_hidden=False)
    assert labelled[0].type == "forehand"
    again = apply_ball(track, labelled, [], "right", recover_hidden=False)
    assert again[0].type == "unclassified" and again[0].timing_source == "peak_hand_speed"
    off = run.compose(track, {}, intake={"handedness": "right", "backhand_style": "one_handed"}, quality=Q, meta=META, confirmations={},
                      options=Options(ball=False, court=False), strokes=apply_ball(track, strokes, _ball_at(track, i, 1.2), "right", recover_hidden=False))  # fmt: skip
    assert off["strokes"][0]["type"] == "unclassified"
