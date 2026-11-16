"""Video validation with real ffmpeg/OpenCV, and the HTTP workflow with a scripted detector."""
import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import load_settings
from app.main import create_app
from app.pipeline.errors import PipelineError
from app.pipeline.video import ffmpeg_exe, normalize, quality_checks

from .synth import FakeDetector, make_video

EVENTS = [(1.5, "forehand"), (3.5, "backhand"), (6.0, "serve")]


# --- video validation -------------------------------------------------------------------


def test_text_file_renamed_to_mp4_is_rejected(tmp_path):
    bad = tmp_path / "notes.mp4"
    bad.write_text("this is not a video")
    with pytest.raises(PipelineError) as e:
        normalize(bad, tmp_path / "out.mp4", max_duration_s=60, min_duration_s=1)
    assert e.value.code in ("unsupported_file", "decode_failed")


def test_truncated_video_is_rejected_or_recovered_cleanly(tmp_path):
    good = make_video(tmp_path / "good.mp4")
    data = good.read_bytes()
    cut = tmp_path / "cut.mp4"
    cut.write_bytes(data[: len(data) // 3])  # moov atom lost
    with pytest.raises(PipelineError):
        normalize(cut, tmp_path / "out.mp4", max_duration_s=60, min_duration_s=1)
    assert not (tmp_path / "out.mp4").exists()


def test_random_bytes_rejected(tmp_path):
    f = tmp_path / "x.mov"
    f.write_bytes(np.random.default_rng(0).bytes(200_000))
    with pytest.raises(PipelineError):
        normalize(f, tmp_path / "out.mp4", max_duration_s=60, min_duration_s=1)


def test_too_long_and_too_short(tmp_path):
    v = make_video(tmp_path / "v.mp4", duration=4)
    with pytest.raises(PipelineError) as e:
        normalize(v, tmp_path / "o.mp4", max_duration_s=2, min_duration_s=1)
    assert e.value.code == "too_long"
    with pytest.raises(PipelineError) as e:
        normalize(v, tmp_path / "o.mp4", max_duration_s=60, min_duration_s=10)
    assert e.value.code == "too_short"


@pytest.mark.parametrize("fps,expected", [(24, 24), (30, 30), (60, 60), (120, 60)])
def test_frame_rates_normalised(tmp_path, fps, expected):
    v = make_video(tmp_path / "v.mp4", duration=2, fps=fps)
    meta = normalize(v, tmp_path / "o.mp4", max_duration_s=60, min_duration_s=1)
    assert meta["fps"] == expected and abs(meta["duration_s"] - 2) < 0.2


def test_portrait_and_rotated_phone_video(tmp_path):
    portrait = make_video(tmp_path / "p.mp4", duration=2, size="360x640")
    meta = normalize(portrait, tmp_path / "po.mp4", max_duration_s=60, min_duration_s=1)
    assert (meta["width"], meta["height"]) == (360, 640)
    # landscape pixels + 90 degree rotation flag, as phones record portrait clips
    rotated = tmp_path / "r.mp4"
    subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-y", "-display_rotation", "90", "-i", str(make_video(tmp_path / "l.mp4", duration=2)), "-c", "copy", str(rotated)], check=True)
    meta = normalize(rotated, tmp_path / "ro.mp4", max_duration_s=60, min_duration_s=1)
    assert meta["height"] > meta["width"], "rotation metadata must be applied"


def test_large_video_downscaled(tmp_path):
    v = make_video(tmp_path / "v.mp4", duration=1, size="1920x1080")
    meta = normalize(v, tmp_path / "o.mp4", max_duration_s=60, min_duration_s=1)
    assert (meta["width"], meta["height"]) == (1280, 720)


def test_quality_flags_dark_low_fps_and_blur(tmp_path):
    dark = make_video(tmp_path / "d.mp4", duration=2, fps=24, source="color=c=0x101010", size="640x360")
    meta = normalize(dark, tmp_path / "do.mp4", max_duration_s=60, min_duration_s=1)
    codes = {i["code"] for i in quality_checks(tmp_path / "do.mp4", meta)["issues"]}
    assert {"low_light", "low_frame_rate", "soft_image", "low_resolution"} <= codes
    sharp = make_video(tmp_path / "s.mp4", duration=2, fps=60, size="1280x720")
    meta = normalize(sharp, tmp_path / "so.mp4", max_duration_s=60, min_duration_s=1)
    q = quality_checks(tmp_path / "so.mp4", meta)
    assert not {"low_light", "low_frame_rate", "soft_image", "low_resolution"} & {i["code"] for i in q["issues"]}


def test_real_pose_model_reports_no_person_on_empty_scene(tmp_path):
    """Runs the actual MediaPipe model. A test pattern contains nobody, so the pipeline must say so."""
    from app.pipeline.pose import MediaPipeDetector, find_candidates

    s = load_settings()
    if not s.model_path.exists():
        pytest.skip("pose model not downloaded")
    v = make_video(tmp_path / "v.mp4", duration=2)
    meta = normalize(v, tmp_path / "o.mp4", max_duration_s=60, min_duration_s=1)
    with pytest.raises(PipelineError) as e:
        find_candidates(MediaPipeDetector(s.model_path), tmp_path / "o.mp4", meta, n_samples=3)
    assert e.value.code == "no_person"


# --- HTTP workflow ----------------------------------------------------------------------


def make_client(tmp_path, detector, **over):
    s = load_settings()
    s.data_dir, s.retention_days, s.max_attempts = tmp_path / "data", 30, 2
    for k, v in over.items():
        setattr(s, k, v)
    app = create_app(s, lambda: detector, start_worker=False)
    return TestClient(app), app.state.worker, s


def drain(worker):
    while worker.run_one():
        pass


def upload(client, path, intake='{"backhand_style":"one_handed"}'):  # the scripted figure hits one-handed
    with open(path, "rb") as f:
        return client.post("/api/sessions", files={"file": (path.name, f, "video/mp4")}, data={"intake": intake})


def test_full_workflow_single_player(tmp_path):
    client, worker, s = make_client(tmp_path, FakeDetector(EVENTS))
    video = make_video(tmp_path / "rally.mp4", duration=8)
    r = upload(client, video, '{"handedness":"right","backhand_style":"one_handed","skill_level":"advanced","goals":["serve"],"bogus":"x","discomfort":{"current":true,"regions":["knee","spleen"]}}')
    assert r.status_code == 201
    sid = r.json()["id"]
    assert client.get(f"/api/sessions/{sid}").json()["status"] == "queued_prepare"
    drain(worker)
    body = client.get(f"/api/sessions/{sid}").json()
    assert body["status"] == "complete", body
    assert body["intake"]["discomfort"]["regions"] == ["knee"] and "bogus" not in body["intake"]
    rep = body["report"]
    assert [x["type"] for x in rep["strokes"]] == ["forehand", "backhand", "serve"]
    assert rep["context_notes"][0]["kind"] == "pain"
    d = worker.session_dir(sid)
    assert not list(d.glob("upload.*")), "original upload must not be retained"
    assert (d / "video.mp4").exists() and (d / "track.npz").exists()

    # video is served with range support, overlay matches the frame count
    rng = client.get(f"/api/sessions/{sid}/video", headers={"Range": "bytes=0-99"})
    assert rng.status_code == 206 and len(rng.content) == 100
    ov = client.get(f"/api/sessions/{sid}/overlay").json()
    assert len(ov["frames"]) == rep["meta"]["n_frames"] and len(ov["frames"][0]) == len(ov["points"]) * 3

    # manual correction: relabel one, delete one, add one; measurements are recomputed
    strokes = rep["strokes"]
    edited = [
        {"id": strokes[0]["id"], "start": strokes[0]["start"], "end": strokes[0]["end"], "type": "backhand"},
        {"id": strokes[2]["id"], "start": strokes[2]["start"], "end": strokes[2]["end"], "type": "serve"},
        {"start": 3.0, "end": 4.2, "type": "forehand"},
    ]
    r = client.put(f"/api/sessions/{sid}/strokes", json={"strokes": edited})
    assert r.status_code == 200
    new = r.json()["report"]["strokes"]
    assert [(x["type"], x["source"]) for x in new] == [("backhand", "manual"), ("forehand", "manual"), ("serve", "auto")]
    assert all(x["measurements"] for x in new)
    assert client.put(f"/api/sessions/{sid}/strokes", json={"strokes": [{"start": 5, "end": 99, "type": "serve"}]}).status_code == 400
    assert client.put(f"/api/sessions/{sid}/strokes", json={"strokes": [{"start": 1, "end": 2, "type": "smash"}]}).status_code == 400

    exp = client.get(f"/api/sessions/{sid}/export")
    assert exp.status_code == 200 and "attachment" in exp.headers["content-disposition"] and exp.json()["report"]["disclaimer"]
    assert len(client.get("/api/sessions").json()) == 1

    # delete removes the row, the video and all derived files
    assert client.delete(f"/api/sessions/{sid}").status_code == 204
    assert not d.exists() and client.get(f"/api/sessions/{sid}").status_code == 404
    assert client.get(f"/api/sessions/{sid}/video").status_code == 404


def test_multiple_players_require_selection(tmp_path):
    client, worker, _ = make_client(tmp_path, FakeDetector(EVENTS, people_x=(0.25, 0.7)))
    sid = upload(client, make_video(tmp_path / "v.mp4", duration=8)).json()["id"]
    drain(worker)
    body = client.get(f"/api/sessions/{sid}").json()
    assert body["status"] == "awaiting_player" and len(body["candidates"]) == 2
    assert client.get(f"/api/sessions/{sid}/candidates.jpg").headers["content-type"] == "image/jpeg"
    assert client.post(f"/api/sessions/{sid}/player", json={"candidate": 5}).status_code == 400
    # candidates are sorted by size; find the one on the right, who is the one swinging
    right = max(range(2), key=lambda i: body["candidates"][i]["box"][0])
    client.post(f"/api/sessions/{sid}/player", json={"candidate": right})
    drain(worker)
    rep = client.get(f"/api/sessions/{sid}").json()["report"]
    assert len(rep["strokes"]) == 3
    client.post(f"/api/sessions/{sid}/player", json={"candidate": 1 - right})  # re-pick the bystander
    drain(worker)
    rep = client.get(f"/api/sessions/{sid}").json()["report"]
    assert rep["strokes"] == [] and rep["summary"]["no_strokes"]


def test_sessions_are_private_to_the_browser_that_created_them(tmp_path):
    client, worker, s = make_client(tmp_path, FakeDetector(EVENTS))
    sid = upload(client, make_video(tmp_path / "v.mp4", duration=8)).json()["id"]
    drain(worker)
    other = TestClient(client.app)
    for path in ("", "/video", "/overlay", "/export", "/candidates.jpg"):
        assert other.get(f"/api/sessions/{sid}{path}").status_code == 404, path
    assert other.delete(f"/api/sessions/{sid}").status_code == 404
    assert other.put(f"/api/sessions/{sid}/strokes", json={"strokes": []}).status_code == 404
    assert other.get("/api/sessions").json() == []
    assert client.get(f"/api/sessions/{sid}").status_code == 200
    # "delete everything" wipes all of this browser's sessions and files
    assert client.delete("/api/me").status_code == 204
    assert not worker.session_dir(sid).exists() and list(s.sessions_dir.iterdir()) == []


def test_upload_validation(tmp_path):
    client, worker, s = make_client(tmp_path, FakeDetector(EVENTS), max_upload_mb=1)
    txt = tmp_path / "a.txt"
    txt.write_text("hello")
    assert upload(client, txt).status_code == 415
    exe = tmp_path / "..%2f..%2fevil.exe"
    exe.write_bytes(b"MZ")
    assert upload(client, exe).status_code == 415
    big = tmp_path / "big.mp4"
    big.write_bytes(b"0" * (3 * 1024 * 1024))
    assert upload(client, big).status_code == 413
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    assert upload(client, empty).status_code == 400
    assert list(s.sessions_dir.iterdir()) == [], "rejected uploads leave nothing on disk"
    # corrupted content with a valid extension fails in processing with a clear message
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video at all" * 100)
    sid = upload(client, bad).json()["id"]
    drain(worker)
    body = client.get(f"/api/sessions/{sid}").json()
    assert body["status"] == "failed" and body["error"]["code"] in ("unsupported_file", "decode_failed") and body["error"]["suggestion"]
    assert not list(worker.session_dir(sid).glob("upload.*"))
    assert client.post(f"/api/sessions/{sid}/retry").status_code == 410


def test_transient_failure_is_retried_automatically(tmp_path):
    det = FakeDetector(EVENTS, fail_times=1)
    client, worker, _ = make_client(tmp_path, det)
    sid = upload(client, make_video(tmp_path / "v.mp4", duration=8)).json()["id"]
    drain(worker)
    assert client.get(f"/api/sessions/{sid}").json()["status"] == "complete" and det.calls == 2


def test_persistent_failure_surfaces_and_manual_retry_works(tmp_path):
    det = FakeDetector(EVENTS, fail_times=2)
    client, worker, _ = make_client(tmp_path, det)
    sid = upload(client, make_video(tmp_path / "v.mp4", duration=8)).json()["id"]
    drain(worker)
    body = client.get(f"/api/sessions/{sid}").json()
    assert body["status"] == "failed" and body["error"]["code"] == "internal_error" and "simulated" not in body["error"]["message"]
    assert client.post(f"/api/sessions/{sid}/retry").status_code == 200
    drain(worker)
    assert client.get(f"/api/sessions/{sid}").json()["status"] == "complete"


def test_interrupted_job_resumes_after_restart(tmp_path):
    client, worker, _ = make_client(tmp_path, FakeDetector(EVENTS))
    sid = upload(client, make_video(tmp_path / "v.mp4", duration=8)).json()["id"]
    worker.db.claim()  # simulate a crash mid-prepare
    assert client.get(f"/api/sessions/{sid}").json()["status"] == "preparing"
    worker.db.requeue_interrupted()
    drain(worker)
    assert client.get(f"/api/sessions/{sid}").json()["status"] == "complete"


def test_retention_purges_old_sessions(tmp_path):
    client, worker, _ = make_client(tmp_path, FakeDetector(EVENTS), retention_days=7)
    sid = upload(client, make_video(tmp_path / "v.mp4", duration=8)).json()["id"]
    drain(worker)
    assert worker.purge_expired(force=True) == 0
    worker.db.update(sid, created_at=0.0)
    assert worker.purge_expired(force=True) == 1
    assert not worker.session_dir(sid).exists() and client.get(f"/api/sessions/{sid}").status_code == 404


def test_compare_sessions(tmp_path):
    client, worker, _ = make_client(tmp_path, FakeDetector(EVENTS))
    a = upload(client, make_video(tmp_path / "a.mp4", duration=8)).json()["id"]
    b = upload(client, make_video(tmp_path / "b.mp4", duration=8)).json()["id"]
    drain(worker)
    r = client.get(f"/api/compare?a={a}&b={b}").json()
    assert len(r["sessions"]) == 2 and "camera" in r["caveat"].lower()
    assert r["sessions"][0]["stroke_summary"]["forehand"]["metrics"]["peak_hand_speed"]["n"] == 1
