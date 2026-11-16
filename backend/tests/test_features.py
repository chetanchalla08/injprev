"""Stages 3 and 4: coach / clinician features, share links, deletion and retention of everything new, encryption at rest, LLM check."""
import base64
import dataclasses
import sqlite3
import time
import types

import pytest
from fastapi.testclient import TestClient

from app import llm
from app.crypto import MAGIC, Vault, VaultError, generate_key
from app.main import create_app

from .synth import FakeDetector, make_video
from .test_video_and_api import EVENTS, drain, make_client, upload

SERVES = [(2.0 + 3 * i, "serve") for i in range(4)]
INTAKE = '{"backhand_style":"one_handed","discomfort":{"current":true,"regions":["knee"],"notes":"sore after matches"},"injury_history":"ankle sprain 2024"}'


@pytest.fixture
def done(tmp_path):
    client, worker, s = make_client(tmp_path, FakeDetector(SERVES, yaw=90, knee_flex=15))
    sid = upload(client, make_video(tmp_path / "v.mp4", duration=14), INTAKE).json()["id"]
    drain(worker)
    assert client.get(f"/api/sessions/{sid}").json()["status"] == "complete"
    return client, worker, s, sid


def test_comments_and_drawings_pinned_to_the_video(done):
    client, _, _, sid = done
    assert client.post(f"/api/sessions/{sid}/comments", json={"t": 2.1, "body": ""}).status_code == 400
    assert client.post(f"/api/sessions/{sid}/comments", json={"t": 999, "body": "x"}).status_code == 400
    c = client.post(f"/api/sessions/{sid}/comments", json={"t": 2.1, "body": "Toss is drifting left here", "author": "Coach Kim"}).json()
    assert client.get(f"/api/sessions/{sid}/comments").json()[0]["body"] == "Toss is drifting left here"
    shapes = [{"type": "line", "points": [[0.1, 0.2], [0.5, 0.6]]}, {"type": "angle", "points": [[0.1, 0.1], [0.2, 0.2], [0.3, 0.1]], "color": "red"}, {"type": "free", "points": [[0.1, 0.1], [0.2, 0.2], [0.25, 0.3]]}]
    d = client.post(f"/api/sessions/{sid}/drawings", json={"t": 2.0, "shapes": shapes, "note": "knee angle"})
    assert d.status_code == 201 and d.json()["shapes"][1]["color"] == "red"
    for bad in ([{"type": "line", "points": [[0, 0]]}], [{"type": "blob", "points": [[0, 0], [1, 1]]}], [], [{"type": "angle", "points": [[0, 0], [1, 1]]}]):
        assert client.post(f"/api/sessions/{sid}/drawings", json={"t": 2.0, "shapes": bad}).status_code == 400
    exp = client.get(f"/api/sessions/{sid}/export").json()
    assert len(exp["comments"]) == 1 and len(exp["drawings"]) == 1
    other = TestClient(client.app)
    assert other.get(f"/api/sessions/{sid}/comments").status_code == 404 and other.delete(f"/api/sessions/{sid}/comments/{c['id']}").status_code == 404
    assert client.delete(f"/api/sessions/{sid}/comments/{c['id']}").status_code == 204 and client.get(f"/api/sessions/{sid}/comments").json() == []
    assert client.delete(f"/api/sessions/{sid}/drawings/{d.json()['id']}").status_code == 204


def test_profiles_and_progress_view_with_camera_caveat(tmp_path):
    client, worker, _ = make_client(tmp_path, FakeDetector(EVENTS))
    assert client.post("/api/profiles", json={"name": ""}).status_code == 400
    p = client.post("/api/profiles", json={"name": "Alex", "handedness": "left", "notes": "junior squad"}).json()
    a = upload(client, make_video(tmp_path / "a.mp4", duration=8)).json()["id"]
    with open(make_video(tmp_path / "b.mp4", duration=8), "rb") as f:
        b = client.post("/api/sessions", files={"file": ("b.mp4", f, "video/mp4")}, data={"intake": '{"backhand_style":"one_handed"}', "profile_id": p["id"]}).json()["id"]
    drain(worker)
    assert client.get(f"/api/sessions/{b}").json()["profile_id"] == p["id"]
    assert client.put(f"/api/sessions/{a}/profile", json={"profile_id": "nope"}).status_code == 404
    assert client.put(f"/api/sessions/{a}/profile", json={"profile_id": p["id"]}).status_code == 200
    assert client.get("/api/profiles").json()[0]["sessions"] == 2
    prog = client.get(f"/api/progress?profile_id={p['id']}").json()
    assert len(prog["sessions"]) == 2 and prog["sessions"][0]["camera_view"] and "camera" in prog["caveat"].lower() and "not proof" in prog["caveat"]
    # a session filmed from another view makes the caveat explicit
    s = client.app.state.db.get(a)
    s["report"]["quality"]["tracking"]["camera_view"] = "mostly side-on"
    client.app.state.db.update(a, report=s["report"])
    prog = client.get("/api/progress").json()
    assert prog["mixed_views"] and "different camera views" in prog["caveat"]
    assert TestClient(client.app).get("/api/profiles").json() == []  # profiles are bound to the browser
    assert client.delete(f"/api/profiles/{p['id']}").status_code == 204
    assert client.get(f"/api/sessions/{a}").json()["profile_id"] is None and client.get(f"/api/sessions/{a}").status_code == 200  # sessions outlive the profile


def test_share_link_is_read_only_scoped_expiring_and_revocable(done):
    client, worker, _, sid = done
    client.post(f"/api/sessions/{sid}/comments", json={"t": 2.1, "body": "note for the physio"})
    assert client.post(f"/api/sessions/{sid}/shares", json={"days": 400}).status_code == 400
    made = client.post(f"/api/sessions/{sid}/shares", json={"days": 2}).json()
    token = made["token"]
    assert len(token) >= 43 and made["active"]  # 32 random bytes
    listed = client.get(f"/api/sessions/{sid}/shares").json()
    assert "token" not in listed[0] and listed[0]["active"]
    row = sqlite3.connect(worker.settings.db_path).execute("SELECT token_hash FROM shares").fetchone()[0]
    assert token not in row and len(row) == 64  # only a hash is stored
    guest = TestClient(client.app)  # a different browser with no cookie
    got = guest.get(f"/api/shared/{token}")
    assert got.status_code == 200
    body = got.json()
    assert body["read_only"] and body["report"]["strokes"] and body["comments"][0]["body"] == "note for the physio"
    # self-reported health details never travel with the link
    assert body["intake"]["discomfort"] == {"current": False, "regions": [], "notes": ""} and body["intake"]["injury_history"] == "" and body["report"]["context_notes"] == []
    assert "sore after matches" not in got.text and "ankle sprain" not in got.text
    assert guest.get(f"/api/shared/{token}/video", headers={"Range": "bytes=0-9"}).status_code == 206 and guest.get(f"/api/shared/{token}/overlay").status_code == 200
    # the link gives nothing else: not the owner API, not writes, not other sessions
    for method, path in (("get", f"/api/sessions/{sid}"), ("delete", f"/api/sessions/{sid}"), ("get", f"/api/sessions/{sid}/export"), ("get", f"/api/sessions/{sid}/shares")):
        assert getattr(guest, method)(path).status_code == 404
    assert guest.post(f"/api/sessions/{sid}/comments", json={"t": 1, "body": "x"}).status_code == 404
    assert guest.post(f"/api/shared/{token}/comments", json={"t": 1, "body": "x"}).status_code in (404, 405)
    for wrong in (token[:-1] + ("A" if token[-1] != "A" else "B"), "x" * 43, "short", made["id"]):
        assert guest.get(f"/api/shared/{wrong}").status_code == 404
    # revoked
    assert client.delete(f"/api/sessions/{sid}/shares/{made['id']}").status_code == 204
    assert guest.get(f"/api/shared/{token}").status_code == 404 and guest.get(f"/api/shared/{token}/video").status_code == 404
    assert client.delete(f"/api/sessions/{sid}/shares/{made['id']}").status_code == 404
    # expired
    t2 = client.post(f"/api/sessions/{sid}/shares", json={"days": 1}).json()
    assert guest.get(f"/api/shared/{t2['token']}").status_code == 200
    db = client.app.state.db
    db._conn.execute("UPDATE shares SET expires_at=? WHERE id=?", (time.time() - 1, t2["id"]))
    assert guest.get(f"/api/shared/{t2['token']}").status_code == 404 and not client.get(f"/api/sessions/{sid}/shares").json()[-1]["active"]
    assert guest.get(f"/api/shared/{t2['token']}").json()["detail"] == guest.get("/api/shared/" + "y" * 43).json()["detail"]  # same answer as an unknown link


def test_deleting_a_session_removes_everything_attached_to_it(done):
    client, worker, s, sid = done
    f = next(f for f in client.get(f"/api/sessions/{sid}").json()["report"]["findings"] if f["status"] == "observed")
    client.post(f"/api/sessions/{sid}/comments", json={"t": 2.1, "body": "c"})
    client.post(f"/api/sessions/{sid}/drawings", json={"t": 2.0, "shapes": [{"type": "line", "points": [[0, 0], [1, 1]]}]})
    client.put(f"/api/sessions/{sid}/labels", json={"labeller": "A B", "expert": False, "strokes": [{"type": "serve", "start": 1, "contact": 2, "end": 2.5}]})
    client.put(f"/api/sessions/{sid}/annotations", json={"annotator": "A B", "frames": [{"frame": 5, "joints": {"nose": [0.5, 0.5]}}]})
    client.put(f"/api/sessions/{sid}/reviews", json={"finding_id": f["id"], "stroke_id": None, "verdict": "correct", "cue_ok": True, "reviewer": "Casey Coach", "role": "coach"})
    token = client.post(f"/api/sessions/{sid}/shares", json={}).json()["token"]
    client.app.state.db.put_item("x", "llm_summary", {"text": "t"}, sid)
    prof = client.post("/api/profiles", json={"name": "Alex"}).json()
    db = client.app.state.db
    assert db.count_items() == 8
    assert client.delete(f"/api/sessions/{sid}").status_code == 204
    assert db.count_items() == 1 and not worker.session_dir(sid).exists()  # only the profile is left
    assert TestClient(client.app).get(f"/api/shared/{token}").status_code == 404
    assert client.delete("/api/me").status_code == 204 and db.count_items() == 0 and client.get("/api/profiles").json() == [] and prof["id"]


def test_retention_purges_attached_items_links_and_scratch_files(tmp_path):
    client, worker, s = make_client(tmp_path, FakeDetector(EVENTS), retention_days=7)
    sid = upload(client, make_video(tmp_path / "v.mp4", duration=8)).json()["id"]
    drain(worker)
    client.post(f"/api/sessions/{sid}/comments", json={"t": 1.0, "body": "c"})
    token = client.post(f"/api/sessions/{sid}/shares", json={}).json()["token"]
    assert list(s.tmp_dir.iterdir()) == []  # nothing left in scratch space after processing
    worker.db.update(sid, created_at=0.0)
    assert worker.purge_expired(force=True) == 1 and client.app.state.db.count_items() == 0
    assert TestClient(client.app).get(f"/api/shared/{token}").status_code == 404
    # an expired link record is dropped a day after expiry even while its session lives on
    sid2 = upload(client, make_video(tmp_path / "w.mp4", duration=8)).json()["id"]
    drain(worker)
    sh = client.post(f"/api/sessions/{sid2}/shares", json={}).json()
    client.app.state.db._conn.execute("UPDATE shares SET expires_at=? WHERE id=?", (time.time() - 2 * 86400, sh["id"]))
    worker.purge_expired(force=True)
    assert client.get(f"/api/sessions/{sid2}/shares").json() == []


def test_structured_pdf_export(done):
    client, _, _, sid = done
    client.post(f"/api/sessions/{sid}/comments", json={"t": 2.1, "body": "Check the “toss” — it drifts", "author": "Coach"})
    r = client.get(f"/api/sessions/{sid}/report.pdf")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf" and r.content[:5] == b"%PDF-" and len(r.content) > 8000
    assert "no-store" in r.headers["cache-control"] and TestClient(client.app).get(f"/api/sessions/{sid}/report.pdf").status_code == 404
    from app.pdf import build_pdf

    s = client.app.state.db.get(sid)
    seen = []
    import fpdf

    orig, orig_cell = fpdf.FPDF.multi_cell, fpdf.FPDF.cell

    def spy(self, w, h=None, text="", *a, **k):
        seen.append(text)
        return orig(self, w, h, text, *a, **k)

    def spy_cell(self, w=None, h=None, text="", *a, **k):
        seen.append(text)
        return orig_cell(self, w, h, text, *a, **k)

    fpdf.FPDF.multi_cell, fpdf.FPDF.cell = spy, spy_cell
    try:
        build_pdf(title=s["title"], created_at=s["created_at"], intake=s["intake"], report=s["report"], track=None)
    finally:
        fpdf.FPDF.multi_cell, fpdf.FPDF.cell = orig, orig_cell
    text = " ".join(seen)
    for needed in ("Not clinically validated", "Observation", "Interpretation", "Evidence", "Action", "Elliott", "not a finding about this person", "single-camera", "Not assessed"[:3], "Self-reported"):
        assert needed in text or needed in " ".join(seen), needed
    s2 = client.get(f"/api/sessions/{sid}/strokes/{s['report']['strokes'][0]['id']}/series").json()
    assert s2["series"] and s2["timing"]["source"] and client.get(f"/api/sessions/{sid}/strokes/nope/series").status_code == 404


# --- encryption at rest -----------------------------------------------------------------


def test_vault_roundtrip_ranges_and_tamper_detection(tmp_path):
    v = Vault.from_env(generate_key())
    data = bytes(range(256)) * 700  # spans several 64 KiB chunks
    p = tmp_path / "f.bin"
    with v.writer(p) as w:
        for i in range(0, len(data), 50_000):
            w.write(data[i : i + 50_000])
    assert p.read_bytes().startswith(MAGIC) and data[:64] not in p.read_bytes()
    assert v.read_file(p) == data and v.plain_size(p) == len(data)
    for a, b in ((0, 0), (10, 99), (65_530, 65_545), (131_072, 131_072), (len(data) - 5, len(data) - 1), (100_000, 10**9)):
        assert b"".join(v.read_range(p, a, b)) == data[a : b + 1]
    assert v.decrypt(v.encrypt(b"")) == b"" and v.open_text(v.seal_text("héllo")) == "héllo" and v.open_text("plain") == "plain"
    raw = bytearray(p.read_bytes())
    raw[5000] ^= 1
    p.write_bytes(bytes(raw))
    with pytest.raises(VaultError) as e:
        v.read_file(p)
    assert e.value.code == "corrupt"
    p.write_bytes(bytes(raw[: len(raw) // 2]))  # truncated
    with pytest.raises(VaultError):
        v.read_file(p)
    with pytest.raises(VaultError) as e:
        Vault.from_env(generate_key()).decrypt(v.encrypt(b"secret"))
    assert e.value.code == "key_mismatch"
    with pytest.raises(VaultError) as e:
        Vault().decrypt(v.encrypt(b"secret"))
    assert e.value.code == "key_missing"
    for bad in ("not base64!!", base64.urlsafe_b64encode(b"short").decode()):
        with pytest.raises(VaultError) as e:
            Vault.from_env(bad)
        assert e.value.code == "bad_key"
    assert not Vault.from_env("").enabled and Vault().encrypt(b"x") == b"x"


def test_encrypted_storage_end_to_end_and_missing_or_wrong_key(tmp_path):
    key = generate_key()
    client, worker, s = make_client(tmp_path, FakeDetector(SERVES, yaw=90, knee_flex=15), data_key=key)
    assert client.get("/api/config").json()["encryption_at_rest"] is True
    sid = upload(client, make_video(tmp_path / "v.mp4", duration=14), INTAKE).json()["id"]
    d = worker.session_dir(sid)
    assert next(d.glob("upload.*")).read_bytes().startswith(MAGIC)  # encrypted from the moment it lands
    drain(worker)
    body = client.get(f"/api/sessions/{sid}").json()
    assert body["status"] == "complete" and body["intake"]["injury_history"] == "ankle sprain 2024"
    client.post(f"/api/sessions/{sid}/comments", json={"t": 2.0, "body": "private coaching note"})
    for name in ("video.mp4", "track.npz", "overlay.json", "aux.json", "candidates.jpg"):
        assert (d / name).read_bytes().startswith(MAGIC), name
    raw = s.db_path.read_bytes()
    for secret in (b"ankle sprain", b"sore after matches", b"private coaching note", b"Limited knee bend", b"disclaimer"):
        assert secret not in raw, secret
    assert list(s.tmp_dir.iterdir()) == []  # the plaintext working copy is gone
    # everything still works through the API, including byte ranges and rebuilds
    plain = b"".join(client.app.state.store.vault.read_range(d / "video.mp4"))
    rng = client.get(f"/api/sessions/{sid}/video", headers={"Range": "bytes=100-299"})
    assert rng.status_code == 206 and rng.content == plain[100:300] and rng.headers["content-range"] == f"bytes 100-299/{len(plain)}"
    assert client.get(f"/api/sessions/{sid}/video").content == plain and plain[4:8] == b"ftyp"
    assert client.get(f"/api/sessions/{sid}/video", headers={"Range": f"bytes={len(plain) + 5}-"}).status_code == 416
    assert client.get(f"/api/sessions/{sid}/overlay").json()["frames"] and client.get(f"/api/sessions/{sid}/report.pdf").content[:5] == b"%PDF-"
    st = body["report"]["strokes"]
    assert client.put(f"/api/sessions/{sid}/strokes", json={"strokes": [{"id": x["id"], "start": x["start"], "end": x["end"], "type": x["type"]} for x in st[:2]]}).status_code == 200
    cookie = client.cookies.get("tl_owner")

    def reopen(**over):
        s2 = dataclasses.replace(s)
        for k, v in over.items():
            setattr(s2, k, v)
        c = TestClient(create_app(s2, lambda: FakeDetector(SERVES), start_worker=False))
        c.cookies.set("tl_owner", cookie)
        return c

    # restarted WITHOUT the key: a clear error, never garbage or plaintext
    nokey = reopen(data_key="")
    r = nokey.get(f"/api/sessions/{sid}")
    assert r.status_code == 503 and "DATA_KEY" in r.json()["detail"] and r.json()["code"] == "storage_key_missing"
    assert nokey.get("/api/sessions").status_code == 503
    # restarted with a DIFFERENT key
    wrong = reopen(data_key=generate_key())
    assert wrong.get(f"/api/sessions/{sid}").json()["code"] == "storage_key_mismatch"
    # the right key again
    assert reopen().get(f"/api/sessions/{sid}").json()["status"] == "complete"
    with pytest.raises(VaultError) as e:
        reopen(data_key="", require_encryption=True)
    assert e.value.code == "key_missing"
    with pytest.raises(VaultError):
        reopen(data_key="zzz")
    # deletion still removes everything
    assert client.delete(f"/api/sessions/{sid}").status_code == 204 and not d.exists()


def test_data_stored_before_a_key_was_set_stays_readable(tmp_path):
    client, worker, s = make_client(tmp_path, FakeDetector(EVENTS))
    sid = upload(client, make_video(tmp_path / "v.mp4", duration=8)).json()["id"]
    drain(worker)
    assert not (worker.session_dir(sid) / "video.mp4").read_bytes().startswith(MAGIC) and client.get("/api/config").json()["encryption_at_rest"] is False
    s.data_key = generate_key()
    c2 = TestClient(create_app(s, lambda: FakeDetector(EVENTS), start_worker=False))
    c2.cookies.set("tl_owner", client.cookies.get("tl_owner"))
    assert c2.get(f"/api/sessions/{sid}").json()["status"] == "complete" and c2.get(f"/api/sessions/{sid}/video", headers={"Range": "bytes=0-9"}).status_code == 206


# --- optional LLM summary ---------------------------------------------------------------


class FakeLLM:
    def __init__(self, text, stop="end_turn"):
        self.text, self.stop, self.calls = text, stop, []
        self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        return types.SimpleNamespace(stop_reason=self.stop, model="fake-model", content=[types.SimpleNamespace(type="text", text=self.text)])


def test_llm_summary_is_checked_against_the_findings(done, monkeypatch):
    client, _, s, sid = done
    assert client.get(f"/api/sessions/{sid}/summary").json() == {"enabled": False, "summary": None}
    assert client.post(f"/api/sessions/{sid}/summary").status_code == 404  # the app works without a key; the feature is simply absent
    s.llm_api_key = "test-key"
    rep = client.get(f"/api/sessions/{sid}").json()["report"]
    f = next(f for f in rep["findings"] if f["rule"] == "serve_knee_bend")
    good = "In 4 of 4 assessable serves, neither knee bent more than about 15° while loading. " + f["action"]["cue"] + " This is technique analysis, not a diagnosis of anything."
    cases = {
        "Your knees bent about 22° on each serve.": "numbers that are not in the findings",  # an invented measurement
        "In 4 of 4 serves your knees stayed fairly straight (Elliott et al., 2003).": "cite",
        "Neither knee bent much in 4 serves, which research shows raises shoulder load. See https://example.org": "cite",
        "In 4 of 4 serves the knee bend was shallow, so you are at risk of tendinitis.": "diagnostic or risk wording",
        "In four of seven serves the knee bend was shallow.": "quantities that are not in the findings",
        "": "empty",
    }
    src = llm.source_document(rep)
    assert "refs" not in str(src) and "doi" not in str(src).lower() and "ankle sprain" not in str(src) and "sore after" not in str(src)
    for text, why in cases.items():
        assert any(why in p for p in llm.check(text, src)), text
    assert llm.check(good.replace("not a diagnosis of anything", "education only"), src) == []

    fake = FakeLLM("Your knees bent about 22° on each serve.")
    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: fake)
    r = client.post(f"/api/sessions/{sid}/summary")
    assert r.status_code == 422 and "discarded" in r.json()["detail"] and "22" in r.json()["detail"]
    assert client.get(f"/api/sessions/{sid}/summary").json()["summary"] is None  # nothing invented is ever stored or shown
    sent = fake.calls[0]
    assert sent["messages"][0]["content"].startswith("{") and "ankle sprain" not in str(sent) and "Elliott" not in str(sent) and "cite" in sent["system"].lower()

    fake.text = good.replace("not a diagnosis of anything", "education only")
    r = client.post(f"/api/sessions/{sid}/summary")
    assert r.status_code == 200 and r.json()["summary"]["text"].startswith("In 4 of 4") and "AI language model" in r.json()["summary"]["notice"]
    assert client.get(f"/api/sessions/{sid}/summary").json()["summary"]["model"] == "fake-model"
    # editing strokes changes the findings, so the cached rewording is dropped
    st = rep["strokes"]
    client.put(f"/api/sessions/{sid}/strokes", json={"strokes": [{"id": x["id"], "start": x["start"], "end": x["end"], "type": x["type"]} for x in st[:1]]})
    assert client.get(f"/api/sessions/{sid}/summary").json()["summary"] is None
    fake.stop = "refusal"
    assert client.post(f"/api/sessions/{sid}/summary").status_code == 502
    assert client.get(f"/api/sessions/{sid}").json()["report"]["findings"]  # the structured report is untouched throughout


def test_live_capture_framing_guidance(tmp_path):
    import cv2
    import numpy as np

    from app.pipeline.framing import assess
    from app.pipeline.pose import FramePoses

    det = FakeDetector([])
    bright = np.random.default_rng(0).integers(60, 220, (360, 640, 3)).astype(np.uint8)
    good = assess(bright, det._poses(0, 640, 360))
    assert good["ready"] and good["height_frac"] > 0.33 and all(c["ok"] for c in good["checks"])
    nobody = assess(bright, FramePoses(np.zeros((0, 33, 4), np.float32), np.zeros((0, 33, 3), np.float32)))
    assert not nobody["ready"] and "No player found" in next(c for c in nobody["checks"] if c["key"] == "person")["message"]
    dark = assess(np.full((360, 640, 3), 10, np.uint8), det._poses(0, 640, 360))
    assert not dark["ready"] and {c["key"] for c in dark["checks"] if c["ok"] is False} == {"lighting", "focus"}
    far = det._poses(0, 640, 360)
    far.lm[:, :, 1] = 0.5 + (far.lm[:, :, 1] - 0.5) * 0.25
    assert "closer" in next(c for c in assess(bright, far)["checks"] if c["key"] == "size")["message"]
    cut = det._poses(0, 640, 360)
    cut.lm[:, :, 1] += 0.2  # feet below the bottom edge
    assert "feet" in next(c for c in assess(bright, cut)["checks"] if c["key"] == "whole_body")["message"]
    client, _, s = make_client(tmp_path, det)
    ok, jpg = cv2.imencode(".jpg", bright)
    r = client.post("/api/framing", content=jpg.tobytes())
    assert r.status_code == 200 and r.json()["people"] == 1
    assert client.post("/api/framing", content=jpg.tobytes()).status_code == 429  # rate limited
    assert TestClient(client.app).post("/api/framing", content=b"nope").status_code == 400
    assert list(s.sessions_dir.iterdir()) == [] and list(s.tmp_dir.iterdir()) == []  # nothing is stored
