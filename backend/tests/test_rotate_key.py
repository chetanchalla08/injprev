"""Key rotation: everything stored moves to the new key, safely and resumably."""
import os

import pytest
from fastapi.testclient import TestClient

from app import rotate_key
from app.crypto import MAGIC, Vault, VaultError, generate_key
from app.main import create_app
from app.storage import DataDirLock

from .synth import FakeDetector, make_video
from .test_video_and_api import EVENTS, drain, make_client, upload

INTAKE = '{"backhand_style":"one_handed","injury_history":"ankle sprain 2024"}'


def _session(tmp_path, key=""):
    client, worker, s = make_client(tmp_path, FakeDetector(EVENTS), data_key=key)
    sid = upload(client, make_video(tmp_path / "v.mp4", duration=8), INTAKE).json()["id"]
    drain(worker)
    client.post(f"/api/sessions/{sid}/comments", json={"t": 2.0, "body": "private coaching note"})
    plain_video = b"".join(client.app.state.store.vault.read_range(worker.session_dir(sid) / "video.mp4"))
    client.app.state.data_lock.release()  # "stop the server"
    return client, worker, s, sid, plain_video


def _reopen(s, cookie, key):
    s.data_key = key
    c = TestClient(create_app(s, lambda: FakeDetector(EVENTS), start_worker=False))
    c.cookies.set("tl_owner", cookie)
    return c


def _snapshot(data_dir):
    return {p: p.read_bytes() for p in data_dir.rglob("*") if p.is_file() and p.name != ".in-use.lock"}


def test_rotation_moves_everything_to_the_new_key(tmp_path):
    a, b = generate_key(), generate_key()
    client, worker, s, sid, plain_video = _session(tmp_path, a)
    cookie = client.cookies.get("tl_owner")
    rep = rotate_key.rotate(s.data_dir, Vault.from_env(a), Vault.from_env(b))
    assert rep.rotated_files == rep.files["old"] > 3 and rep.rotated_values == rep.values["old"] > 3
    nb = Vault.from_env(b)
    for p in worker.session_dir(sid).iterdir():
        assert p.read_bytes().startswith(MAGIC + nb.key_id), p.name
    assert not list(s.data_dir.rglob("*.rotating"))
    c = _reopen(s, cookie, b)
    body = c.get(f"/api/sessions/{sid}").json()
    assert body["status"] == "complete" and body["intake"]["injury_history"] == "ankle sprain 2024"
    assert c.get(f"/api/sessions/{sid}/comments").json()[0]["body"] == "private coaching note"
    assert c.get(f"/api/sessions/{sid}/video", headers={"Range": "bytes=100-299"}).content == plain_video[100:300]
    c.app.state.data_lock.release()
    assert _reopen(s, cookie, a).get(f"/api/sessions/{sid}").json()["code"] == "storage_key_mismatch"


def test_data_stored_without_a_key_is_encrypted_by_rotation(tmp_path):
    b = generate_key()
    client, worker, s, sid, plain_video = _session(tmp_path)
    assert not (worker.session_dir(sid) / "video.mp4").read_bytes().startswith(MAGIC)
    rep = rotate_key.rotate(s.data_dir, Vault(), Vault.from_env(b))
    assert rep.files["plain"] == rep.rotated_files > 3 and rep.values["plain"] == rep.rotated_values
    assert all(p.read_bytes().startswith(MAGIC) for p in worker.session_dir(sid).iterdir())
    assert b"private coaching note" not in s.db_path.read_bytes() and b"ankle sprain" not in s.db_path.read_bytes()
    assert _reopen(s, client.cookies.get("tl_owner"), b).get(f"/api/sessions/{sid}/video").content == plain_video


def test_refuses_while_a_server_is_running_and_server_waits_for_rotation(tmp_path):
    a, b = generate_key(), generate_key()
    client, _, s, _, _ = _session(tmp_path, a)
    running = DataDirLock(s.data_dir)  # a server holding the directory
    with pytest.raises(RuntimeError, match="running server"):
        rotate_key.rotate(s.data_dir, Vault.from_env(a), Vault.from_env(b))
    running.release()
    tool = DataDirLock(s.data_dir, exclusive=True)
    with pytest.raises(RuntimeError, match="maintenance tool"):
        create_app(s, lambda: FakeDetector(EVENTS), start_worker=False)
    tool.release()


def test_nothing_changes_when_some_data_is_under_an_unknown_key(tmp_path):
    a, b = generate_key(), generate_key()
    _, _, s, _, _ = _session(tmp_path, a)
    before = _snapshot(s.data_dir)
    with pytest.raises(VaultError) as e:
        rotate_key.rotate(s.data_dir, Vault.from_env(generate_key()), Vault.from_env(b))
    assert e.value.code == "key_mismatch" and "nothing was changed" in e.value.message
    with pytest.raises(VaultError):
        rotate_key.rotate(s.data_dir, Vault(), Vault.from_env(b))  # key forgotten: encrypted data cannot be read
    with pytest.raises(VaultError) as e:
        rotate_key.rotate(s.data_dir, Vault.from_env(a), Vault.from_env(a))
    assert e.value.code == "bad_key"
    assert _snapshot(s.data_dir) == before


def test_interrupted_rotation_can_be_run_again(tmp_path, monkeypatch):
    a, b = generate_key(), generate_key()
    client, worker, s, sid, plain_video = _session(tmp_path, a)
    real, calls = os.replace, []

    def crash_on_third(src, dst):
        calls.append(src)
        if len(calls) == 3:
            raise KeyboardInterrupt  # power cut mid-run: a half-done .rotating copy is left behind
        real(src, dst)

    monkeypatch.setattr(rotate_key.os, "replace", crash_on_third)
    with pytest.raises(KeyboardInterrupt):
        rotate_key.rotate(s.data_dir, Vault.from_env(a), Vault.from_env(b))
    monkeypatch.setattr(rotate_key.os, "replace", real)
    assert list(s.data_dir.rglob("*.rotating"))
    rep = rotate_key.rotate(s.data_dir, Vault.from_env(a), Vault.from_env(b))
    assert rep.files["new"] == 2 and rep.rotated_files == rep.files["old"]
    assert not list(s.data_dir.rglob("*.rotating"))
    c = _reopen(s, client.cookies.get("tl_owner"), b)
    assert c.get(f"/api/sessions/{sid}/video").content == plain_video
    c.app.state.data_lock.release()
    again = rotate_key.rotate(s.data_dir, Vault.from_env(a), Vault.from_env(b))  # a third run is a no-op
    assert again.rotated_files == again.rotated_values == 0


def test_check_finds_corruption_and_cli(tmp_path, monkeypatch, capsys):
    a, b = generate_key(), generate_key()
    _, worker, s, sid, _ = _session(tmp_path, a)
    monkeypatch.setenv("DATA_DIR", str(s.data_dir))
    monkeypatch.setenv("DATA_KEY", a)
    assert rotate_key.main(["--check"]) == 0 and "Nothing was changed" in capsys.readouterr().out
    monkeypatch.setenv("NEW_DATA_KEY", b)
    assert rotate_key.main([]) == 0 and "Done:" in capsys.readouterr().out
    monkeypatch.setenv("DATA_KEY", b)
    p = worker.session_dir(sid) / "track.npz"
    raw = bytearray(p.read_bytes())
    raw[-5] ^= 1
    p.write_bytes(bytes(raw))
    assert rotate_key.main(["--check"]) == 1 and "track.npz" in capsys.readouterr().err
