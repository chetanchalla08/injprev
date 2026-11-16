"""Where session files live. Every read and write of stored data goes through here so that
encryption at rest (app/crypto.py) and deletion apply to everything, old and new.

Stored files keep their plain names (video.mp4, track.npz, ...); when DATA_KEY is set their
contents are ciphertext. Processing that needs a real file (ffmpeg, OpenCV) works in a
scratch directory under DATA_DIR/tmp that is removed afterwards and wiped at start-up.
"""
from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse

from .crypto import Vault, VaultError


class Store:
    def __init__(self, sessions_dir: Path, tmp_dir: Path, vault: Vault) -> None:
        self.root, self.tmp, self.vault = sessions_dir, tmp_dir, vault
        self.root.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(self.tmp, ignore_errors=True)  # plaintext left behind by a crash mid-processing
        self.tmp.mkdir(parents=True, exist_ok=True)

    def dir(self, sid: str) -> Path:
        return self.root / sid

    def path(self, sid: str, name: str) -> Path:
        return self.root / sid / name

    def exists(self, sid: str, name: str) -> bool:
        return self.path(sid, name).exists()

    def write_bytes(self, sid: str, name: str, data: bytes) -> None:
        self.dir(sid).mkdir(parents=True, exist_ok=True)
        self.path(sid, name).write_bytes(self.vault.encrypt(data))

    def read_bytes(self, sid: str, name: str) -> bytes:
        return self.vault.read_file(self.path(sid, name))

    def import_file(self, sid: str, name: str, src: Path) -> None:
        """Move a plaintext file produced in scratch space into the session (encrypting it if a key is set)."""
        self.dir(sid).mkdir(parents=True, exist_ok=True)
        dst = self.path(sid, name)
        if self.vault.enabled:
            self.vault.encrypt_file(src, dst)
            src.unlink(missing_ok=True)
        else:
            shutil.move(str(src), str(dst))

    def writer(self, sid: str, name: str):
        self.dir(sid).mkdir(parents=True, exist_ok=True)
        return self.vault.writer(self.path(sid, name))

    @contextmanager
    def scratch(self) -> Iterator[Path]:
        d = Path(tempfile.mkdtemp(dir=self.tmp))
        try:
            yield d
        finally:
            shutil.rmtree(d, ignore_errors=True)

    @contextmanager
    def plain(self, sid: str, name: str) -> Iterator[Path]:
        """A real plaintext file for tools that need one. The stored file itself when it is not encrypted."""
        p = self.path(sid, name)
        if not self.vault.is_encrypted(p):
            yield p
            return
        with self.scratch() as d:
            out = d / name
            self.vault.decrypt_file(p, out)
            yield out

    def delete(self, sid: str) -> None:
        shutil.rmtree(self.dir(sid), ignore_errors=True)

    def response(self, sid: str, name: str, media: str, request: Request | None = None) -> Response:
        p = self.path(sid, name)
        if not p.exists():
            raise HTTPException(404, "Not available")
        headers = {"Cache-Control": "private, no-store"}
        if not self.vault.is_encrypted(p):
            return FileResponse(p, media_type=media, headers=headers)
        try:
            total = self.vault.plain_size(p)
            rng = request.headers.get("range") if request else None
            if rng and rng.startswith("bytes="):
                a, _, b = rng[6:].split(",")[0].partition("-")
                start = int(a) if a else max(total - int(b), 0)
                end = min(int(b), total - 1) if a and b else total - 1
                if start >= total or start > end:
                    return Response(status_code=416, headers={"Content-Range": f"bytes */{total}"})
                first = self.vault.read_range(p, start, end)
                head = next(first, b"")  # decrypt the first chunk now so key problems surface as an error, not a broken stream
                body = (c for part in ([head], first) for c in part)
                return StreamingResponse(body, status_code=206, media_type=media, headers={**headers, "Accept-Ranges": "bytes", "Content-Range": f"bytes {start}-{end}/{total}", "Content-Length": str(end - start + 1)})
            it = self.vault.read_range(p)
            head = next(it, b"")
            return StreamingResponse((c for part in ([head], it) for c in part), media_type=media, headers={**headers, "Accept-Ranges": "bytes", "Content-Length": str(total)})
        except VaultError as e:
            raise HTTPException(503, e.message)
        except ValueError:
            raise HTTPException(400, "Bad Range header")
