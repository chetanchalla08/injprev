"""Encryption at rest with one server-held key (DATA_KEY in the environment).

What it does: videos, pose data, overlays, exports and the JSON/text columns that hold
intake details, reports, comments, drawings and labels are stored as AES-256-GCM
ciphertext. Files are encrypted in 64 KiB chunks so a byte range of a video can be served
without decrypting the whole file.

What it protects against: someone who obtains the data directory or a backup of it
(a stolen disk, a mis-shared volume, a leaked backup) without also obtaining the key.

What it does NOT protect against: anyone who can read the server's environment or memory
(the key is there), a compromised application, or the hosting provider if they hold both.
Plaintext also exists briefly in DATA_DIR/tmp while a video is being processed, because
ffmpeg and OpenCV need a real file. It is not end-to-end encryption and it is not a
compliance claim. See README "Encryption at rest".
"""
from __future__ import annotations

import base64
import hashlib
import os
import struct
from pathlib import Path
from typing import BinaryIO, Iterator

MAGIC = b"BLENC1\n"
TEXT_PREFIX = "enc1:"
CHUNK = 64 * 1024
TAG = 16
HEADER = len(MAGIC) + 4 + 8  # magic, key id, per-file nonce prefix


class VaultError(Exception):
    """Raised when stored data cannot be decrypted. code: key_missing | key_mismatch | corrupt | bad_key"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


def generate_key() -> str:
    return base64.urlsafe_b64encode(os.urandom(32)).decode()


class Vault:
    def __init__(self, key: bytes | None = None) -> None:
        self._aes = None
        self.key_id = b""
        if key is not None:
            if len(key) != 32:
                raise VaultError("bad_key", "DATA_KEY must decode to exactly 32 bytes. Generate one with: python -m app.crypto")
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            self._aes = AESGCM(key)
            self.key_id = hashlib.sha256(b"baseline-key-id" + key).digest()[:4]

    @classmethod
    def from_env(cls, value: str | None) -> "Vault":
        if not value:
            return cls(None)
        try:
            key = base64.urlsafe_b64decode(value.strip() + "=" * (-len(value.strip()) % 4))
        except Exception as e:
            raise VaultError("bad_key", "DATA_KEY is not valid base64. Generate one with: python -m app.crypto") from e
        return cls(key)

    @property
    def enabled(self) -> bool:
        return self._aes is not None

    # --- header / chunk helpers ---------------------------------------------------------
    def _check(self, header: bytes) -> bytes:
        """Validate an encrypted header and return the per-file nonce prefix."""
        if not self.enabled:
            raise VaultError("key_missing", "This data is encrypted but no DATA_KEY is configured on the server.")
        if header[len(MAGIC) : len(MAGIC) + 4] != self.key_id:
            raise VaultError("key_mismatch", "This data was encrypted with a different DATA_KEY than the one configured.")
        return header[len(MAGIC) + 4 : HEADER]

    def _seal(self, prefix: bytes, i: int, last: bool, pt: bytes) -> bytes:
        return self._aes.encrypt(prefix + struct.pack(">I", i), pt, MAGIC + self.key_id + (b"L" if last else b"M"))

    def _open(self, prefix: bytes, i: int, last: bool, ct: bytes) -> bytes:
        from cryptography.exceptions import InvalidTag

        try:
            return self._aes.decrypt(prefix + struct.pack(">I", i), ct, MAGIC + self.key_id + (b"L" if last else b"M"))
        except InvalidTag as e:
            raise VaultError("corrupt", "Stored data failed its integrity check (corrupted, truncated or tampered with).") from e

    # --- bytes --------------------------------------------------------------------------
    def encrypt(self, data: bytes) -> bytes:
        if not self.enabled:
            return data
        prefix = os.urandom(8)
        out = [MAGIC, self.key_id, prefix]
        n = max(1, -(-len(data) // CHUNK))
        for i in range(n):
            out.append(self._seal(prefix, i, i == n - 1, data[i * CHUNK : (i + 1) * CHUNK]))
        return b"".join(out)

    def decrypt(self, blob: bytes) -> bytes:
        if not blob.startswith(MAGIC):
            return blob  # stored before encryption was switched on
        prefix = self._check(blob[:HEADER])
        body, step = blob[HEADER:], CHUNK + TAG
        n = max(1, -(-len(body) // step))
        return b"".join(self._open(prefix, i, i == n - 1, body[i * step : (i + 1) * step]) for i in range(n))

    def seal_text(self, text: str) -> str:
        return TEXT_PREFIX + base64.b64encode(self.encrypt(text.encode())).decode() if self.enabled else text

    def open_text(self, text: str) -> str:
        if not text.startswith(TEXT_PREFIX):
            return text
        return self.decrypt(base64.b64decode(text[len(TEXT_PREFIX) :])).decode()

    # --- files --------------------------------------------------------------------------
    def writer(self, path: Path) -> "_Writer":
        return _Writer(self, path)

    def encrypt_file(self, src: Path, dst: Path) -> None:
        with open(src, "rb") as f, self.writer(dst) as w:
            while chunk := f.read(1024 * 1024):
                w.write(chunk)

    @staticmethod
    def is_encrypted(path: Path) -> bool:
        with open(path, "rb") as f:
            return f.read(len(MAGIC)) == MAGIC

    def plain_size(self, path: Path) -> int:
        size = path.stat().st_size
        if not self.is_encrypted(path):
            return size
        body = size - HEADER
        return body - TAG * max(1, -(-body // (CHUNK + TAG)))

    def read_range(self, path: Path, start: int = 0, end: int | None = None) -> Iterator[bytes]:
        """Yield plaintext bytes [start, end] (inclusive end, like an HTTP Range) of a stored file."""
        if not self.is_encrypted(path):
            with open(path, "rb") as f:
                f.seek(start)
                left = (end - start + 1) if end is not None else None
                while left is None or left > 0:
                    chunk = f.read(CHUNK if left is None else min(CHUNK, left))
                    if not chunk:
                        return
                    if left is not None:
                        left -= len(chunk)
                    yield chunk
            return
        total = self.plain_size(path)
        end = total - 1 if end is None else min(end, total - 1)
        step = CHUNK + TAG
        n = max(1, -(-(path.stat().st_size - HEADER) // step))
        with open(path, "rb") as f:
            prefix = self._check(f.read(HEADER))
            for i in range(start // CHUNK, end // CHUNK + 1):
                f.seek(HEADER + i * step)
                pt = self._open(prefix, i, i == n - 1, f.read(step))
                lo = start - i * CHUNK if i == start // CHUNK else 0
                hi = end - i * CHUNK + 1 if i == end // CHUNK else len(pt)
                yield pt[lo:hi]

    def read_file(self, path: Path) -> bytes:
        return b"".join(self.read_range(path))

    def decrypt_file(self, src: Path, dst: Path) -> None:
        with open(dst, "wb") as out:
            for chunk in self.read_range(src):
                out.write(chunk)


class _Writer:
    """Streaming writer: plaintext passthrough when no key is set, chunked AES-GCM otherwise."""

    def __init__(self, vault: Vault, path: Path) -> None:
        self.v, self.f, self.buf, self.i = vault, open(path, "wb"), b"", 0
        self.prefix = os.urandom(8)
        if vault.enabled:
            self.f.write(MAGIC + vault.key_id + self.prefix)

    def write(self, data: bytes) -> None:
        if not self.v.enabled:
            self.f.write(data)
            return
        self.buf += data
        while len(self.buf) > CHUNK:  # keep at least one byte back so the final chunk can be marked as last
            self.f.write(self.v._seal(self.prefix, self.i, False, self.buf[:CHUNK]))
            self.buf, self.i = self.buf[CHUNK:], self.i + 1

    def close(self) -> None:
        if self.v.enabled:
            self.f.write(self.v._seal(self.prefix, self.i, True, self.buf))
        self.f.close()

    def __enter__(self) -> "_Writer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _open_binary(path: Path) -> BinaryIO:  # pragma: no cover - small helper kept for scripts
    return open(path, "rb")


if __name__ == "__main__":
    print("Add this line to backend/.env (keep a copy somewhere safe: without it the stored data cannot be read):")
    print(f"DATA_KEY={generate_key()}")
