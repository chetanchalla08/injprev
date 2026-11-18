"""Re-encrypt everything stored under DATA_DIR with a new DATA_KEY.

    .venv/bin/python -m app.rotate_key --check          # read everything with the current key, change nothing
    NEW_DATA_KEY=... .venv/bin/python -m app.rotate_key # rotate (prompts for the new key if NEW_DATA_KEY is unset)
    .venv/bin/python -m app.rotate_key --generate       # rotate to a freshly generated key and print it

The current key is DATA_KEY (environment or backend/.env), as the server reads it. Data
stored before any key was set is encrypted under the new key too, so after a successful run
nothing in the data directory is readable without the new key.

Covered: every file under DATA_DIR/sessions and the encrypted database columns (session
titles, upload names, intake, reports and other JSON columns; all item payloads). Share-link
records, sign-offs and job status are not encrypted and are left alone.

Safety:
- The server must be stopped. The tool takes an exclusive lock on DATA_DIR and refuses to
  run while a server holds it; a server will not start while the tool is running.
- Before changing anything every stored value is classified by the key it was written with;
  if anything is under neither the current nor the new key, nothing is changed.
- Files are rewritten to a temporary name, flushed to disk, then renamed over the original,
  so each file is always whole under one key or the other. The database is rewritten in one
  transaction. If the run is interrupted, run it again with the same two keys: values already
  under the new key are skipped.
- Backups taken before rotation still need the OLD key. Keep it until those have expired.
"""
from __future__ import annotations

import argparse
import base64
import getpass
import os
import shutil
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .crypto import HEADER, MAGIC, TEXT_PREFIX, Vault, VaultError, generate_key
from .db import SEALED_COLS
from .storage import DataDirLock

TMP_SUFFIX = ".rotating"


@dataclass
class Report:
    files: dict[str, int] = field(default_factory=lambda: {"plain": 0, "old": 0, "new": 0})
    values: dict[str, int] = field(default_factory=lambda: {"plain": 0, "old": 0, "new": 0})
    rotated_files: int = 0
    rotated_values: int = 0


def _state(header: bytes, old: Vault, new: Vault) -> str:
    """plain | old | new | unknown for the start of a stored blob."""
    if not header.startswith(MAGIC):
        return "plain"
    kid = header[len(MAGIC) : len(MAGIC) + 4]
    if new.enabled and kid == new.key_id:
        return "new"
    if old.enabled and kid == old.key_id:
        return "old"
    return "unknown"


def _text_state(value: str, old: Vault, new: Vault) -> str:
    if not value.startswith(TEXT_PREFIX):
        return "plain"
    try:
        head = base64.b64decode(value[len(TEXT_PREFIX) :])[:HEADER]
    except Exception:
        return "unknown"
    return _state(head, old, new)


def _files(sessions: Path) -> list[Path]:
    return sorted(p for p in sessions.rglob("*") if p.is_file() and not p.name.endswith(TMP_SUFFIX)) if sessions.exists() else []


def _db_values(conn: sqlite3.Connection) -> list[tuple[str, str, str, str]]:
    """(table, column, row id, value) for every non-empty encrypted column."""
    out = []
    cols = ", ".join(SEALED_COLS)
    for r in conn.execute(f"SELECT id, {cols} FROM sessions"):
        for i, c in enumerate(SEALED_COLS, 1):
            if r[i]:
                out.append(("sessions", c, r[0], r[i]))
    for r in conn.execute("SELECT id, data FROM items"):
        if r[1]:
            out.append(("items", "data", r[0], r[1]))
    return out


def rotate(data_dir: Path, old: Vault, new: Vault, check_only: bool = False, log: Callable[[str], None] = lambda _: None) -> Report:
    """Re-encrypt DATA_DIR from `old` (may be disabled: data was never encrypted) to `new`.
    With check_only, decrypt everything with `old` to prove it is readable and change nothing."""
    if not new.enabled and not check_only:
        raise VaultError("key_missing", "A new key is required.")
    if new.enabled and old.enabled and new.key_id == old.key_id:
        raise VaultError("bad_key", "The new key is the same as the current one.")
    lock = DataDirLock(data_dir, exclusive=True)
    try:
        sessions, db_path = data_dir / "sessions", data_dir / "app.sqlite3"
        for p in sessions.rglob(f"*{TMP_SUFFIX}") if sessions.exists() else []:
            p.unlink()  # half-written copy from an interrupted run; the original is still intact
        shutil.rmtree(data_dir / "tmp", ignore_errors=True)  # plaintext working copies left by a crash
        rep = Report()

        # 1. classify everything before touching anything
        files = _files(sessions)
        file_state = {}
        for p in files:
            with open(p, "rb") as f:
                file_state[p] = _state(f.read(HEADER), old, new)
        conn = sqlite3.connect(db_path, isolation_level=None) if db_path.exists() else None
        values = _db_values(conn) if conn else []
        value_state = [_text_state(v[3], old, new) for v in values]
        bad = [str(p.relative_to(data_dir)) for p, s in file_state.items() if s == "unknown"]
        bad += [f"{t}.{c} of {rid}" for (t, c, rid, _), s in zip(values, value_state) if s == "unknown"]
        if bad:
            hint = "" if old.enabled else " DATA_KEY is not set, so only unencrypted data can be read."
            raise VaultError("key_mismatch", f"{len(bad)} stored item(s) are encrypted with neither the current nor the new key, so nothing was changed.{hint} First: {', '.join(bad[:5])}")
        for s in file_state.values():
            rep.files[s] += 1
        for s in value_state:
            rep.values[s] += 1
        log(f"Files: {rep.files['old']} under the current key, {rep.files['plain']} unencrypted, {rep.files['new']} already under the new key.")
        log(f"Database values: {rep.values['old']} under the current key, {rep.values['plain']} unencrypted, {rep.values['new']} already under the new key.")

        # 2. files: decrypt with old (verifies every chunk), write under new, swap in atomically
        for p in files:
            if file_state[p] == "new":
                continue
            try:
                if check_only:
                    for _ in old.read_range(p):
                        pass
                    continue
                tmp = p.with_name(p.name + TMP_SUFFIX)
                with new.writer(tmp) as w:
                    for chunk in old.read_range(p):
                        w.write(chunk)
                fd = os.open(tmp, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
                os.replace(tmp, p)
                rep.rotated_files += 1
            except VaultError as e:
                raise VaultError(e.code, f"{p.relative_to(data_dir)}: {e.message} Files before it were rotated; nothing after it was. Fix or remove it, then run again.") from e

        # 3. database: one transaction
        if conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                for (table, col, rid, v), s in zip(values, value_state):
                    if s == "new":
                        continue
                    try:
                        plain = old.open_text(v)
                    except VaultError as e:
                        raise VaultError(e.code, f"{table}.{col} of {rid}: {e.message} The database was not changed.") from e
                    if not check_only:
                        conn.execute(f"UPDATE {table} SET {col} = ? WHERE id = ?", (new.seal_text(plain), rid))
                        rep.rotated_values += 1
                conn.execute("ROLLBACK" if check_only else "COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()
        return rep
    finally:
        lock.release()


def _read_new_key(generate: bool) -> str:
    if generate:
        key = generate_key()
        print("New key (put it in DATA_KEY once this finishes; keep a copy outside the data directory):")
        print(f"DATA_KEY={key}\n")
        return key
    key = os.environ.get("NEW_DATA_KEY", "").strip()
    if key:
        return key
    if not sys.stdin.isatty():
        raise SystemExit("Set NEW_DATA_KEY, or pass --generate. (The key is not taken as an argument so it stays out of shell history and process lists.)")
    key = getpass.getpass("New DATA_KEY (generate one with `python -m app.crypto`): ").strip()
    if key != getpass.getpass("Again: ").strip():
        raise SystemExit("The two entries differ; nothing was changed.")
    return key


def main(argv: list[str] | None = None) -> int:
    from .config import load_settings

    ap = argparse.ArgumentParser(prog="python -m app.rotate_key", description="Re-encrypt all stored data under a new DATA_KEY. Stop the server first.")
    ap.add_argument("--check", action="store_true", help="only verify that everything can be read with the current DATA_KEY")
    ap.add_argument("--generate", action="store_true", help="generate the new key and print it")
    args = ap.parse_args(argv)
    settings = load_settings()
    try:
        old = Vault.from_env(settings.data_key)
        new = Vault() if args.check else Vault.from_env(_read_new_key(args.generate))
        print(f"Data directory: {settings.data_dir}")
        rep = rotate(settings.data_dir, old, new, check_only=args.check, log=print)
    except (VaultError, RuntimeError) as e:
        print(f"Error: {getattr(e, 'message', e)}", file=sys.stderr)
        return 1
    if args.check:
        print("Everything stored can be read with the current key. Nothing was changed.")
        return 0
    print(f"Done: {rep.rotated_files} file(s) and {rep.rotated_values} database value(s) re-encrypted under the new key.")
    print("Next: set DATA_KEY to the new key (backend/.env or the server environment) and start the server.")
    print("Keep the old key until every backup taken before now has expired: those backups still need it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
