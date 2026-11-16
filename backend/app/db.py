"""SQLite persistence.

sessions           one row per upload; the row is also its own job record
items              everything a person attaches to a session or to this browser: stroke
                   labels, joint annotations, rule reviews, pinned comments, drawings, player
                   profiles, cached LLM summaries. One table so deletion and retention cover
                   all of it in one place. The payload is JSON, encrypted when DATA_KEY is set.
shares             read-only links to one session (only a hash of the token is stored)
signoffs           reviewer sign-off of evidence entries and fixed wording. Not player data:
                   it is not removed when sessions are deleted.

Text and JSON columns that can hold personal or health-related content are passed through
the vault (app/crypto.py) on the way in and out.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from .crypto import Vault

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    status TEXT NOT NULL,           -- queued_prepare | preparing | awaiting_player | queued_analyze | analyzing | complete | failed
    failed_phase TEXT,
    stage TEXT NOT NULL DEFAULT '',
    progress REAL NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    error_code TEXT,
    error_message TEXT,
    error_suggestion TEXT,
    upload_name TEXT,
    intake TEXT NOT NULL DEFAULT '{}',
    meta TEXT,
    quality TEXT,
    candidates TEXT,
    selected INTEGER,
    report TEXT,
    profile_id TEXT,
    confirmations TEXT
);
CREATE INDEX IF NOT EXISTS sessions_owner ON sessions(owner, created_at);
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    session_id TEXT,
    kind TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS items_session ON items(session_id, kind);
CREATE INDEX IF NOT EXISTS items_owner ON items(owner, kind);
CREATE TABLE IF NOT EXISTS shares (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    owner TEXT NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    revoked_at REAL
);
CREATE INDEX IF NOT EXISTS shares_session ON shares(session_id);
CREATE TABLE IF NOT EXISTS signoffs (
    id TEXT PRIMARY KEY,
    target TEXT NOT NULL,           -- evidence:<id> | rule:<id> | text:<id>
    content_hash TEXT NOT NULL,     -- hash of the exact wording that was reviewed
    decision TEXT NOT NULL,         -- approved | changes_requested
    reviewer TEXT NOT NULL,
    role TEXT NOT NULL,
    credentials TEXT NOT NULL,
    comment TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    withdrawn_at REAL
);
CREATE INDEX IF NOT EXISTS signoffs_target ON signoffs(target);
"""
JSON_COLS = ("intake", "meta", "quality", "candidates", "report", "confirmations")
SEALED_COLS = ("title", "upload_name", *JSON_COLS)  # encrypted at rest when a key is configured
MIGRATIONS = {"sessions": {"profile_id": "TEXT", "confirmations": "TEXT"}}


class DB:
    def __init__(self, path: Path, vault: Vault | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.vault = vault or Vault()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            for table, cols in MIGRATIONS.items():  # databases created by the first build
                have = {r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})")}
                for col, typ in cols.items():
                    if col not in have:
                        self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")

    # --- sessions -----------------------------------------------------------------------
    def _row(self, r: sqlite3.Row | None) -> dict | None:
        if r is None:
            return None
        d = dict(r)
        for k in SEALED_COLS:
            if d[k]:
                d[k] = self.vault.open_text(d[k])
        for k in JSON_COLS:
            d[k] = json.loads(d[k]) if d[k] else None
        return d

    def _seal(self, fields: dict) -> dict:
        for k in JSON_COLS:
            if k in fields and fields[k] is not None:
                fields[k] = json.dumps(fields[k])
        for k in SEALED_COLS:
            if fields.get(k) is not None:
                fields[k] = self.vault.seal_text(fields[k])
        return fields

    def create(self, sid: str, owner: str, title: str, upload_name: str, intake: dict, profile_id: str | None = None) -> None:
        f = self._seal({"title": title, "upload_name": upload_name, "intake": intake})
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (id, owner, title, created_at, status, upload_name, intake, profile_id) VALUES (?,?,?,?,?,?,?,?)",
                (sid, owner, f["title"], time.time(), "queued_prepare", f["upload_name"], f["intake"], profile_id),
            )

    def get(self, sid: str, owner: str | None = None) -> dict | None:
        with self._lock:
            if owner is None:
                r = self._conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
            else:
                r = self._conn.execute("SELECT * FROM sessions WHERE id=? AND owner=?", (sid, owner)).fetchone()
        return self._row(r)

    def list(self, owner: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM sessions WHERE owner=? ORDER BY created_at DESC", (owner,)).fetchall()
        return [self._row(r) for r in rows]

    def update(self, sid: str, **fields) -> None:
        if not fields:
            return
        fields = self._seal(fields)
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._conn.execute(f"UPDATE sessions SET {cols} WHERE id=?", (*fields.values(), sid))

    def delete(self, sid: str) -> None:
        """Remove a session and everything attached to it: labels, annotations, reviews, comments, drawings, summaries, share links."""
        with self._lock:
            self._conn.execute("DELETE FROM items WHERE session_id=?", (sid,))
            self._conn.execute("DELETE FROM shares WHERE session_id=?", (sid,))
            self._conn.execute("DELETE FROM sessions WHERE id=?", (sid,))

    def delete_owner(self, owner: str) -> None:
        """Remove what is left for a browser after its sessions are gone (profiles and any stray items or links)."""
        with self._lock:
            self._conn.execute("DELETE FROM items WHERE owner=?", (owner,))
            self._conn.execute("DELETE FROM shares WHERE owner=?", (owner,))

    def claim(self) -> dict | None:
        """Atomically take the oldest queued job."""
        with self._lock:
            r = self._conn.execute("SELECT * FROM sessions WHERE status IN ('queued_prepare','queued_analyze') ORDER BY created_at LIMIT 1").fetchone()
            if r is None:
                return None
            new = "preparing" if r["status"] == "queued_prepare" else "analyzing"
            self._conn.execute("UPDATE sessions SET status=?, attempts=attempts+1, progress=0, stage='Starting' WHERE id=?", (new, r["id"]))
            d = self._row(r)
            d["status"], d["attempts"] = new, d["attempts"] + 1
            return d

    def requeue_interrupted(self) -> None:
        """Jobs that were running when the server stopped go back in the queue."""
        with self._lock:
            self._conn.execute("UPDATE sessions SET status='queued_prepare' WHERE status='preparing'")
            self._conn.execute("UPDATE sessions SET status='queued_analyze' WHERE status='analyzing'")

    def expired(self, before: float) -> list[str]:
        with self._lock:
            return [r["id"] for r in self._conn.execute("SELECT id FROM sessions WHERE created_at < ?", (before,)).fetchall()]

    # --- items --------------------------------------------------------------------------
    def _item(self, r: sqlite3.Row) -> dict:
        return {"id": r["id"], "session_id": r["session_id"], "kind": r["kind"], "created_at": r["created_at"], "updated_at": r["updated_at"], **json.loads(self.vault.open_text(r["data"]))}

    def put_item(self, owner: str, kind: str, data: dict, session_id: str | None = None, item_id: str | None = None) -> dict:
        now, body = time.time(), self.vault.seal_text(json.dumps(data))
        with self._lock:
            if item_id:
                cur = self._conn.execute("UPDATE items SET data=?, updated_at=? WHERE id=? AND owner=? AND kind=?", (body, now, item_id, owner, kind))
                if cur.rowcount == 0:
                    raise KeyError(item_id)
            else:
                item_id = uuid.uuid4().hex[:16]
                self._conn.execute("INSERT INTO items (id, owner, session_id, kind, created_at, updated_at, data) VALUES (?,?,?,?,?,?,?)", (item_id, owner, session_id, kind, now, now, body))
            r = self._conn.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
        return self._item(r)

    def items(self, kind: str, *, session_id: str | None = None, owner: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM items WHERE kind=?", [kind]
        if session_id is not None:
            q, args = q + " AND session_id=?", [*args, session_id]
        if owner is not None:
            q, args = q + " AND owner=?", [*args, owner]
        with self._lock:
            rows = self._conn.execute(q + " ORDER BY created_at", args).fetchall()
        return [self._item(r) for r in rows]

    def get_item(self, item_id: str, kind: str, owner: str | None = None) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM items WHERE id=? AND kind=?" + (" AND owner=?" if owner else ""), (item_id, kind, owner) if owner else (item_id, kind)).fetchone()
        return self._item(r) if r else None

    def delete_item(self, item_id: str, kind: str, owner: str) -> bool:
        with self._lock:
            return self._conn.execute("DELETE FROM items WHERE id=? AND kind=? AND owner=?", (item_id, kind, owner)).rowcount > 0

    def delete_items(self, kind: str, session_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM items WHERE kind=? AND session_id=?", (kind, session_id))

    def clear_profile(self, owner: str, profile_id: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE sessions SET profile_id=NULL WHERE owner=? AND profile_id=?", (owner, profile_id))

    def count_items(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] + self._conn.execute("SELECT COUNT(*) FROM shares").fetchone()[0]

    # --- share links --------------------------------------------------------------------
    def add_share(self, session_id: str, owner: str, token_hash: str, expires_at: float) -> dict:
        sid = uuid.uuid4().hex[:12]
        with self._lock:
            self._conn.execute("INSERT INTO shares (id, session_id, owner, token_hash, created_at, expires_at) VALUES (?,?,?,?,?,?)", (sid, session_id, owner, token_hash, time.time(), expires_at))
        return {"id": sid, "session_id": session_id, "created_at": time.time(), "expires_at": expires_at, "revoked_at": None}

    def shares(self, session_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT id, session_id, created_at, expires_at, revoked_at FROM shares WHERE session_id=? ORDER BY created_at", (session_id,)).fetchall()
        return [dict(r) for r in rows]

    def share_by_hash(self, token_hash: str) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM shares WHERE token_hash=?", (token_hash,)).fetchone()
        return dict(r) if r else None

    def revoke_share(self, share_id: str, session_id: str) -> bool:
        with self._lock:
            return self._conn.execute("UPDATE shares SET revoked_at=? WHERE id=? AND session_id=? AND revoked_at IS NULL", (time.time(), share_id, session_id)).rowcount > 0

    def purge_shares(self, before: float) -> int:
        """Drop link records that expired or were revoked before the given time."""
        with self._lock:
            return self._conn.execute("DELETE FROM shares WHERE expires_at < ? OR (revoked_at IS NOT NULL AND revoked_at < ?)", (before, before)).rowcount

    # --- reviewer sign-off --------------------------------------------------------------
    def add_signoff(self, target: str, content_hash: str, decision: str, reviewer: str, role: str, credentials: str, comment: str) -> dict:
        sid = uuid.uuid4().hex[:12]
        with self._lock:
            self._conn.execute("INSERT INTO signoffs (id, target, content_hash, decision, reviewer, role, credentials, comment, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                               (sid, target, content_hash, decision, reviewer, role, credentials, comment, time.time()))  # fmt: skip
        return {"id": sid}

    def signoffs(self, include_withdrawn: bool = False) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM signoffs" + ("" if include_withdrawn else " WHERE withdrawn_at IS NULL") + " ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    def withdraw_signoff(self, signoff_id: str) -> bool:
        with self._lock:
            return self._conn.execute("UPDATE signoffs SET withdrawn_at=? WHERE id=? AND withdrawn_at IS NULL", (time.time(), signoff_id)).rowcount > 0
