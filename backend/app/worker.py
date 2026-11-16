"""Background worker: one thread that runs queued jobs, retries unexpected failures and purges expired sessions."""
from __future__ import annotations

import logging
import threading
import time
import traceback
from pathlib import Path
from typing import Callable

from .config import Settings
from .crypto import VaultError
from .db import DB
from .demo import DEMO_MARKER, demo_detector
from .pipeline import run
from .pipeline.errors import PipelineError
from .pipeline.pose import Detector
from .storage import Store

log = logging.getLogger("worker")


class Worker:
    def __init__(self, db: DB, settings: Settings, detector_factory: Callable[[], Detector], store: Store | None = None) -> None:
        self.db, self.settings, self.detector_factory = db, settings, detector_factory
        self.store = store or Store(settings.sessions_dir, settings.tmp_dir, db.vault)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_purge = 0.0

    def session_dir(self, sid: str) -> Path:
        return self.settings.sessions_dir / sid

    def start(self) -> None:
        self.db.requeue_interrupted()
        self._thread = threading.Thread(target=self._loop, name="analysis-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.purge_expired()
                if not self.run_one():
                    self._stop.wait(0.5)
            except Exception:  # never let the loop die
                log.exception("worker loop error")
                self._stop.wait(2)

    def purge_expired(self, force: bool = False) -> int:
        days = self.settings.retention_days
        if days <= 0 or (not force and time.time() - self._last_purge < 3600):
            return 0
        self._last_purge = time.time()
        ids = self.db.expired(time.time() - days * 86400)
        for sid in ids:
            self.delete_session(sid)
        self.db.purge_shares(time.time() - 86400)  # link records are kept one day past expiry or revocation, then dropped
        return len(ids)

    def delete_session(self, sid: str) -> None:
        """Remove the database row, everything attached to it (labels, annotations, reviews, comments, drawings, summaries, share links) and every file derived from the upload."""
        self.db.delete(sid)
        self.store.delete(sid)

    def run_one(self) -> bool:
        job = self.db.claim()
        if job is None:
            return False
        sid, phase = job["id"], "prepare" if job["status"] == "preparing" else "analyze"
        d = self.session_dir(sid)

        def progress(stage: str, frac: float) -> None:
            self.db.update(sid, stage=stage, progress=round(float(frac), 3))

        try:
            is_demo = (d / DEMO_MARKER).exists()
            detector = demo_detector() if is_demo else self.detector_factory()
            if phase == "prepare":
                uploads = sorted(d.glob("upload.*"))
                if not uploads and not (d / run.VIDEO).exists():
                    raise PipelineError("upload_missing", "The uploaded file is no longer available.", "Upload the video again.")
                if uploads:
                    with self.store.scratch() as tmp, self.store.plain(sid, uploads[0].name) as up:
                        res = run.prepare(tmp, up, detector, max_duration_s=self.settings.max_duration_s, min_duration_s=self.settings.min_duration_s, progress=progress)
                        self.store.import_file(sid, run.VIDEO, tmp / run.VIDEO)
                        self.store.import_file(sid, run.PICKER, tmp / run.PICKER)
                    uploads[0].unlink(missing_ok=True)  # the original is not kept once the normalised copy exists
                    res["meta"]["demo"] = is_demo
                    self.db.update(sid, meta=res["meta"], quality=res["quality"], candidates=res["candidates"])
                    people = res["candidates"]["people"]
                else:  # retry after the original was already converted
                    people = (self.db.get(sid) or {}).get("candidates", {}).get("people", [])
                if self.db.get(sid) is None:
                    self.store.delete(sid)
                    return True  # deleted while processing
                if len(people) == 1:
                    self.db.update(sid, selected=0, status="queued_analyze", attempts=0, stage="Queued", progress=0)
                else:
                    self.db.update(sid, status="awaiting_player", attempts=0, stage="Waiting for you to pick the player", progress=1)
            else:
                with self.store.scratch() as tmp, self.store.plain(sid, run.VIDEO) as video:
                    report = run.analyze(tmp, detector, meta=job["meta"], quality=job["quality"], candidates=job["candidates"], selected=job["selected"] or 0, intake=job["intake"],
                                         max_fps=self.settings.analysis_max_fps, progress=progress, options=self.settings.options(), video=video, confirmations=job.get("confirmations") or {},
                                         object_model=self.settings.object_model_path)  # fmt: skip
                    if self.db.get(sid) is None:
                        self.store.delete(sid)
                        return True
                    for f in sorted(tmp.iterdir()):
                        self.store.import_file(sid, f.name, f)
                self.db.update(sid, report=report, status="complete", stage="Done", progress=1, error_code=None, error_message=None, error_suggestion=None)
        except VaultError as e:
            self.db.update(sid, status="failed", failed_phase=phase, error_code="storage_" + e.code, error_message=e.message, error_suggestion="This needs the server operator: check the DATA_KEY setting.")
        except PipelineError as e:
            self.db.update(sid, status="failed", failed_phase=phase, error_code=e.code, error_message=e.message, error_suggestion=e.suggestion)
            if e.code in ("unsupported_file", "decode_failed", "too_long", "too_short", "unreadable_file"):
                for f in d.glob("upload.*"):  # nothing useful to keep from a rejected upload
                    f.unlink(missing_ok=True)
        except Exception as e:
            log.error("job %s (%s) failed on attempt %s: %s\n%s", sid, phase, job["attempts"], e, traceback.format_exc())
            if job["attempts"] < self.settings.max_attempts:
                self.db.update(sid, status=f"queued_{phase}", stage="Retrying after an error")
            else:
                self.db.update(sid, status="failed", failed_phase=phase, error_code="internal_error",
                               error_message="Something went wrong while processing this video.", error_suggestion="You can retry. If it keeps failing, try a different clip.")  # fmt: skip
        return True
