"""HTTP API.

Access model: there are no accounts in this version. Each browser gets a random secret in
an HttpOnly cookie; only its SHA-256 hash is stored, and every session row, video and
derived file is reachable only with the matching cookie. Clearing cookies therefore
orphans the data until the retention period deletes it. This protects against other users
of the same server; it is not a substitute for real authentication.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from . import api_extra
from .config import BACKEND_DIR, Settings, load_settings
from .crypto import Vault, VaultError
from .db import DB
from .pipeline import run
from .pipeline.errors import PipelineError
from .pipeline.pose import Detector, MediaPipeDetector
from .pipeline.segment import STROKE_TYPES
from .pipeline.video import ALLOWED_EXTENSIONS
from .review import apply_review_status
from .storage import Store
from .worker import Worker

COOKIE = "tl_owner"
REGIONS = {"shoulder", "elbow", "wrist", "lower_back", "knee", "ankle", "other"}
GOALS = {"serve", "forehand", "backhand", "consistency", "power", "general"}


def clean_intake(raw: dict) -> dict:
    """Whitelist and bound every intake field; free text is length-limited."""

    def pick(key: str, allowed: set[str], default: str) -> str:
        v = raw.get(key)
        return v if v in allowed else default

    disc = raw.get("discomfort") if isinstance(raw.get("discomfort"), dict) else {}
    return {
        "handedness": pick("handedness", {"right", "left"}, "right"),
        "skill_level": pick("skill_level", {"beginner", "intermediate", "advanced"}, "intermediate"),
        "backhand_style": pick("backhand_style", {"one_handed", "two_handed"}, "two_handed"),
        "goals": [g for g in (raw.get("goals") or []) if g in GOALS][:6],
        "playing_frequency": pick("playing_frequency", {"", "less_than_weekly", "1_2_per_week", "3_4_per_week", "5_plus_per_week"}, ""),
        "workload_change": pick("workload_change", {"", "none", "increased", "decreased"}, ""),
        "discomfort": {
            "current": bool(disc.get("current")),
            "regions": [r for r in (disc.get("regions") or []) if r in REGIONS][:7],
            "notes": str(disc.get("notes") or "")[:500],
        },
        "injury_history": str(raw.get("injury_history") or "")[:500],
    }


def create_app(settings: Settings | None = None, detector_factory: Callable[[], Detector] | None = None, start_worker: bool = True) -> FastAPI:
    settings = settings or load_settings()
    vault = Vault.from_env(settings.data_key)  # raises VaultError("bad_key") for a malformed key
    if settings.require_encryption and not vault.enabled:
        raise VaultError("key_missing", "REQUIRE_ENCRYPTION=1 but DATA_KEY is not set. Generate a key with `python -m app.crypto` and put it in the environment.")
    db = DB(settings.db_path, vault)
    store = Store(settings.sessions_dir, settings.tmp_dir, vault)
    worker = Worker(db, settings, detector_factory or (lambda: MediaPipeDetector(settings.model_path)), store)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if start_worker:
            worker.start()
        yield
        worker.stop()

    app = FastAPI(title="Tennis technique review", lifespan=lifespan)
    app.state.db, app.state.worker, app.state.settings, app.state.store = db, worker, settings, store

    @app.exception_handler(VaultError)
    async def vault_error(_: Request, e: VaultError) -> JSONResponse:  # stored data cannot be read: say why, never return garbage
        return JSONResponse({"detail": e.message, "code": "storage_" + e.code}, status_code=503)

    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    def owner(request: Request, response: Response) -> str:
        token = request.cookies.get(COOKIE)
        if not token or len(token) < 32:
            token = secrets.token_urlsafe(32)
            response.set_cookie(COOKIE, token, max_age=365 * 86400, httponly=True, samesite="lax", secure=settings.cookie_secure)
        return hashlib.sha256(token.encode()).hexdigest()

    def owned(sid: str, who: str) -> dict:
        s = db.get(sid, who)
        if s is None:  # same answer whether it does not exist or belongs to someone else
            raise HTTPException(404, "Session not found")
        return s

    def public(s: dict, full: bool) -> dict:
        out = {
            "id": s["id"], "title": s["title"], "created_at": s["created_at"], "status": s["status"], "stage": s["stage"], "progress": s["progress"],
            "error": {"code": s["error_code"], "message": s["error_message"], "suggestion": s["error_suggestion"]} if s["status"] == "failed" else None,
            "expires_at": s["created_at"] + settings.retention_days * 86400 if settings.retention_days > 0 else None,
            "duration_s": (s["meta"] or {}).get("duration_s"),
            "summary": ((s["report"] or {}).get("summary") or {}).get("headline"), "profile_id": s.get("profile_id"),
            "camera_view": (((s["report"] or {}).get("quality") or {}).get("tracking") or {}).get("camera_view"), "demo": bool((s["meta"] or {}).get("demo")),
        }  # fmt: skip
        if full:
            cands = s["candidates"] or {}
            out |= {"intake": s["intake"], "meta": s["meta"], "quality": s["quality"], "selected": s["selected"], "report": apply_review_status(s["report"], db.signoffs()),
                    "confirmations": s.get("confirmations") or {},
                    "candidates": [{"box": p["box"]} for p in cands.get("people", [])]}  # fmt: skip
        return out

    @app.get("/api/config")
    def config(_: str = Depends(owner)) -> dict:
        return {
            "max_upload_mb": settings.max_upload_mb, "max_duration_s": settings.max_duration_s, "retention_days": settings.retention_days,
            "extensions": sorted(ALLOWED_EXTENSIONS), "supported_strokes": list(STROKE_TYPES), "pose_model": f"MediaPipe Pose Landmarker ({settings.pose_model})", "demo_enabled": settings.enable_demo,
            "encryption_at_rest": vault.enabled, "reviewer_mode": bool(settings.reviewer_code), "llm_summary": bool(settings.llm_api_key),
            "share_ttl_days": settings.share_ttl_days, "share_max_ttl_days": settings.share_max_ttl_days, "options": settings.options().to_dict(),
        }  # fmt: skip

    @app.post("/api/sessions", status_code=201)
    async def create_session(request: Request, file: UploadFile = File(...), intake: str = Form("{}"), title: str = Form(""), profile_id: str = Form(""), who: str = Depends(owner)) -> dict:
        limit = settings.max_upload_mb * 1024 * 1024
        if int(request.headers.get("content-length") or 0) > limit + 1024 * 1024:
            raise HTTPException(413, f"The file is larger than the {settings.max_upload_mb} MB limit.")
        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(415, f"Unsupported file type. Upload one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}.")
        try:
            intake_data = clean_intake(json.loads(intake))
        except (ValueError, AttributeError):
            raise HTTPException(400, "Intake details were not valid.")
        sid = uuid.uuid4().hex
        d = worker.session_dir(sid)
        d.mkdir(parents=True)
        size = 0
        try:
            with store.writer(sid, f"upload{ext}") as out:  # server-chosen name; the client filename never touches the filesystem
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"The file is larger than the {settings.max_upload_mb} MB limit.")
                    out.write(chunk)
            if size == 0:
                raise HTTPException(400, "The uploaded file is empty.")
        except HTTPException:
            shutil.rmtree(d, ignore_errors=True)
            raise
        name = Path(file.filename or "video").name[:120]
        prof = db.get_item(profile_id, "profile", who) if profile_id else None
        db.create(sid, who, (title.strip() or Path(name).stem)[:80], name, intake_data, prof["id"] if prof else None)
        return {"id": sid}

    @app.post("/api/demo", status_code=201)
    def create_demo(who: str = Depends(owner)) -> dict:
        """Synthetic stick-figure session for trying the interface. Off unless ENABLE_DEMO=1; never mixed with real uploads."""
        if not settings.enable_demo:
            raise HTTPException(404, "Demo sessions are not enabled.")
        from .demo import DEMO_MARKER, render_demo_video

        sid = uuid.uuid4().hex
        d = worker.session_dir(sid)
        d.mkdir(parents=True)
        (d / DEMO_MARKER).touch()
        with store.scratch() as tmp:
            store.import_file(sid, "upload.mp4", render_demo_video(tmp / "upload.mp4"))
        db.create(sid, who, "DEMO · synthetic figure (not real analysis)", "demo.mp4", clean_intake({"goals": ["serve"], "backhand_style": "one_handed"}))
        return {"id": sid}

    @app.get("/api/sessions")
    def list_sessions(who: str = Depends(owner)) -> list[dict]:
        return [public(s, False) for s in db.list(who)]

    @app.get("/api/sessions/{sid}")
    def get_session(sid: str, who: str = Depends(owner)) -> dict:
        return public(owned(sid, who), True)

    @app.post("/api/sessions/{sid}/player")
    def choose_player(sid: str, body: dict = Body(...), who: str = Depends(owner)) -> dict:
        s = owned(sid, who)
        people = (s["candidates"] or {}).get("people", [])
        idx = body.get("candidate")
        if not isinstance(idx, int) or not 0 <= idx < len(people):
            raise HTTPException(400, "Choose one of the highlighted players.")
        if s["status"] not in ("awaiting_player", "complete", "failed"):
            raise HTTPException(409, "This session is busy.")
        db.update(sid, selected=idx, status="queued_analyze", attempts=0, stage="Queued", progress=0, report=None)
        return {"ok": True}

    @app.post("/api/sessions/{sid}/retry")
    def retry(sid: str, who: str = Depends(owner)) -> dict:
        s = owned(sid, who)
        if s["status"] != "failed":
            raise HTTPException(409, "Only failed sessions can be retried.")
        phase = s["failed_phase"] or "prepare"
        if phase == "analyze" and not (worker.session_dir(sid) / run.VIDEO).exists():
            phase = "prepare"
        if phase == "prepare" and not list(worker.session_dir(sid).glob("upload.*")) and not (worker.session_dir(sid) / run.VIDEO).exists():
            raise HTTPException(410, "The original upload was removed. Please upload the video again.")
        db.update(sid, status=f"queued_{phase}", attempts=0, stage="Queued", progress=0, error_code=None, error_message=None, error_suggestion=None)
        return {"ok": True}

    @app.put("/api/sessions/{sid}/strokes")
    def edit_strokes(sid: str, body: dict = Body(...), who: str = Depends(owner)) -> dict:
        s = owned(sid, who)
        if s["status"] != "complete" or not s["report"]:
            raise HTTPException(409, "The analysis has not finished.")
        specs = body.get("strokes")
        if not isinstance(specs, list) or len(specs) > 300:
            raise HTTPException(400, "Invalid stroke list.")
        try:
            report = rebuild_report(s, specs)
        except PipelineError as e:
            raise HTTPException(400, e.message)
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, "Invalid stroke list.")
        db.update(sid, report=report)
        db.delete_items("llm_summary", sid)  # a cached rewording no longer matches the findings
        return public(db.get(sid, who), True)

    def rebuild_report(s: dict, specs: list[dict] | None = None, confirmations: dict | None = None) -> dict:
        """Recompute the report from stored pose data (after stroke edits or a re-acquisition decision)."""
        sid = s["id"]
        if specs is None:
            specs = [{"id": x["id"], "start": x["start"], "end": x["end"], "type": x["type"]} for x in s["report"]["strokes"] if x["source"] == "manual"]
            keep_auto = False
        else:
            keep_auto = True
        return run.rebuild(lambda name: store.read_bytes(sid, name) if store.exists(sid, name) else None, meta=s["meta"], quality=s["quality"], intake=s["intake"], stroke_specs=specs,
                           previous=s["report"]["strokes"], confirmations=confirmations if confirmations is not None else (s.get("confirmations") or {}), redetect=not keep_auto, options=settings.options())  # fmt: skip

    @app.get("/api/sessions/{sid}/video")
    def video(sid: str, request: Request, who: str = Depends(owner)):
        owned(sid, who)
        return store.response(sid, run.VIDEO, "video/mp4", request)

    @app.get("/api/sessions/{sid}/candidates.jpg")
    def picker(sid: str, who: str = Depends(owner)):
        owned(sid, who)
        return store.response(sid, run.PICKER, "image/jpeg")

    @app.get("/api/sessions/{sid}/overlay")
    def overlay(sid: str, who: str = Depends(owner)):
        owned(sid, who)
        return store.response(sid, run.OVERLAY, "application/json")

    @app.get("/api/sessions/{sid}/export")
    def export(sid: str, who: str = Depends(owner)):
        s = owned(sid, who)
        if not s["report"]:
            raise HTTPException(409, "No report yet.")
        body = {"title": s["title"], "created_at": s["created_at"], "intake": s["intake"], "report": apply_review_status(s["report"], db.signoffs()),
                "comments": db.items("comment", session_id=sid), "drawings": db.items("drawing", session_id=sid)}
        return JSONResponse(body, headers={"Content-Disposition": f'attachment; filename="tennis-report-{sid[:8]}.json"', "Cache-Control": "private, no-store"})

    @app.delete("/api/sessions/{sid}", status_code=204)
    def delete_session(sid: str, who: str = Depends(owner)) -> Response:
        owned(sid, who)
        worker.delete_session(sid)
        return Response(status_code=204)

    @app.delete("/api/me", status_code=204)
    def delete_everything(response: Response, who: str = Depends(owner)) -> Response:
        for s in db.list(who):
            worker.delete_session(s["id"])
        db.delete_owner(who)  # profiles and anything else tied to this browser
        out = Response(status_code=204)
        out.delete_cookie(COOKIE)
        return out

    @app.get("/api/compare")
    def compare(a: str, b: str, who: str = Depends(owner)) -> dict:
        out = []
        for sid in (a, b):
            s = owned(sid, who)
            if not s["report"]:
                raise HTTPException(409, "Both sessions need a finished report.")
            r = s["report"]
            out.append({"id": sid, "title": s["title"], "created_at": s["created_at"], "camera_view": r["quality"]["tracking"]["camera_view"],
                        "fps": r["meta"]["fps"], "stroke_summary": r["stroke_summary"],
                        "observed": [{"title": f["title"], "stroke_type": f["stroke_type"], "support": f["support"]} for f in r["findings"] if f["status"] == "observed"]})  # fmt: skip
        same_view = out[0]["camera_view"] == out[1]["camera_view"]
        return {"sessions": out, "comparable": same_view,
                "caveat": "Angles and speeds depend on camera position. " + ("Both sessions were filmed from a similar view, but small differences in camera placement still shift the numbers."
                          if same_view else "These sessions were filmed from different views, so differences in the numbers may come from the camera rather than the player.")}  # fmt: skip

    api_extra.register(app, db=db, store=store, worker=worker, settings=settings, owner=owner, owned=owned, public=public, rebuild_report=rebuild_report)

    dist = BACKEND_DIR.parent / "frontend" / "dist"
    if dist.exists():

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            f = (dist / path).resolve()
            if path and f.is_file() and dist.resolve() in f.parents:
                return FileResponse(f)
            return FileResponse(dist / "index.html", headers={"Cache-Control": "no-cache"})

    return app


def app_factory() -> FastAPI:
    return create_app()
