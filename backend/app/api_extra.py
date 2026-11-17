"""API added after the first build: evaluation tooling (labels, joint annotations, reviews),
coach features (profiles, comments, drawings, share links), clinician features (PDF,
angle plots), re-acquisition confirmation and the optional LLM summary.

Everything a user creates here is stored as an ``items`` row tied to a session or to the
browser, so deleting a session, deleting everything, or retention expiry removes it too.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import secrets
import time

from fastapi import Body, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from . import angles, review
from .evaltools.format import ANNOTATION_FORMAT, CAMERA_VIEWS, LABEL_FORMAT, FormatError, parse_labels
from .pipeline import run
from .pipeline.signals import PoseTrack

NOT_FOUND_SHARE = "This link is not valid, has expired or was revoked."
SHAPES = {"line": (2, 2), "angle": (3, 3), "free": (2, 400)}  # shape -> (min points, max points)


def _text(v, limit: int) -> str:
    return str(v or "").strip()[:limit]


def register(app: FastAPI, *, db, store, worker, settings, owner, owned, public, rebuild_report) -> None:
    def complete(sid: str, who: str) -> dict:
        s = owned(sid, who)
        if s["status"] != "complete" or not s["report"]:
            raise HTTPException(409, "The analysis has not finished.")
        return s

    def track_of(sid: str) -> PoseTrack:
        if not store.exists(sid, run.TRACK):
            raise HTTPException(409, "Pose data is not available for this session.")
        return PoseTrack.load(io.BytesIO(store.read_bytes(sid, run.TRACK)))

    def single(kind: str, sid: str) -> dict | None:
        found = db.items(kind, session_id=sid)
        return found[0] if found else None

    def put_single(kind: str, sid: str, who: str, data: dict) -> dict:
        old = single(kind, sid)
        return db.put_item(who, kind, data, sid, old["id"] if old else None)

    # --- stroke labelling (evaluation set) ------------------------------------------------

    @app.get("/api/sessions/{sid}/labels")
    def get_labels(sid: str, who: str = Depends(owner)) -> dict:
        s = owned(sid, who)
        got = single("label_set", sid)
        return got or {"labeller": "", "labeller_role": "", "expert": False, "camera_view": "unknown", "notes": "", "strokes": [],
                       "handedness": s["intake"].get("handedness", "right"), "backhand_style": s["intake"].get("backhand_style", "two_handed")}  # fmt: skip

    def _labels_doc(sid: str, s: dict, body: dict) -> dict:
        defaults = {"handedness": body.get("handedness") or s["intake"].get("handedness", "right"), "backhand_style": body.get("backhand_style") or s["intake"].get("backhand_style"),
                    "camera_view": body.get("camera_view") if body.get("camera_view") in CAMERA_VIEWS else "unknown"}  # fmt: skip
        doc = {"format": LABEL_FORMAT, "clip": sid, "labeller": _text(body.get("labeller"), 80), "labeller_role": _text(body.get("labeller_role"), 60), "expert": body.get("expert"),
               "method": "Marked in the app's labelling mode against the normalised video.", "notes": _text(body.get("notes"), 500), "strokes": body.get("strokes") or []}  # fmt: skip
        try:
            parsed = parse_labels(doc, defaults)
        except FormatError as e:
            raise HTTPException(400, str(e))
        if len(parsed.strokes) > 500:
            raise HTTPException(400, "Too many strokes.")
        dur = (s["meta"] or {}).get("duration_s") or 1e9
        if any(st.end > dur + 0.5 for st in parsed.strokes):
            raise HTTPException(400, "A labelled stroke ends after the end of the video.")
        return {**parsed.to_dict(), **defaults}

    @app.put("/api/sessions/{sid}/labels")
    def put_labels(sid: str, body: dict = Body(...), who: str = Depends(owner)) -> dict:
        s = owned(sid, who)
        return put_single("label_set", sid, who, _labels_doc(sid, s, body))

    @app.get("/api/sessions/{sid}/labels/export")
    def export_labels(sid: str, who: str = Depends(owner)):
        s = owned(sid, who)
        got = single("label_set", sid)
        if not got:
            raise HTTPException(404, "No labels saved for this session yet.")
        doc = {k: got[k] for k in ("format", "clip", "labeller", "labeller_role", "expert", "method", "labelled_at", "notes", "strokes")}
        doc["clip_defaults"] = {"handedness": got["handedness"], "backhand_style": got["backhand_style"], "camera_view": got["camera_view"], "title": s["title"], "duration_s": (s["meta"] or {}).get("duration_s")}
        return JSONResponse(doc, headers={"Content-Disposition": f'attachment; filename="labels-{sid[:8]}.json"', "Cache-Control": "private, no-store"})

    # --- joint annotation and reference angles ----------------------------------------------

    @app.get("/api/sessions/{sid}/annotations")
    def get_annotations(sid: str, who: str = Depends(owner)) -> dict:
        owned(sid, who)
        return single("annotation_set", sid) or {"annotator": "", "frames": []}

    @app.put("/api/sessions/{sid}/annotations")
    def put_annotations(sid: str, body: dict = Body(...), who: str = Depends(owner)) -> dict:
        s = owned(sid, who)
        frames = body.get("frames")
        if not isinstance(frames, list) or len(frames) > 400:
            raise HTTPException(400, "Invalid annotation list.")
        n = (s["meta"] or {}).get("n_frames") or 10**9
        clean = []
        for f in frames:
            try:
                idx = int(f["frame"])
                joints = {k: [min(max(float(v[0]), 0.0), 1.0), min(max(float(v[1]), 0.0), 1.0)] for k, v in (f.get("joints") or {}).items() if k in angles.JOINTS and v is not None}
            except (KeyError, TypeError, ValueError, IndexError):
                raise HTTPException(400, "Invalid annotation frame.")
            if not 0 <= idx < n:
                raise HTTPException(400, "Annotated frame is outside the video.")
            if joints:
                clean.append({"frame": idx, "joints": joints})
        annotator = _text(body.get("annotator"), 80)
        if clean and not annotator:
            raise HTTPException(400, "Say who made these annotations.")
        return put_single("annotation_set", sid, who, {"format": ANNOTATION_FORMAT, "annotator": annotator, "frames": sorted(clean, key=lambda f: f["frame"])})

    @app.get("/api/sessions/{sid}/annotations/export")
    def export_annotations(sid: str, who: str = Depends(owner)):
        owned(sid, who)
        got = single("annotation_set", sid)
        if not got:
            raise HTTPException(404, "No annotations saved for this session yet.")
        doc = {"format": got["format"], "clip": sid, "annotator": got["annotator"], "joints": list(angles.JOINTS), "frames": got["frames"]}
        return JSONResponse(doc, headers={"Content-Disposition": f'attachment; filename="joints-{sid[:8]}.json"', "Cache-Control": "private, no-store"})

    @app.get("/api/sessions/{sid}/annotations/report")
    def annotation_report(sid: str, who: str = Depends(owner)) -> dict:
        s = complete(sid, who)
        got = single("annotation_set", sid)
        if not got or not got["frames"]:
            return {"available": False, "reason": "No joints have been annotated on this video, so no accuracy figure can be given."}
        rep = angles.compare_annotations(track_of(sid), got["frames"])
        return {"available": True, "annotator": got["annotator"], "camera_view": s["report"]["quality"]["tracking"]["camera_view"], **rep}

    @app.post("/api/sessions/{sid}/mocap")
    async def mocap_report(sid: str, request: Request, offset_s: float = 0.0, who: str = Depends(owner)) -> dict:
        """Compare this session's angles with a motion-capture CSV. The CSV is used for this response only and is not stored."""
        s = complete(sid, who)
        raw = await request.body()
        if len(raw) > 5 * 1024 * 1024:
            raise HTTPException(413, "The CSV is larger than 5 MB.")
        try:
            ref = angles.parse_mocap_csv(raw.decode("utf-8-sig", errors="replace"))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"camera_view": s["report"]["quality"]["tracking"]["camera_view"], **angles.compare_mocap(track_of(sid), ref, offset_s)}

    @app.get("/api/sessions/{sid}/strokes/{stroke_id}/series")
    def stroke_series(sid: str, stroke_id: str, who: str = Depends(owner)) -> dict:
        s = complete(sid, who)
        st = next((x for x in s["report"]["strokes"] if x["id"] == stroke_id), None)
        if st is None:
            raise HTTPException(404, "Stroke not found")
        return {"stroke": {k: st[k] for k in ("id", "type", "start", "end", "peak")}, "timing": st.get("timing"), "phases": st.get("phases"), **angles.stroke_series(track_of(sid), st["start"], st["end"])}

    # --- reviewer mode ----------------------------------------------------------------------

    def check_code(code) -> None:
        if not settings.reviewer_code:
            raise HTTPException(403, "Reviewer sign-off is not enabled on this server (the operator has not set REVIEWER_CODE).")
        if not isinstance(code, str) or not hmac.compare_digest(code.encode(), settings.reviewer_code.encode()):
            raise HTTPException(403, "That reviewer code is not correct.")

    @app.get("/api/review/items")
    def review_items(_: str = Depends(owner)) -> dict:
        signoffs = db.signoffs()
        items = review.reviewables()
        return {"reviewer_mode": bool(settings.reviewer_code), "roles": sorted(review.ROLES),
                "items": [{"id": k, "kind": v["kind"], "title": v["title"], "needs": v["needs"], "text": v["text"], "hash": v["hash"], "status": review.status_of(k, signoffs, items)} for k, v in items.items()]}  # fmt: skip

    @app.post("/api/review/signoff", status_code=201)
    def signoff(body: dict = Body(...), _: str = Depends(owner)) -> dict:
        check_code(body.get("code"))
        items = review.reviewables()
        target, decision, role = body.get("target"), body.get("decision"), body.get("role")
        reviewer, cred = _text(body.get("reviewer"), 80), _text(body.get("credentials"), 120)
        if target not in items or decision not in ("approved", "changes_requested") or role not in review.ROLES:
            raise HTTPException(400, "Choose an item, a decision and your role.")
        if len(reviewer) < 3:
            raise HTTPException(400, "Enter your full name; it is shown next to the item you sign off.")
        if body.get("hash") != items[target]["hash"]:
            raise HTTPException(409, "This wording changed after you opened it. Reload and review the current text.")
        if decision == "changes_requested" and not _text(body.get("comment"), 2000):
            raise HTTPException(400, "Say what should change.")
        db.add_signoff(target, items[target]["hash"], decision, reviewer, role, cred, _text(body.get("comment"), 2000))
        return {"status": review.status_of(target, db.signoffs(), items)}

    @app.post("/api/review/signoff/{signoff_id}/withdraw")
    def withdraw(signoff_id: str, body: dict = Body(...), _: str = Depends(owner)) -> dict:
        check_code(body.get("code"))
        if not db.withdraw_signoff(signoff_id):
            raise HTTPException(404, "Sign-off not found")
        return {"ok": True}

    @app.get("/api/sessions/{sid}/reviews")
    def get_reviews(sid: str, who: str = Depends(owner)) -> list[dict]:
        owned(sid, who)
        return db.items("rule_review", session_id=sid)

    @app.put("/api/sessions/{sid}/reviews")
    def put_review(sid: str, body: dict = Body(...), who: str = Depends(owner)) -> dict:
        s = complete(sid, who)
        f = next((f for f in s["report"]["findings"] if f["id"] == body.get("finding_id")), None)
        if f is None:
            raise HTTPException(404, "Finding not found")
        stroke_id = body.get("stroke_id")
        if stroke_id is not None and stroke_id not in {o["stroke_id"] for o in f["occurrences"]}:
            raise HTTPException(400, "That stroke is not one of this finding's flagged clips.")
        reviewer, role = _text(body.get("reviewer"), 80), body.get("role")
        if len(reviewer) < 3 or role not in review.ROLES:
            raise HTTPException(400, "Enter your name and role before reviewing.")
        if body.get("verdict") not in ("correct", "incorrect", "unsure") or body.get("cue_ok") not in (True, False, None):
            raise HTTPException(400, "Choose whether the observation is correct.")
        occ = next((o for o in f["occurrences"] if o["stroke_id"] == stroke_id), None)
        data = {"finding_id": f["id"], "rule": f["rule"], "stroke_type": f["stroke_type"], "stroke_id": stroke_id, "verdict": body["verdict"], "cue_ok": body.get("cue_ok"),
                "comment": _text(body.get("comment"), 1000), "reviewer": reviewer, "role": role, "measured_value": occ and occ.get("value"), "observation": f["observation"],
                "camera_view": s["report"]["quality"]["tracking"]["camera_view"]}  # fmt: skip
        old = next((r for r in db.items("rule_review", session_id=sid) if r["finding_id"] == f["id"] and r["stroke_id"] == stroke_id and r["reviewer"].lower() == reviewer.lower()), None)
        return db.put_item(who, "rule_review", data, sid, old["id"] if old else None)

    @app.get("/api/review/export")
    def review_export(who: str = Depends(owner)):
        rows = db.items("rule_review", owner=who)
        return JSONResponse({"reviews": rows, "summary": review_summary(rows)}, headers={"Content-Disposition": 'attachment; filename="rule-reviews.json"', "Cache-Control": "private, no-store"})

    # --- re-acquisition confirmation --------------------------------------------------------

    @app.post("/api/sessions/{sid}/segments/{seg_id}")
    def decide_segment(sid: str, seg_id: str, body: dict = Body(...), who: str = Depends(owner)) -> dict:
        s = complete(sid, who)
        segs = {str(x["id"]): x for x in (s["report"].get("tracking_events") or {}).get("segments", [])}
        if seg_id not in segs or segs[seg_id]["status"] == "auto" or body.get("decision") not in ("confirm", "reject"):
            raise HTTPException(400, "Choose 'confirm' or 'reject' for a stretch that is waiting for your decision.")
        conf = {**(s.get("confirmations") or {}), seg_id: body["decision"] == "confirm"}
        report = rebuild_report(s, None, conf)
        db.update(sid, report=report, confirmations=conf)
        db.delete_items("llm_summary", sid)
        return public(db.get(sid, who), True)

    # --- comments and drawings --------------------------------------------------------------

    @app.get("/api/sessions/{sid}/comments")
    def comments(sid: str, who: str = Depends(owner)) -> list[dict]:
        owned(sid, who)
        return db.items("comment", session_id=sid)

    @app.post("/api/sessions/{sid}/comments", status_code=201)
    def add_comment(sid: str, body: dict = Body(...), who: str = Depends(owner)) -> dict:
        s = owned(sid, who)
        text = _text(body.get("body"), 1000)
        try:
            t = float(body.get("t"))
        except (TypeError, ValueError):
            raise HTTPException(400, "A comment needs a time in the video.")
        if not text or not 0 <= t <= ((s["meta"] or {}).get("duration_s") or 0) + 0.5:
            raise HTTPException(400, "A comment needs some text and a time inside the video.")
        if len(db.items("comment", session_id=sid)) >= 500:
            raise HTTPException(400, "This session has the maximum number of comments.")
        return db.put_item(who, "comment", {"t": round(t, 3), "body": text, "author": _text(body.get("author"), 60), "stroke_id": _text(body.get("stroke_id"), 16) or None}, sid)

    @app.delete("/api/sessions/{sid}/comments/{cid}", status_code=204)
    def delete_comment(sid: str, cid: str, who: str = Depends(owner)) -> Response:
        owned(sid, who)
        if not db.delete_item(cid, "comment", who):
            raise HTTPException(404, "Comment not found")
        return Response(status_code=204)

    @app.get("/api/sessions/{sid}/drawings")
    def drawings(sid: str, who: str = Depends(owner)) -> list[dict]:
        owned(sid, who)
        return db.items("drawing", session_id=sid)

    @app.post("/api/sessions/{sid}/drawings", status_code=201)
    def add_drawing(sid: str, body: dict = Body(...), who: str = Depends(owner)) -> dict:
        s = owned(sid, who)
        shapes = body.get("shapes")
        try:
            t = float(body.get("t"))
            clean = []
            for sh in shapes:
                lo, hi = SHAPES[sh["type"]]
                pts = [[round(min(max(float(x), 0.0), 1.0), 4), round(min(max(float(y), 0.0), 1.0), 4)] for x, y in sh["points"]]
                if not lo <= len(pts) <= hi:
                    raise ValueError
                clean.append({"type": sh["type"], "points": pts, "color": sh.get("color") if sh.get("color") in ("yellow", "red", "blue", "white") else "yellow"})
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, "Invalid drawing.")
        if not clean or len(clean) > 40 or not 0 <= t <= ((s["meta"] or {}).get("duration_s") or 0) + 0.5:
            raise HTTPException(400, "A drawing needs between 1 and 40 shapes and a time inside the video.")
        if len(db.items("drawing", session_id=sid)) >= 200:
            raise HTTPException(400, "This session has the maximum number of drawings.")
        return db.put_item(who, "drawing", {"t": round(t, 3), "shapes": clean, "note": _text(body.get("note"), 300), "author": _text(body.get("author"), 60)}, sid)

    @app.delete("/api/sessions/{sid}/drawings/{did}", status_code=204)
    def delete_drawing(sid: str, did: str, who: str = Depends(owner)) -> Response:
        owned(sid, who)
        if not db.delete_item(did, "drawing", who):
            raise HTTPException(404, "Drawing not found")
        return Response(status_code=204)

    # --- player profiles (browser-bound, no accounts) and progress ----------------------------

    def clean_profile(body: dict) -> dict:
        name = _text(body.get("name"), 60)
        if not name:
            raise HTTPException(400, "A profile needs a name.")
        pick = lambda k, allowed, d: body.get(k) if body.get(k) in allowed else d  # noqa: E731
        return {"name": name, "handedness": pick("handedness", {"right", "left"}, "right"), "backhand_style": pick("backhand_style", {"one_handed", "two_handed"}, "two_handed"),
                "skill_level": pick("skill_level", {"beginner", "intermediate", "advanced"}, "intermediate"), "notes": _text(body.get("notes"), 500)}  # fmt: skip

    @app.get("/api/profiles")
    def profiles(who: str = Depends(owner)) -> list[dict]:
        counts: dict[str, int] = {}
        for s in db.list(who):
            if s.get("profile_id"):
                counts[s["profile_id"]] = counts.get(s["profile_id"], 0) + 1
        return [{**p, "sessions": counts.get(p["id"], 0)} for p in db.items("profile", owner=who)]

    @app.post("/api/profiles", status_code=201)
    def add_profile(body: dict = Body(...), who: str = Depends(owner)) -> dict:
        if len(db.items("profile", owner=who)) >= 100:
            raise HTTPException(400, "This browser has the maximum number of profiles.")
        return db.put_item(who, "profile", clean_profile(body))

    @app.put("/api/profiles/{pid}")
    def edit_profile(pid: str, body: dict = Body(...), who: str = Depends(owner)) -> dict:
        try:
            return db.put_item(who, "profile", clean_profile(body), item_id=pid)
        except KeyError:
            raise HTTPException(404, "Profile not found")

    @app.delete("/api/profiles/{pid}", status_code=204)
    def delete_profile(pid: str, who: str = Depends(owner)) -> Response:
        """Deletes the profile only. Its sessions stay (unassigned) until they are deleted or expire."""
        if not db.delete_item(pid, "profile", who):
            raise HTTPException(404, "Profile not found")
        db.clear_profile(who, pid)
        return Response(status_code=204)

    @app.put("/api/sessions/{sid}/profile")
    def set_profile(sid: str, body: dict = Body(...), who: str = Depends(owner)) -> dict:
        owned(sid, who)
        pid = body.get("profile_id")
        if pid is not None and db.get_item(str(pid), "profile", who) is None:
            raise HTTPException(404, "Profile not found")
        db.update(sid, profile_id=pid)
        return {"ok": True}

    @app.get("/api/progress")
    def progress(profile_id: str | None = None, who: str = Depends(owner)) -> dict:
        """Per-session medians over time. Sessions filmed from different views are flagged, because the camera changes the numbers."""
        rows = []
        for s in sorted(db.list(who), key=lambda s: s["created_at"]):
            if s["status"] != "complete" or not s["report"] or (s["meta"] or {}).get("demo"):
                continue
            if profile_id is not None and s.get("profile_id") != profile_id:
                continue
            r = s["report"]
            rows.append({"id": s["id"], "title": s["title"], "created_at": s["created_at"], "camera_view": r["quality"]["tracking"]["camera_view"], "fps": r["meta"]["fps"],
                         "handedness": r.get("handedness"), "stroke_summary": r["stroke_summary"],
                         "observed": [{"rule": f["rule"], "title": f["title"], "stroke_type": f["stroke_type"], "fired": f["support"]["fired"], "assessed": f["support"]["assessed"]} for f in r["findings"] if f["status"] == "observed"]})  # fmt: skip
        views = sorted({r["camera_view"] for r in rows})
        caveat = ("All of these sessions were filmed from a similar view, but small differences in camera position, distance and frame rate still move the numbers. "
                  if len(views) <= 1 else
                  "These sessions were filmed from different camera views (" + "; ".join(views) + "). Angles and speeds change with the view, so compare only sessions marked with the same view. ")  # fmt: skip
        return {"sessions": rows, "views": views, "mixed_views": len(views) > 1,
                "caveat": caveat + "A change between sessions is a prompt to look at the video, not proof that technique changed. Demo sessions are left out."}  # fmt: skip

    # --- share links ------------------------------------------------------------------------

    @app.get("/api/sessions/{sid}/shares")
    def list_shares(sid: str, who: str = Depends(owner)) -> list[dict]:
        owned(sid, who)
        now = time.time()
        return [{**x, "active": x["revoked_at"] is None and x["expires_at"] > now} for x in db.shares(sid)]

    @app.post("/api/sessions/{sid}/shares", status_code=201)
    def create_share(sid: str, body: dict = Body(default={}), who: str = Depends(owner)) -> dict:
        complete(sid, who)
        try:
            days = float(body.get("days") or settings.share_ttl_days)
        except (TypeError, ValueError):
            raise HTTPException(400, "Invalid expiry.")
        if not 1 / 24 <= days <= settings.share_max_ttl_days:
            raise HTTPException(400, f"A link can last between one hour and {settings.share_max_ttl_days} days.")
        if sum(1 for x in db.shares(sid) if x["revoked_at"] is None and x["expires_at"] > time.time()) >= 10:
            raise HTTPException(400, "This session already has 10 active links. Revoke one first.")
        token = secrets.token_urlsafe(32)  # 256 bits; only its hash is stored, so the link cannot be recovered from the database
        rec = db.add_share(sid, who, hashlib.sha256(token.encode()).hexdigest(), time.time() + days * 86400)
        return {**rec, "active": True, "token": token, "path": f"/#/shared/{token}", "note": "Copy this link now. It is not stored and cannot be shown again."}

    @app.delete("/api/sessions/{sid}/shares/{share_id}", status_code=204)
    def revoke_share(sid: str, share_id: str, who: str = Depends(owner)) -> Response:
        owned(sid, who)
        if not db.revoke_share(share_id, sid):
            raise HTTPException(404, "Link not found or already revoked")
        return Response(status_code=204)

    def shared(token: str) -> dict:
        """Resolve a share token to its session. Unknown, expired and revoked links all give the same answer."""
        rec = db.share_by_hash(hashlib.sha256(token.encode()).hexdigest()) if 20 <= len(token) <= 100 else None
        if rec is None or rec["revoked_at"] is not None or rec["expires_at"] <= time.time():
            raise HTTPException(404, NOT_FOUND_SHARE)
        s = db.get(rec["session_id"])
        if s is None or s["status"] != "complete":
            raise HTTPException(404, NOT_FOUND_SHARE)
        s["_share"] = rec
        return s

    @app.get("/api/shared/{token}")
    def shared_session(token: str) -> dict:
        """Read-only view of one session. Self-reported health details (discomfort, injury history, notes) are never included."""
        s = shared(token)
        out = public(s, True)
        intake = out["intake"] or {}
        out["intake"] = {k: intake.get(k) for k in ("handedness", "skill_level", "backhand_style", "goals")} | {"playing_frequency": "", "workload_change": "", "discomfort": {"current": False, "regions": [], "notes": ""}, "injury_history": ""}
        rep = out["report"]
        if rep:
            rep["context_notes"] = []
            for f in rep["findings"]:
                f["user_context"] = None
        out |= {"read_only": True, "shared_until": s["_share"]["expires_at"], "candidates": [], "expires_at": None, "profile_id": None, "confirmations": {},
                "comments": [{k: c[k] for k in ("id", "t", "body", "author", "created_at")} for c in db.items("comment", session_id=s["id"])],
                "drawings": [{k: d[k] for k in ("id", "t", "shapes", "note", "author")} for d in db.items("drawing", session_id=s["id"])]}  # fmt: skip
        return out

    @app.get("/api/shared/{token}/video")
    def shared_video(token: str, request: Request):
        return store.response(shared(token)["id"], run.VIDEO, "video/mp4", request)

    @app.get("/api/shared/{token}/overlay")
    def shared_overlay(token: str):
        return store.response(shared(token)["id"], run.OVERLAY, "application/json")

    # --- PDF export -------------------------------------------------------------------------

    @app.get("/api/sessions/{sid}/report.pdf")
    def report_pdf(sid: str, who: str = Depends(owner)):
        """Built on request and streamed; no copy is kept on the server."""
        from .pdf import build_pdf

        s = complete(sid, who)
        rep = review.apply_review_status(s["report"], db.signoffs())
        track = track_of(sid) if store.exists(sid, run.TRACK) else None
        data = build_pdf(title=s["title"], created_at=s["created_at"], intake=s["intake"], report=rep, track=track, comments=db.items("comment", session_id=sid))
        return Response(data, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="swingwise-report-{sid[:8]}.pdf"', "Cache-Control": "private, no-store"})

    # --- live-capture framing guidance ----------------------------------------------------

    last_check: dict[str, float] = {}
    framing_detector: list = []

    @app.post("/api/framing")
    async def framing(request: Request, who: str = Depends(owner)) -> dict:
        """Check one camera snapshot (JPEG body) before recording. The image is examined in memory and not stored."""
        import cv2
        import numpy as np

        from .pipeline.framing import assess

        raw = await request.body()
        if not raw or len(raw) > 600_000:
            raise HTTPException(413, "Send one JPEG snapshot of at most 600 kB.")
        now = time.time()
        if now - last_check.get(who, 0) < 0.5:
            raise HTTPException(429, "Too many framing checks; wait a moment.")
        last_check[who] = now
        if len(last_check) > 5000:
            last_check.clear()
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None or min(frame.shape[:2]) < 60:
            raise HTTPException(400, "That was not a readable image.")
        if not framing_detector:
            framing_detector.append(worker.detector_factory())
        return assess(frame, framing_detector[0].detect_image(frame))

    # --- optional LLM summary ---------------------------------------------------------------

    @app.get("/api/sessions/{sid}/summary")
    def get_summary(sid: str, who: str = Depends(owner)) -> dict:
        owned(sid, who)
        got = single("llm_summary", sid)
        return {"enabled": bool(settings.llm_api_key), "summary": got}

    @app.post("/api/sessions/{sid}/summary")
    def make_summary(sid: str, who: str = Depends(owner)) -> dict:
        from . import llm

        s = complete(sid, who)
        if not settings.llm_api_key:
            raise HTTPException(404, "Plain-language summaries are not enabled on this server.")
        try:
            res = llm.summarise(s["report"], api_key=settings.llm_api_key, model=settings.llm_model)
        except llm.SummaryRejected as e:
            raise HTTPException(422, f"The generated summary was discarded because it did not match the findings ({e}). The structured report above is unaffected.")
        except llm.SummaryUnavailable as e:
            raise HTTPException(502, f"The summary service could not be reached ({e}). The structured report above is unaffected.")
        return {"enabled": True, "summary": put_single("llm_summary", sid, who, res)}

    @app.delete("/api/sessions/{sid}/summary", status_code=204)
    def delete_summary(sid: str, who: str = Depends(owner)) -> Response:
        owned(sid, who)
        db.delete_items("llm_summary", sid)
        return Response(status_code=204)


def review_summary(rows: list[dict]) -> dict:
    """Per rule: how many flagged clips reviewers judged correct, and how often two reviewers of the same clip agreed."""
    out: dict[str, dict] = {}
    for rule in sorted({r["rule"] for r in rows}):
        mine = [r for r in rows if r["rule"] == rule and r["stroke_id"] is not None]
        by_clip: dict[tuple, list[dict]] = {}
        for r in mine:
            by_clip.setdefault((r["session_id"], r["finding_id"], r["stroke_id"]), []).append(r)
        pairs = agree = 0
        for group in by_clip.values():
            v = [g["verdict"] for g in group if g["verdict"] != "unsure"]
            for i in range(len(v)):
                for j in range(i + 1, len(v)):
                    pairs, agree = pairs + 1, agree + (v[i] == v[j])
        cue = [r["cue_ok"] for r in rows if r["rule"] == rule and r["cue_ok"] is not None]
        decided = [r for r in mine if r["verdict"] != "unsure"]
        out[rule] = {"clips_reviewed": len(by_clip), "reviews": len(mine), "reviewers": sorted({r["reviewer"] for r in rows if r["rule"] == rule}),
                     "judged_correct": sum(r["verdict"] == "correct" for r in decided), "judged_incorrect": sum(r["verdict"] == "incorrect" for r in decided), "unsure": len(mine) - len(decided),
                     "share_correct": round(sum(r["verdict"] == "correct" for r in decided) / len(decided), 2) if decided else None,
                     "reviewer_pairs": pairs, "pair_agreement": round(agree / pairs, 2) if pairs else None,
                     "cue_judged_appropriate": sum(cue), "cue_judged_inappropriate": len(cue) - sum(cue),
                     "comments": [r["comment"] for r in rows if r["rule"] == rule and r["comment"]]}  # fmt: skip
    return out
