"""Human review of the fixed wording and evidence.

Nothing in this file marks anything as reviewed. A review status changes only when a named
reviewer signs off through the reviewer mode (which needs the operator-issued REVIEWER_CODE).
A sign-off records a hash of the exact text that was reviewed; if the text is edited later,
the sign-off no longer applies and the status falls back to "changed since review".
"""
from __future__ import annotations

import hashlib
import json
import time

from .pipeline import drills
from .pipeline.evidence import EVIDENCE, REVIEW_STATUS
from .pipeline.findings import PAIN_GUIDANCE, RULES

CLINICAL_ROLES = {"physiotherapist", "sports_physician", "other_clinician"}
COACH_ROLES = {"coach"}
ROLES = CLINICAL_ROLES | COACH_ROLES


def reviewables() -> dict[str, dict]:
    """Everything a reviewer can sign off: id -> {kind, title, text, needs}. ``needs`` is who must approve it."""
    out: dict[str, dict] = {}
    for k, e in EVIDENCE.items():
        out[f"evidence:{k}"] = {"kind": "evidence", "title": e["citation"], "needs": "clinician",
                                "text": {"citation": e["citation"], "doi": e["doi"], "population": e["population"], "summary": e["summary"], "limits": e["limits"]}}  # fmt: skip
    for r in RULES:
        text = {"title": r["title"], "cue": r["cue"], "cue_beginner": r["cue_beginner"], "limits": r["limits"]}
        if r["category"] == "health_related":
            text |= {"interpretation": r["interpretation"], "evidence_strength": r["evidence"]["strength"], "evidence_refs": list(r["evidence"]["refs"])}
        out[f"rule:{r['id']}"] = {"kind": "rule", "title": r["title"], "needs": "clinician" if r["category"] == "health_related" else "coach", "text": text}
    out["text:pain_guidance"] = {"kind": "text", "title": "Pain guidance shown when a player reports discomfort", "needs": "clinician", "text": {"body": PAIN_GUIDANCE}}
    for d in drills.DRILLS.values():
        out[f"drill:{d['id']}"] = {"kind": "drill", "title": d["name"], "needs": "coach", "text": {"name": d["name"], "steps": d["steps"], "focus": d["focus"], "caution": d.get("caution", "")}}
    for v in out.values():
        v["hash"] = hashlib.sha256(json.dumps(v["text"], sort_keys=True).encode()).hexdigest()[:16]
    return out


def status_of(target: str, signoffs: list[dict], items: dict[str, dict] | None = None) -> dict:
    """Review status of one reviewable, from the sign-offs on record. Latest decision per reviewer counts."""
    items = items or reviewables()
    item = items[target]
    default = REVIEW_STATUS if item["needs"] == "clinician" else "Not yet coach-reviewed"
    mine = [s for s in signoffs if s["target"] == target and not s.get("withdrawn_at")]
    if not mine:
        return {"state": "unreviewed", "label": default, "signoffs": []}
    latest: dict[str, dict] = {}
    for s in sorted(mine, key=lambda s: s["created_at"]):
        latest[s["reviewer"].strip().lower()] = s
    current = [s for s in latest.values() if s["content_hash"] == item["hash"]]
    qualified = CLINICAL_ROLES if item["needs"] == "clinician" else ROLES
    who = lambda s: f"{s['reviewer']} ({s['role'].replace('_', ' ')}{', ' + s['credentials'] if s['credentials'] else ''}) on {time.strftime('%Y-%m-%d', time.gmtime(s['created_at']))}"  # noqa: E731
    public = [{"reviewer": s["reviewer"], "role": s["role"], "credentials": s["credentials"], "decision": s["decision"], "comment": s["comment"], "date": time.strftime("%Y-%m-%d", time.gmtime(s["created_at"])),
               "current": s["content_hash"] == item["hash"], "id": s["id"]} for s in latest.values()]  # fmt: skip
    if not current:
        return {"state": "stale", "label": "Wording changed since it was last reviewed; review needed again", "signoffs": public}
    changes = [s for s in current if s["decision"] == "changes_requested"]
    if changes:
        return {"state": "changes_requested", "label": "Changes requested by " + "; ".join(who(s) for s in changes), "signoffs": public}
    ok = [s for s in current if s["decision"] == "approved" and s["role"] in qualified]
    if ok:
        return {"state": "approved", "label": "Reviewed and approved by " + "; ".join(who(s) for s in ok), "signoffs": public}
    return {"state": "unreviewed", "label": default + " (reviewed by " + "; ".join(who(s) for s in current) + ", who is not in the role this item needs)", "signoffs": public}


def apply_review_status(report: dict | None, signoffs: list[dict]) -> dict | None:
    """Fill in the current review status wherever a stored report shows evidence or reviewable wording."""
    if not report:
        return report
    items = reviewables()
    ev = {k: status_of(f"evidence:{k}", signoffs, items)["label"] for k in EVIDENCE}

    def fix(refs: list[dict]) -> None:
        for r in refs or []:
            if r.get("id") in ev:
                r["review_status"] = ev[r["id"]]

    for f in report.get("findings", []):
        if f.get("evidence"):
            fix(f["evidence"]["refs"])
        if f"rule:{f.get('rule')}" in items:
            f["wording_review"] = status_of(f"rule:{f['rule']}", signoffs, items)["label"]
        for d in f.get("drills") or []:
            if f"drill:{d['id']}" in items:
                d["review_status"] = status_of(f"drill:{d['id']}", signoffs, items)["label"]
    for n in report.get("context_notes", []):
        fix(n.get("refs"))
        if n.get("kind") == "pain":
            n["review_note"] = status_of("text:pain_guidance", signoffs, items)["label"]
    states = [status_of(t, signoffs, items)["state"] for t in items if items[t]["needs"] == "clinician"]
    if states and all(s == "approved" for s in states):
        report["validation_status"] = "Stroke detection, measurements and thresholds are unvalidated heuristics. Evidence summaries and health-related wording have been signed off by a clinician (see each reference)."
    return report
