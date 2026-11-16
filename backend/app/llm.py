"""Optional plain-language rewording of the structured findings (off unless ENABLE_LLM_SUMMARY=1 and ANTHROPIC_API_KEY are set).

The model's only job is to reword. It is sent a JSON document built from the findings and
nothing else: no video, no pose data, no intake or health details, no citations. Its output
is then checked against that same JSON before anyone sees it, and discarded if it contains
  * any number that is not in the source,
  * anything that looks like a reference (DOI, URL, "et al.", journal-style wording), or
  * diagnostic or risk wording the report itself is not allowed to use.
The check is deliberately blunt: a false rejection costs a summary; a false acceptance
would put an invented measurement in front of a player. The API key stays on the server.
"""
from __future__ import annotations

import hashlib
import json
import re
import time

MAX_CHARS = 2500
SYSTEM = (
    "You reword a tennis technique report for the player. You are given a JSON document of findings produced by fixed rules. "
    "Rewrite it as two to four short plain-language paragraphs.\n"
    "Rules: use only what is in the JSON. Do not add observations, causes, measurements, advice, drills or encouragement that are not there. "
    "Write every number as digits exactly as it appears in the JSON, and use no other numbers. Do not cite or mention studies, sources or references. "
    "Do not diagnose, mention injuries or conditions, or describe risk. Keep each stated limitation's meaning; if something could not be assessed, say so. "
    "Plain text only, no headings, lists or markdown."
)
NUMBER_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
                "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50", "half": "0.5", "twice": "2", "double": "2", "dozen": "12"}  # fmt: skip
REFERENCE_MARKERS = ("doi", "http", "www.", "et al", "journal", "according to", "research shows", "studies show", "study shows", "a study", "evidence shows", "published", "researchers")
BANNED = ("risk score", "% risk", "% chance", "you have a", "you may have", "diagnos", "tendin", "tear", "will cause", "safe range", "injury risk", "at risk", "likely to be injured", "prevent injur")


class SummaryRejected(Exception):
    """The generated text failed the check against the findings and was discarded."""


class SummaryUnavailable(Exception):
    """The model could not be reached or declined."""


def source_document(report: dict) -> dict:
    """Exactly what the model is shown. Evidence references and self-reported health details are left out on purpose."""
    return {
        "headline": report["summary"]["headline"],
        "recording_limitations": [i["message"] for i in report["quality"]["issues"]],
        "stroke_timing": (report.get("ball") or {}).get("note"),
        "findings": [
            {"title": f["title"], "stroke": f["stroke_type"], "status": f["status"], "observation": f["observation"], "possible_interpretation": f.get("interpretation"),
             "suggested_cue": f["action"].get("cue"), "limitations": f.get("limitations") or [], "how_clearly_seen": (f.get("confidence") or {}).get("level")}
            for f in report["findings"]
        ],
        "not_assessed": [n["topic"] for n in report.get("not_assessed", [])],
        "disclaimer": report["disclaimer"],
    }  # fmt: skip


def _numbers(text: str) -> set[str]:
    out = set()
    for m in re.findall(r"\d+(?:[.,]\d+)?", text):
        v = m.replace(",", ".")
        out.add(v.rstrip("0").rstrip(".") if "." in v else v.lstrip("0") or "0")
    return out


def check(text: str, source: dict) -> list[str]:
    """Reasons to reject ``text`` as a rewording of ``source``; an empty list means it passed."""
    problems = []
    src = json.dumps(source, ensure_ascii=False)
    allowed = _numbers(src)
    low = text.lower()
    extra = sorted(_numbers(text) - allowed)
    if extra:
        problems.append("it contains numbers that are not in the findings: " + ", ".join(extra))
    words = sorted({w for w in re.findall(r"[a-z]+", low) if w in NUMBER_WORDS and NUMBER_WORDS[w] not in allowed and w not in src.lower()})
    if words:
        problems.append("it contains quantities that are not in the findings: " + ", ".join(words))
    refs = [m for m in REFERENCE_MARKERS if m in low and m not in src.lower()]
    if refs or re.search(r"\([A-Z][a-z]+,? (?:and|&)? ?[A-Za-z ]*\d{4}\)|\[\d+\]", text):
        problems.append("it appears to cite a reference" + (f" ({', '.join(refs)})" if refs else ""))
    bad = [b for b in BANNED if b in low and b not in src.lower()]
    if bad:
        problems.append("it uses diagnostic or risk wording: " + ", ".join(bad))
    if not text.strip() or len(text) > MAX_CHARS:
        problems.append("it is empty or too long")
    return problems


def summarise(report: dict, *, api_key: str, model: str, client=None) -> dict:
    source = source_document(report)
    payload = json.dumps(source, ensure_ascii=False, sort_keys=True)
    try:
        import anthropic

        client = client or anthropic.Anthropic(api_key=api_key, timeout=60.0)
        try:
            resp = client.beta.messages.create(
                model=model, max_tokens=2000, system=SYSTEM, output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",  # if the model declines, the API retries on a fallback model in the same call
                messages=[{"role": "user", "content": payload}],
            )  # fmt: skip
        except anthropic.AuthenticationError as e:
            raise SummaryUnavailable("the server's API key was not accepted") from e
        except anthropic.RateLimitError as e:
            raise SummaryUnavailable("rate limited; try again later") from e
        except anthropic.APIStatusError as e:
            raise SummaryUnavailable(f"API error {e.status_code}") from e
        except anthropic.APIConnectionError as e:
            raise SummaryUnavailable("network error") from e
    except ImportError as e:
        raise SummaryUnavailable("the anthropic package is not installed") from e
    if getattr(resp, "stop_reason", None) == "refusal":
        raise SummaryUnavailable("the model declined to produce a summary")
    if getattr(resp, "stop_reason", None) == "max_tokens":
        raise SummaryRejected("it was cut off before finishing")
    text = "\n\n".join(b.text.strip() for b in resp.content if getattr(b, "type", None) == "text").strip()
    problems = check(text, source)
    if problems:
        raise SummaryRejected("; ".join(problems))
    return {"text": text, "model": getattr(resp, "model", model), "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source_hash": hashlib.sha256(payload.encode()).hexdigest()[:16],
            "notice": "Reworded by an AI language model from the findings below and checked against them automatically. It adds nothing new; where it differs from the findings, the findings are right."}  # fmt: skip
