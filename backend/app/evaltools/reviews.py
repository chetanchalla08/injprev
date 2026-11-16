"""Summarise reviewer feedback held in the app database (operator-side; reads every browser's reviews).

    .venv/bin/python -m app.evaltools.reviews

Per rule: clips reviewed, share judged correct, agreement between reviewers of the same clip,
cue verdicts and wording comments; then the sign-off status of every evidence entry and
fixed text. Reviews of a deleted session are gone with it, so export before deleting.
"""
from __future__ import annotations

import json

from .. import review
from ..api_extra import review_summary
from ..config import load_settings
from ..crypto import Vault
from ..db import DB


def main() -> int:
    s = load_settings()
    db = DB(s.db_path, Vault.from_env(s.data_key))
    rows = db.items("rule_review")
    print(f"{len(rows)} clip/rule reviews on record.")
    print(json.dumps(review_summary(rows), indent=1))
    signoffs, items = db.signoffs(), review.reviewables()
    print("\nSign-off status")
    for k in items:
        st = review.status_of(k, signoffs, items)
        print(f"  [{st['state']:<17}] {k:<34} {st['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
