# Swingwise — stroke technique review (tennis now; pickleball and golf coming soon)

Upload or record a tennis clip, get forehands, backhands and serves marked on the video with
measured observations, coaching cues and drills, and carefully limited notes on movement patterns.

**This is a technique-analysis and education tool. It does not diagnose, does not predict
injury, and has not been clinically validated.** See [docs/VALIDATION.md](docs/VALIDATION.md).

## Run it

Requires Python 3.11+ and Node 20+. No system ffmpeg needed (a binary ships with `imageio-ffmpeg`).

```bash
cd backend
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_models.py      # MediaPipe pose weights (~40 MB) from Google's model bucket
cp .env.example .env                              # optional; defaults work

cd ../frontend && npm install && npm run build    # builds the UI into frontend/dist

cd ../backend && .venv/bin/uvicorn app.main:app_factory --factory --port 8000
```

Open http://localhost:8000. For frontend development run `npm run dev` in `frontend/`
(port 5173, proxies `/api` to 8000). Tests: `cd backend && .venv/bin/python -m pytest tests`.
Scoring against the evaluation set: `.venv/bin/python -m app.evaltools.score --name <run>`
(see [evalset/README.md](evalset/README.md)).

Set `ENABLE_DEMO=1` to get a "Create a demo session" link that makes a clearly labelled
synthetic stick-figure session for trying the interface. It never touches real uploads.

## Layout

```
backend/app/pipeline/   video.py      validate, normalise, quality checks
                        pose.py       MediaPipe wrapper, candidates, tracking, re-acquisition, optional cropped pass
                        visual.py     scene cuts, per-person appearance, ball candidates (one pass over pixels)
                        ball.py       ball tracks and contact;  court.py  court lines, mapping, movement
                        signals.py    pose track container;     segment.py  strokes, ball-based timing, phases
                        measure.py    per-stroke measurements, each tagged with its method
                        findings.py   rules -> findings, confidence, report;  drills.py  fixed drill text
                        evidence.py   the only source of health-related references
                        framing.py    live-capture checks;      options.py  switch for every optional stage
                        run.py        prepare / analyze / rebuild
backend/app/            main.py + api_extra.py (API), worker.py (queue, retries, retention), db.py (SQLite),
                        crypto.py + storage.py (encryption at rest), review.py (sign-off), angles.py (angle series,
                        reference comparison), pdf.py, llm.py (optional, checked), demo.py (synthetic figure)
backend/app/evaltools/  format.py, score.py, add.py, angles.py, reviews.py
backend/tests/          127 tests
frontend/src/           Home, Session (Player / Coach / Clinician / Label / Annotate / Review tabs), Compare,
                        Duo (side by side), Progress, Review (sign-off), Capture, Privacy
evalset/                manifest, labels, stored runs (footage itself is not in git)
docs/                   ANALYSIS_LOGIC.md, VALIDATION.md
```

Stack: FastAPI + SQLite + one in-process worker thread; React + TypeScript + Vite. Pose model:
MediaPipe Pose Landmarker on CPU. Findings are rule-based fixed text filled with measured
values; no language model is involved unless the optional summary is switched on, and that
only rewords.

## What works

- Upload or in-browser recording → validation → quality check → player selection → pose → strokes → report.
- Scene cuts are detected; the player is followed across a cut or a gap only on a clear clothing
  match, and doubtful stretches wait for the user's confirmation before being analysed.
- Ball tracking: strokes are timed at ball contact when it is found, and say which timestamp was used.
- Manual add / relabel / remove of strokes; everything recomputes from saved pose data.
- Player, coach and clinician views; findings keep observation, interpretation, evidence and
  action apart; "unable to assess" with the reason; 2D vs single-camera 3D labelled everywhere.
- Players: side-by-side replay of two strokes aligned at their timestamps; drills per cue
  (marked "Not yet coach-reviewed"); progress across sessions with camera-view caveats.
- Coaches: player profiles (per browser), comments pinned to timestamps, line / angle / freehand
  drawings on frames, expiring and revocable read-only share links.
- Clinicians: structured PDF, joint-angle-over-time plots with method, visibility and gaps.
- Evaluation: labelling mode, joint annotation, motion-capture comparison, scoring harness with
  a fixed tune / held-out split and stored runs, reviewer mode with sign-off.
- Background jobs with retry and crash recovery; per-browser access; retention and deletion that
  cover everything attached to a session.

## Encryption at rest

Set `DATA_KEY` (generate one with `.venv/bin/python -m app.crypto`). Videos, pose data,
overlays, analysis caches, and the database columns holding titles, intake (including
discomfort and injury history), reports, comments, drawings, labels and annotations are then
stored with AES-256-GCM; files are encrypted in chunks so video seeking still works.

- **Key handling.** The key lives only in the server's environment. Keep a copy outside the data
  directory and its backups: without it the data is unreadable and cannot be recovered. With
  `REQUIRE_ENCRYPTION=1` the app refuses to start without a key. Starting with no key or the
  wrong key gives a clear error on every read; it never returns scrambled data. There is no key
  rotation tool yet: changing the key makes existing data unreadable.
- **Protects against** someone obtaining the data directory, the database file or a backup
  without the key.
- **Does not protect against** anyone who can read the server's environment or memory, a
  compromised application, or a host that holds both data and key. During processing a plaintext
  working copy of the video exists in `DATA_DIR/tmp` (removed afterwards, wiped at start-up).
  Data stored before a key was set stays unencrypted. Share-link records, reviewer sign-offs,
  job status and timestamps are not encrypted. It is not end-to-end encryption.

## Optional LLM summary

Off by default (`ENABLE_LLM_SUMMARY=1` and `ANTHROPIC_API_KEY` turn it on; the key stays on the
server). The model is sent a JSON of the findings and nothing else and is told only to reword.
Its text is discarded unless every number in it appears in that JSON and it contains no
reference-like or diagnostic wording. Turning it on means the written findings leave the server.

## What is not validated or not built

- **Accuracy is known only on a tiny, non-expert-labelled set** (39 strokes, 4 players). On
  held-out footage every stroke was found; 86% got the right label, but two of five one-handed
  backhands are still labelled as serves. See docs/VALIDATION.md for every number.
- Ball contact was found for 2 of 14 held-out strokes; contact point is an unvalidated
  observation. Court position needs five court lines in view and was found in none of the
  evaluation clips; footwork observations are unvalidated. Wrist/grip and segment sequencing are
  still not assessed.
- No joint-angle accuracy figure exists (no reference data yet). No coach has reviewed flagged
  clips. No clinician has reviewed evidence or wording; drills are not coach-reviewed.
- Racket tracking is not delivered: a pretrained detector was evaluated but there is nothing to score it against.
- Live capture has not been tried on a real camera in this environment.
- No accounts (cookie-bound access only), no GPU or separate worker machine, no key rotation.
  No regulatory compliance is claimed.

## Next priorities

1. Coach-labelled, consented footage across views, levels and handedness; then finish one-handed
   backhand vs serve (partly fixed; two of five held-out cases remain) and judge it on new held-out footage.
2. Joint annotation and motion-capture reference so angles can be given an error.
3. Coach review of flagged clips; clinician sign-off of evidence and wording.
4. Full-court footage to test court mapping before any footwork claim.
5. Accounts and key rotation before any hosted deployment.
