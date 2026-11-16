# Evaluation set

Everything that judges the analysis lives here. Scores from this set are a development aid.
**Every label in it today was made by a non-expert (the developer's AI assistant, from film
strips), so nothing here is validated accuracy.** Coach labels replace them the same way.

```
manifest.json          one entry per clip: source, licence, fixed split, handedness, backhand style, camera view
clips/<id>/source.*    footage (not in git; fetch from the page in the manifest)
labels/<id>.json       stroke labels for a clip
annotations/<id>.json  hand-clicked joint positions (optional; none yet)
mocap/<id>.csv         motion-capture reference angles (optional; none yet)
runs/*.json            stored scoring runs
cache/                 normalised clips and cached pose output; safe to delete (score --clear-cache)
```

## Commands (run in `backend/`)

```bash
.venv/bin/python -m app.evaltools.score --name my-change          # run the real pipeline, print and store scores
.venv/bin/python -m app.evaltools.score --baseline --name base    # first-build behaviour (all later stages off)
.venv/bin/python -m app.evaltools.score --compare base my-change  # before / after on both splits
.venv/bin/python -m app.evaltools.score --list                    # stored runs
.venv/bin/python -m app.evaltools.add --help                      # add a clip and its label file
.venv/bin/python -m app.evaltools.angles                          # pose / angle error where reference data exists
.venv/bin/python -m app.evaltools.reviews                         # reviewer feedback and sign-off status
```

Any option in `app/pipeline/options.py` can be overridden for a run, e.g. `--ball 0`, `--pose-mode cropped`.

## What is scored

A detection matches a labelled stroke when its timestamp is within 0.5 s of the labelled
contact (closest pairs first, one-to-one). Reported per split and by stroke type, camera view
and handedness: detection recall and precision, label confusion (including "left unlabelled"
and "missed"), share correct end to end, and timing error against contact. Strokes labelled
`other` (volley, slice, smash) are neither misses nor false detections. A clip without a label
file is run and listed but not scored.

## Splits

`tune` may be looked at while changing the analysis. `held_out` may not: it is scored, and
that is all. A clip's split is fixed when it is added, before seeing how the app does on it.

## Label file (`labels/<id>.json`)

```json
{"format": "baseline-labels/1", "clip": "<id>", "labeller": "Name", "labeller_role": "LTA level 3 coach",
 "expert": true, "method": "how the times were found", "labelled_at": "2026-11-20", "notes": "",
 "strokes": [{"type": "forehand|backhand|serve|other", "start": 1.07, "contact": 1.87, "end": 2.57,
              "contact_kind": "contact|peak", "handedness": "right|left", "backhand_style": "one_handed|two_handed|null",
              "camera_view": "side|behind|front|oblique|far|unknown", "notes": ""}]}
```

Times are seconds in the clip. `contact_kind` is `contact` when the ball is seen on the
strings, `peak` for a shadow swing or when contact is not visible. Per-stroke handedness,
backhand style and camera view default to the clip's values in the manifest. `expert` must be
true only for a qualified coach.

**How a coach supplies labels:** upload the clip in the app, open the session's **Label** tab,
mark start / contact / end for every stroke (found by the app or not), press *Export label
file*, then run `app.evaltools.add` with the same video and that file.

## Joint annotations and motion capture

*Annotate* tab → click joint centres on chosen frames → *Export annotations* → save as
`annotations/<id>.json` (`{"format": "baseline-joints/1", "annotator": "...", "frames": [{"frame": 120, "joints": {"left_knee": [x, y]}}]}`,
coordinates as fractions of frame width and height).

Motion capture: `mocap/<id>.csv` with a header row, a `time_s` column (seconds from the start
of the clip) and any of `knee_flexion_left`, `knee_flexion_right` (0 = straight),
`elbow_angle_left`, `elbow_angle_right` (180 = straight), `trunk_lean`, all in degrees. Lines
starting with `#` are ignored. Use `--offset <id>=<seconds>` if the clocks differ.

## What is in the set now

13 clips from 7 Wikimedia Commons files (sources and licences are in `manifest.json`):

| Split | Clips | Labelled strokes | What |
|---|---|---|---|
| tune | 6 labelled | 25 (14 forehand, 10 backhand, 1 serve) | player 1 shadow swings (2 clips), player 1 two-handed backhands with ball, Sharapova match clip, left-hander Rola (2 clips) |
| held_out | 3 labelled | 14 (5 forehand, 5 one-handed backhand, 4 serve) | player 2 forehands and backhands with ball, player 1 serves |
| tune | 3 unlabelled | – | whole instructional files with hard cuts, for cut and re-acquisition checks |
| held_out | 1 unlabelled | – | doubles from the stands (the pipeline correctly refuses it) |

About 154 s of labelled footage, 4 players, 39 strokes, all right-handed except Rola, no
expert labels. Player 1 appears in both splits (different footage). That is far too small to
generalise from; see docs/VALIDATION.md for what the numbers can and cannot support.

The Ogg source files hold a title card for several seconds with a jump in their timestamps; clip trims in
the manifest are in presentation time.
