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
contact (closest pairs first, one-to-one). Reported per split and by stroke type, camera view,
handedness, backhand style and player: detection recall and precision, label confusion (including
"left unlabelled" and "missed"), share correct end to end, and timing error against contact. Strokes
labelled `other` (volley, slice, smash) are neither misses nor false detections, but when they carry a
`subtype` the report lists what the app took each kind for (e.g. "smash taken as serve 3, missed 1"),
which is how serve look-alikes are measured. A clip without a label file is run and listed but not scored.

## Splits

`tune` may be looked at while changing the analysis. `held_out` may not: it is scored, and
that is all. A clip's split is fixed when it is added, before seeing how the app does on it.

Each clip names its `player` (a stable id such as `p12`; `null` when several people are shown).
A held-out result only says something about new people if those people are not in tune, so
`app.evaltools.add --player` warns when a player would end up in both splits. Player `p1` is in
both today (different footage); new held-out clips should use new players.

## Label file (`labels/<id>.json`)

```json
{"format": "baseline-labels/1", "clip": "<id>", "labeller": "Name", "labeller_role": "LTA level 3 coach",
 "expert": true, "method": "how the times were found", "labelled_at": "2026-11-20", "notes": "",
 "strokes": [{"type": "forehand|backhand|serve|other", "start": 1.07, "contact": 1.87, "end": 2.57,
              "contact_kind": "contact|peak", "handedness": "right|left", "backhand_style": "one_handed|two_handed|null",
              "camera_view": "side|behind|front|oblique|far|unknown", "subtype": "slice|smash|volley|null", "notes": ""}]}
```

Times are seconds in the clip. `contact_kind` is `contact` when the ball is seen on the
strings, `peak` for a shadow swing or when contact is not visible. Per-stroke handedness,
backhand style and camera view default to the clip's values in the manifest. `subtype` is optional
and allowed only on `other` strokes. `expert` must be true only for a qualified coach.

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

59 labelled clips (plus 4 unlabelled): 58 cut from 22 freely licensed films and one phone recording provided by the project owner (sources and licences are in `manifest.json`;
what was found but left out, and why, is in [CANDIDATES.md](CANDIDATES.md)):

| Split | Clips | Labelled strokes | Players |
|---|---|---|---|
| tune | 46 labelled | 133 (53 forehand, 26 backhand, 23 serve, 15 smash, 10 volley, 6 slice) | p1, p3, Sharapova, Rola, Tilden, p4, p5 (Tilden's pupils), p6 (EVD demonstrator), p12 (club player, modern phone footage) |
| held_out | 13 labelled | 45 (12 forehand, 13 backhand, 6 serve, 4 smash, 6 volley, 4 slice) | p1 (other footage), p2, p8, p9 (1938 newsreel), p10 (junior), Gibson, p11 |
| tune | 3 unlabelled | – | whole instructional files with hard cuts, for cut and re-acquisition checks |
| held_out | 1 unlabelled | – | doubles from the stands (the pipeline correctly refuses it) |

178 strokes from 15 players, no expert labels. Much of the new footage is 1930s-1950s film (black and
white, 24-30 fps, frequent cuts) or 320x240 coaching video, so it stresses the pose model more than modern
phone footage would. Player p1 appears in both splits (different footage); every other held-out player is
new. Held-out was scored once on 2026-11-21 (run `smash_fix_heldout`); see `docs/VALIDATION.md`.

The Ogg source files hold a title card for several seconds with a jump in their timestamps; clip trims in
the manifest are in presentation time.

## Coverage targets

The aim is a set that looks like the people who will use the app, not more of the same few
players. Counts are labelled strokes unless stated; "now" is the set as of 2026-11-22.

| Dimension | Target | Now |
|---|---|---|
| Labelled strokes | 300+ | 178 |
| Distinct players | 20+ | 15 |
| Held-out players never seen in tune | 8+ | 6 (p2, p8, p9, p10, p11, Gibson) |
| Playing level (recorded in clip notes as `level: beginner/club/advanced/pro`) | 5+ players at each of beginner, club, advanced | beginner 3 (p4, p5, p10); club 1 (p12); the rest instructional demos and pros |
| Left-handed players | 4+ players, 60+ strokes | 1 player, 10 strokes |
| One-handed backhand players | 6+ players, 60+ backhands | 6 players, 22 backhands |
| Two-handed backhand players | 8+ players, 60+ backhands | 4 players, 17 backhands |
| Serves | 60+, from 10+ players | 29, from 7 players |
| Forehands | 80+ | 65 |
| Camera: side | 50+ | 10 |
| Camera: behind (incl. high behind the baseline) | 50+ | 16 |
| Camera: front / across the net | 50+ | 119 |
| Camera: oblique | 50+ | 25 |
| Frame rate 50-60 fps | half the footage | 0 (all 24-30 fps) |
| **Hard case:** one-handed backhand on a high ball | 20+ | 0 labelled as such |
| **Hard case:** serve with the toss arm low or hidden | 15+ | 0 |
| **Hard case:** smash / overhead (`other`, `smash`) | 20+ | 19 |
| **Hard case:** slice backhand (`other`, `slice`) | 20+ | 10 |
| Volleys (`other`, `volley`) | 10+ | 16 |

Update the "Now" column whenever clips are added (the counts come from the label files).
Licensed footage found but not yet added is listed in [CANDIDATES.md](CANDIDATES.md); how to
record your own clips (consent, camera, angles, strokes) is in [RECORDING.md](RECORDING.md).
