# Validation status and plan

Two different things are kept apart here. **Technical validation** asks whether the software
does what it says: finds strokes, times them, measures angles. **Medical validation** asks
whether anything it reports means something about a person's body. Some technical checking
has been done on a very small set. No medical validation has been done.

## 1. Technical validation

### 1.1 Software tests
`backend/tests` (168 tests) cover the original cases (corrupt and odd video files, frame
rates, rotation, synthetic players from six camera angles, hidden limbs, tracking gaps, two
people, upload limits, access control, deletion, retention, retries) plus, for the newer work:
label-file validation and scoring arithmetic; labelling, joint-annotation and motion-capture
comparison; reviewer sign-off (wrong code, wrong role, stale wording, withdrawal); ball found
and contact located; **no ball found**; a blob that only moves with the hand rejected as the
ball; court recovered from a rendered court; **no court lines** and too few lines; a real
encoded scene cut; a **cut in the middle of a stroke**; **a different person after a cut**, a
bystander dressed like the player, a doubtful match held for confirmation and the confirm /
reject flow; comments, drawings, profiles and progress; share links (unguessable, read-only,
no health details, **expired**, **revoked**); deletion and retention of every new kind of data;
PDF export; encryption round trips, byte ranges, tampering, **missing key**, wrong key; framing
guidance; and the LLM check (**invented number**, citation, risk wording).

These show the software behaves as designed. They say nothing about accuracy on real players.

### 1.2 Evaluation set
`evalset/` (format and commands in `evalset/README.md`). Today: 14 clips cut from 7 freely
licensed Wikimedia Commons files; 10 are labelled, about 173 s, 5 players, **41 strokes** (tune
27, held-out 14). **All labels are non-expert** (made by the developer's AI assistant from film
strips; contact to about ±0.04 s where the ball is visible, ±0.1 to ±0.2 s otherwise). The
held-out split is real hitting with a visible ball that was not looked at while tuning; one
player appears in both splits in different footage. With numbers this small one stroke moves a
percentage by 4 to 7 points. Treat everything below as a development aid.

Score definitions: a detection matches a labelled stroke within ±0.5 s of contact. *Correct* =
detected and given the right label. *Timing* = median absolute difference from labelled contact.

### 1.3 Results by change

| Configuration | Tune (25 strokes) recall / precision / correct / timing | Held-out (14) recall / precision / correct / timing | Decision |
|---|---|---|---|
| First build (baseline) | 76% / 100% / 44% / 0.10 s | 100% / 100% / 64% / 0.05 s | – |
| + scene cuts and re-acquisition | 76% / 100% / 44% / 0.10 s | 100% / 100% / 64% / 0.05 s | Adopted (see below) |
| + ball tracking for contact time | 76% / 100% / 44% / **0.04 s** | 100% / 100% / 64% / 0.05 s | Adopted for timing; contact point not validated |
| + hidden-arm recovery from the ball | 76% / **34%** / 44% / 0.04 s | 100% / 93% / 64% / 0.05 s | **Rejected**, off by default |
| cropped second pose pass (instead of full frame) | 72% / 95% / 44% / 0.07 s | 100% / 100% / 64% / 0.07 s | **Not adopted** |
| Final defaults (cuts, re-acquisition, ball, court) | 76% / 100% / 44% / 0.04 s | 100% / 100% / 64% / 0.05 s | Shipped |
| + high finish after a low fastest point is a groundstroke, not a serve | 76% / 100% / 44% / 0.03 s | 100% / **93%** / **86%** / 0.04 s | Adopted (see below) |
| + player 3 tune clip (27 strokes from here); held high finish, arm coming down, hidden backswing | 74% / 100% / 44% / 0.03 s | 100% / **100%** / **100%** / 0.04 s | Adopted (see below; **second use of held-out**) |

What the baseline numbers hide:
- Tune: 6 strokes missed (3 very slow shadow backhands, 3 by the panning-camera left-hander) and
  8 found but left unlabelled, mostly a right-hander's shadow forehands seen from the front.
- Held-out: every stroke was found, but **all 5 one-handed backhands were labelled as serves**
  (the high finish looks like a serve reach). This was seen only on held-out footage, so it has
  deliberately not been tuned away; it is the top stroke-labelling priority and needs tune-split
  footage of one-handed backhands first.

**One-handed backhand vs serve (run `bh1_follow_through`).** A high hand is no longer a serve
when the hitting hand was fastest well below the head (≥ 0.5 torso lengths under the nose) at
least 0.1 s before reaching its highest point: in a serve the hand is fastest near the top, in
a groundstroke it is fastest around the waist and rises in the follow-through. The fastest
moment comes from the single-camera 3D estimate where available, because a swing toward the
camera nearly vanishes in 2D. The rule was set from the mechanics of the two strokes and checked
on synthetic swings from six views (new tests); held-out details were not looked at, and the
held-out split was scored **once** with no changes afterwards. Result: 3 of 5 one-handed
backhands now labelled correctly, 2 still labelled serves, and one new false detection in the
same clip; tune split and all serves unchanged. The held-out split has now been used to judge
this change, so the next improvement to this confusion needs new held-out footage, and tune-split
one-handed backhands are still missing.

**One-handed backhand vs serve, second round (run `p3_bh1_tune`).** A new tune clip,
`p3_bh1_shadow`, was cut from the part of `0REVES 1 MANO` before its first cut: player 3, two
shadow one-handed backhands seen from the front, both finishing with the hand above the head. It
does not overlap the held-out clip from the same file (a different player, after the cuts). Labels are
non-expert, like the rest. On it the old rule did not call either swing a serve, but it showed three
weaknesses: (1) the hand was fastest at shoulder height (0.44 TL below the nose), so the "groundstroke
finishing high" escape (needed ≤ 0.5 TL below) would not have applied had the finish been slightly
higher, and the search for the fastest moment started only 0.4 s before the top, so a finish held up
there could hide the swing that raised it; (2) bringing the racket back down from the high finish was
detected as a separate "forehand", twice; (3) the hand went out of sight behind the body in the
backswing, so the hidden-arm guard left the real backhand unlabelled. Changes, each from the mechanics
of the strokes and checked on synthetic swings from six views: the escape now applies when the hand is
fastest ≥ 0.25 TL below the nose (serve contact is well above the head) and the fastest moment is
searched from before the hand first got high; a fast movement that starts above the nose and is still
falling at its peak is not a swing; and a hidden backswing whose estimated position is across the body,
followed by a visible swing out to the hitting-arm side, is a backhand at confidence 0.45. A first
version of the "coming down" rule dropped one of Rola's looped forehands (racket above the head in the
take-back); requiring the hand to be falling at the peak fixed it, and a looped-forehand synthetic test
was added. Tune split: the six existing clips are unchanged stroke for stroke; on the new clip both false
strokes are gone and one backhand is labelled (the other is a deliberately slow demonstration swing,
below the 6 TL/s swing threshold, and stays missed). The held-out split was then scored **once**, with no
changes afterwards: 14 of 14 correct (was 12), no false detections (was 1); both remaining one-handed
backhands are no longer serves. **This is the second time the held-out split has judged a change to this
confusion**, and the second change was designed after seeing the first's held-out summary (2 of 5
wrong), so 14 of 14 overstates what to expect on new footage. Treat the held-out split as used up for
this question: the next check needs new players.

**Ball tracking.** Two detectors were tried on the tune clips: a motion-and-colour finder and
a pretrained general object detector (YOLOX-S, COCO "sports ball"). Contact was located for 12
of 14 strokes with the first, 6 of 14 with the second, 13 of 14 with both; the first is used
(no extra weights, about 2 ms per frame). A first version also "found" contact in all 5 shadow
swings that have no ball, because it followed the racket; requiring the ball to come from
beyond arm's reach cut that to 1 of 5 and lifted tune timing to 0.04 s. On held-out footage
contact was located for only **2 of 14** strokes and timing did not improve. So: the stroke
timestamp uses ball contact when found and says which was used, but contact point relative to
the body stays in the "not assessed" list as an unvalidated observation, and no rule uses it.
The racket: the same pretrained detector boxes the racket in many frames, but there is no
reference to score it against, so nothing is concluded from it and it is not part of the
pipeline (`pipeline/objects.py` is kept for future evaluation). Racket tracking is therefore
not delivered.

**Scene cuts and re-acquisition.** All 6 known cuts in the three cut clips were found, with no
false cuts in any of the 13 clips. After a cut to a different player (2 clips) nothing was
followed; after a cut to the same player in a wider shot (1 clip) the player was picked up
automatically (clothing match 0.81, just over the 0.80 bar) and tracked coverage rose from 38%
to 87%. Three clips is an anecdote, and the thresholds were set on them.

**Court lines.** The fit recovers a rendered full court to within 0.3 m, but found a mapping in
**0 of 13** real clips, which show two or three lines each: too few to place the court without
guessing. Position on court, recovery and change of direction are therefore untested on real
footage, and camera view is still judged from body orientation on all of this set.

**Hops (split-step candidates).** Built from pose alone; no footage is labelled for split
steps, so there is no number to report.

### 1.4 "Not assessed" items revisited
| Item | Outcome | Why |
|---|---|---|
| Contact point and timing | Timestamp uses ball contact when found; contact point **stays not assessed** | Found for 2 of 14 held-out strokes; 1 of 5 false in ball-less swings |
| Wrist position and grip | Stays not assessed | Cropped higher-resolution pose scored no better; no reference data for hand landmarks |
| Segment sequencing | Stays not assessed | Gaps of a few hundredths of a second are about one frame at 25–30 fps; no reference data |
| Footwork (split step, recovery, direction changes) | Built, **stays not assessed** | Court found in 0 of 13 clips; no split-step labels |

Nothing moved into the supported set.

### 1.5 Not measured at all
- **Joint position and angle accuracy.** The tools exist (Annotate tab, motion-capture CSV,
  `python -m app.evaltools.angles`). No reference data has been supplied, so there is no figure.
- **Whether each rule's flagged clips are right.** The Review tab and export exist. No coach has
  reviewed any clip.
- Live camera capture was built and its framing checks are unit-tested, but it has not been run
  against a real camera.

## 2. Review by people (process, and what is still blank)

| What | Who | How | Status |
|---|---|---|---|
| Stroke labels | Qualified coaches | Session → **Label** tab → export → `app.evaltools.add` | None supplied (all labels non-expert) |
| Joint positions | Anyone careful; ideally two people | Session → **Annotate** tab → export to `evalset/annotations/` | None |
| Reference angles | Biomechanics lab | CSV per clip in `evalset/mocap/` or upload in the Annotate tab | None |
| Flagged clips and cues | Two or more coaches, independently | Session → **Review** tab; `app.evaltools.reviews` reports agreement | None |
| Evidence entries, interpretations, pain guidance | Sports physiotherapist or physician | **Review** page with the operator's `REVIEWER_CODE` | None signed off |
| Drills | Coach | **Review** page | None signed off ("Not yet coach-reviewed") |

A sign-off records who, in what role, and a fingerprint of the exact wording; editing the
wording afterwards returns the item to "review needed". The software never marks anything as
reviewed by itself. Before release the Hewett 2005 and Krosshaug 2016 DOIs should be re-checked
by the clinical reviewer; no new references were added in this round.

## 3. Medical validation

None has been done and none is claimed. Nothing in the evaluation above bears on injury.
Showing that any video-derived pattern relates to injury would need prospective studies and is
outside this product's scope. Detection confidence describes how clearly something was seen; it
is not a probability of anything about the player's body.

## 4. Next steps, in order
1. Coach-labelled footage: one- and two-handed backhands, left-handers, side-on and behind views,
   recreational players; at least a few hundred strokes, with a held-out split of different players.
2. Re-judge one-handed backhand vs serve (14 of 14 on a twice-used held-out split) on new tune
   footage, and judge it on new held-out footage.
3. Footage with a full court in view and labelled positions, before any footwork claim.
4. Joint annotation on 100+ frames across views, then motion capture, before trusting any angle.
5. Coach review of flagged clips; clinician sign-off of evidence and wording.
