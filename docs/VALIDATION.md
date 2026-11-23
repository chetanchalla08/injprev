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

**Smash vs serve (run `smash_fix_tune`, tune split only; tried and reverted, see below).** With the expanded set (146 strokes, 14
players; see `evalset/README.md`) the new `subtype` report showed that 13 of the 14 tune smashes the app
found were called serves. Measured on tune clips only: in the 2 s before the hit, the other arm stayed
above the head for 0.07-0.40 s in serves (one exception at 0.77 s) and 0.53-1.36 s in smashes (one
exception at 0.30 s, a far shot); in a smash it points at the lob while the player waits. Rule: ≥ 0.6 s
up → overhead, left unlabelled (0.6 rather than 0.5 so the synthetic serves, held about 0.53 s, keep a
margin). Tune result: smashes called serve 13 → 3, smashes left unlabelled 1 → 11; one serve lost (the
0.77 s exception), serves labelled correctly 9 → 8; no other stroke changed in the 45 tune clips. Serves
with a slow, high toss are the known risk.

**Held-out, scored once after this change (run `smash_fix_heldout`, 13 clips, 31 strokes).** Overall recall
81%, precision 89%, correct 52%. Smashes: none called a serve (3 unlabelled, 1 missed). But serves: only 1
of 6 labelled correctly, 5 left unlabelled, and 4 of those 5 are the overhead rule itself: p1's four serves in
`p1_serve_ball` held the toss arm up 0.7-0.8 s, past the 0.6 s cutoff. The same four serves were labelled
correctly in the 20 Nov run (`p3_bh1_tune`), before the rule. So the risk named above is real: the rule does
not generalise to a slower toss, and the tune numbers overstate it. Held-out has now been seen, so the
cutoff must not be tuned against these four serves; any rework needs a reason found on tune footage.

**Reverted.** Tune was then searched for a second measure that keeps slow-toss serves (lowest racket-hand
height in the 2 s before, racket-hand height when and while the toss arm is up, hip movement): none
separates the one slow-toss tune serve (p6, 0.77 s) from the smashes, so no rework could be justified on
tune. The rule was removed: serves are a supported stroke and smashes are not, so losing real serves costs
more than calling a smash a serve. Smashes are therefore called serves again (13 of 15 on tune before the
rule). To revisit it, tune needs serves with a slow, high toss and smashes inside rallies.

**Ball contact (runs `ball_tune_*`, `ball_b2_tune`, `ball_b2_heldout`; tried and reverted).** The report now
says, of detected strokes with a visible ball, how often contact was found, and on how many strokes with no
visible ball (shadow swings, any stroke type) a contact was claimed anyway. Before this change: tune 10 of 14
found, 16 claimed with no ball; held-out (`smash_fix_heldout`) 4 of 14 found, 3 claimed with no ball. (The
README's earlier 12/14 and 2/14 were from older runs.)

Diagnosis on tune only: three of the four tune misses are clay-court backhands in `p1_bh2_ball`, where the
blurred ball reads as very bright pale orange (OpenCV hue 10-14, value about 250), outside the yellow-green
range; the fourth (`rola_a` 5.45 s) has the ball out of frame. Most false claims come from blobs on the
player (face, shoes, a yellow shirt) or from a court blob linked onto racket blur, where one point beyond
arm's reach was enough for the track to count as a ball. Variants on tune:

| variant | found (of 14) | claimed with no ball (of 84) |
|---|---|---|
| current | 10 | 16 |
| + pale-orange hue band (5-20, value ≥ 230) | 13 | not counted on all strokes (6 on fh/bh/serve, up from 3) |
| + mask of head, legs, torso | 13 | 18 |
| two far points only | 9 | 9 |
| **B2: hue band + mask + two far points** | **11** | **9** |

**Held-out, scored once with B2 (run `ball_b2_heldout`, 13 clips).** Contact found 4 → 7 of 14, claimed with no
ball 3 → 3. But stroke timing got worse: on the 14 detected strokes with a visible ball, median error
0.04 → 0.07 s, within 0.10 s 11 → 9, more than 0.20 s off 0 → 2. Several new "contacts" replaced pose
timestamps that were already right (`p1_serve_ball` 5.64 s: 0.00 → 0.28 s; `p2_fh_ball` 8.8 s: 0.00 →
0.28 s). Tune had shown the same direction (median 0.05 → 0.07 s). Since the report presents ball timing as
the better source, a change that finds more contacts but times strokes worse is not an improvement.

**Reverted.** Only the measure was kept. Held-out has now been seen for this change, so B2 must not be tuned
against these strokes. A future attempt needs, on tune, a check that a found contact beats the pose
timestamp (for example agreement with the hand-speed peak) before it replaces it.

**Small player in modern phone footage (tune clip `p12_rally_front`, runs `p12_baseline`, `p12_cropped`,
`p12_ball_side`; tune split only).** A user's 2.5-minute rally clip (960x540, 30 fps, fixed camera across the
court, player about 90 px tall, about 20% of the frame height) came back as 18 forehands, 1 backhand, 2 serves and
13 unlabelled; read by eye at 0.1 s steps it holds 25 forehands, 7 two-handed backhands and no serves. Added to
tune as player p12 (32 strokes). Baseline on the clip: 25 of 32 found, 16 labelled correctly, 9 false detections.
Why, from the pose data: the hitting wrist's visibility flickers frame to frame (about half of the frames below
0.5 have a frame within ±2 that is above 0.7), and the wrist *positions* on those frames are also wrong.
- 5 of the 7 backhands were missed or unlabelled: during a two-handed backhand the wrists overlap and visibility
  falls to 0.34-0.43, under the swing finder's 0.45.
- 10 strokes were left unlabelled by the hidden-lead-up rule, which fires because of flicker, not a hidden arm.
- Both "serves" rest on single frames: at 108.6 s a three-frame wrist jump above the head while in the ready
  position, at 119.8 s a forehand with a high finish whose fastest frame was an unseen, interpolated one.
- The 9 false detections are real hand movement (racket coming back from the finish 0.9-1.3 s after a shot,
  ready-position movement), not jitter: neither speed, travel on well-seen frames, nor direction separates them.

Tried and **rejected**: treating short visibility drops as flicker (rolling maximum over ±2 frames) — false
detections 9 → 18 on the clip, because the positions on those frames are wrong too. The cropped second pose pass
(`p12_cropped`) found more strokes (25 → 28) but labelled fewer correctly (16 → 13; unlabelled 8 → 13) — **not
adopted**.

**Adopted: label an unlabelled stroke from the ball's side at contact** (`segment.ball_side_label`). When ball
tracking found a contact and the pose left the stroke unlabelled, the ball's position relative to the hip centre
decides it: on the hitting-arm side, forehand; across the body, backhand (confidence 0.45). It abstains when the
ball is within 0.8 torso lengths of the hips or the shoulders are within 0.25 torso lengths of each other across
the image (side-on, where left and right say nothing). Of the clip's 23 real strokes with a found contact it
gives a side for 5, all right and all forehands; it abstains on both backhands that have a contact. Tune (46 clips, 102 strokes): correct 41 → 45 (40% → 44%), unlabelled
22 → 18, wrong labels 3 → 3, detection unchanged. `p12_rally_front` 16 → 19, `p1_fh_shadow` 0 → 1. Held-out
was not scored for this change.

Still open on this clip: the missed two-handed backhands, the two false serves, and the 9 false detections. The
serve fixes tried (a toss seen on more than one frame; ignoring unseen frames when finding the fastest moment)
changed nothing on the clip and were not kept. The real remedy is better input: the player filling at least a
third of the frame height, and 60 fps.

**Backhands with a hidden lead-up on small players (tune clip `p12_rally_front`, runs `p12_bh_hidden`,
`ho_before_bh`, `ho_bh_hidden`).** Of the clip's 7 two-handed backhands, 4 are never detected (hitting-wrist
visibility 0.34–0.45 around the peak, or too little hand travel) and 2 were detected but left unlabelled by the
hidden-arm guard. The guard's backhand exception required the pose model's low-visibility estimate to put the hidden
hand across the body; on this clip that estimate is on the wrong side even for real backhands (median +0.78 and
+0.09 hip widths toward the hitting-arm side), while the part of the swing seen before the hit travels clearly
backhand-wise. The exception no longer uses where the hidden hand was estimated; it still needs the hand seen
again before the hit (visibility ≥ 0.5) and travel out to the hitting-arm side of at least 0.6 hip widths
(confidence 0.45). A forehand with a hidden backswing stays unlabelled as long as its seen part travels
forehand-wise; the risk is a stroke whose seen part travels backhand-wise, as the false detection below does. Tune: correct 45 → 49 (44% → 48%),
unlabelled 18 → 14, wrong labels 3 → 3, detection and false detections unchanged; on `p12_rally_front` backhands
labelled 1 → 3, and one false detection (a recovery movement at 99.4 s) is now labelled backhand instead of left
unlabelled. Held-out, checked once and not tuned on: no change on any count (correct 20 → 20, wrong 1 → 1).
Rejected: the "use the better-seen wrist" idea for two-handed backhands (on this clip the other wrist is seen even
less, mean visibility 0.07–0.57) and lowering `MIN_SWING_VIS` (0.40: one more backhand, one more false detection;
0.35: one backhand labelled forehand and two more false detections). The 4 undetected backhands remain.

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
6. Small players in phone footage (`p12_rally_front`): find two-handed backhands whose hitting wrist is
   barely seen (4 of 7 undetected) without adding false swings; stop single noisy high-wrist frames and high
   forehand finishes from becoming serves; tell recovery movements from swings. More clips like it are needed
   before any of these is tuned, so one clip does not decide the rules.
