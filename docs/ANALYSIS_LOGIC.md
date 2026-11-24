# Analysis logic

Everything below is implemented in `backend/app/pipeline/`. Constants are named in the code.
All thresholds are developer heuristics, not validated values and not safety limits.

## 1. Video
Reject non-video, undecodable, too long/short files. Re-encode to H.264 MP4, constant
frame rate (source rate, clamped 15–60), longest side ≤ 1280, rotation applied, no audio.
The original is deleted. Quality warnings: brightness, Laplacian sharpness, frame rate
< 50, short side < 480 px, global image shift between frames (camera motion).

## 2. Player selection and tracking
People are detected in nine sampled frames; the frame with the most people is shown for
selection (auto-selected if one). All frames are then processed with up to four poses.
The chosen player is followed forwards and backwards by nearest torso centre within a
gate of 0.35 × body height (at most doubling while missing), requiring similar size, and
abandoned after 1.5 s without a match or at a scene cut. Under 30 % tracked → the job fails
with advice.

**Scene cuts** (`visual.py`): a frame whose colour histogram distance from the previous frame
is ≥ 0.35 and whose brightness-pattern change is ≥ 14 grey levels and ≥ 3.5 × the local median;
also the first visible frame after a run of near-black frames. Slow dissolves are not detected.

**Re-acquisition** (`pose.track_player`): after a cut, or after the player leaves the frame, a
person can be picked up again only if they were not already visible alongside the player, and
their torso and upper-leg colour histograms match the player's. Match ≥ 0.80 and clearly better
than anyone else in view → continues automatically. Match 0.60–0.80, or nothing to compare and
no cut → kept but marked *needs confirmation* and **left out of the analysis until the user
confirms**. Below 0.60 → not followed. Colour matching cannot tell apart people dressed alike.

**Optional cropped pose pass** (`POSE_MODE=cropped`): the pose model is run again on an enlarged
crop of the player. Not better on the evaluation set, so off by default.

## 3. Signals
Landmarks with visibility < 0.5 are masked; gaps ≤ 0.12 s are interpolated; a 3- or
5-frame binomial filter is applied. Body-size unit: shoulder-midpoint to hip-midpoint
distance (rolling 2 s median), "torso lengths" (TL).

## 4. Stroke detection
Hitting-wrist position is taken relative to the shoulder midpoint (removes running and
camera pan), median-filtered, and differentiated to a speed in TL/s.
- **Swing**: local speed maximum >= 6 TL/s, hand travel >= 1.2 TL within +-0.3 s of which >= 0.5 TL is side to side in the image (`MIN_ACROSS`), mean wrist visibility >= 0.45, >= 0.7 s from a faster one. (Lifting the racket into the ready position, or bringing it down after a shot, is fast and long but almost all up and down.) Not a swing if the hand was above the nose within the 0.3 s before and is still falling at the peak, >= 0.5 TL lower: that is the racket being brought back down after a high finish (a looped forehand also starts high, but is rising again at the hit).
- **Serve**: hand >= 0.5 TL above the nose for 0.08-1.5 s, with a toss-arm lift beforehand or a fast hand nearby. Timed at the highest hand position. Without a toss it may be an overhead. Swings from 1.6 s before to 1.0 s after belong to the serve. Not a serve if the hand was fastest (3D estimate where available, searched from 0.4 s before the hand first got that high) ≥ 0.25 TL below the nose and ≥ 0.1 s before its highest point: that is a groundstroke finishing high, such as a one-handed backhand. (Serve contact is well above the head.)
- **Forehand / backhand**: for two-handed-backhand players, the median distance between the wrists over the 0.4 s before the peak (< 0.35 m model units = backhand, > 0.42 = forehand), cross-checked with the hand's travel along the hip line; for one-handed players the hip-line travel alone (>= 0.6 hip widths), with confidence capped when the player is side-on.
- **Hidden-arm guard**: if the hitting wrist was below 0.5 visibility for most of the 0.4-1.2 s before the peak, the swing is left unlabelled. Exception: if the hand is seen again 0.1-0.4 s before the peak and then travels out to the hitting-arm side (≥ 0.6 hip widths), it is a backhand whose backswing went behind the body; labelled backhand with confidence 0.45. Where the model estimates the hidden hand is not used, because for a small player that estimate is often on the wrong side of the body; so any stroke whose seen part travels that way is labelled backhand. Second exception: if the hand is seen again 0.1-0.4 s before the peak and then swings across the body from the hitting-arm side by ≥ 1.5 hip widths (much more than the backhand's 0.6, so a recovery movement that drifts across is not taken), it is a forehand whose take-back was hidden, typically behind the racket head when filmed from the front; labelled forehand with confidence 0.45 (`HIDDEN_FH_MIN_SCORE`).

### 4a. Ball, contact and phases (`ball.py`, `segment.apply_ball`)
Ball candidates are small blobs that differ from both the previous and the next frame (after
cancelling camera shift) and are tennis-ball yellow-green; candidates are linked into smooth
tracks, and blobs that never get further than 2.8 torso lengths from the hand (the racket) are
ignored. Contact is where an incoming and an outgoing track meet, or one track turns sharply,
within 2.2 torso lengths of the hitting wrist and within 0.3 s of the pose-based timestamp.
When found, the stroke is timed at contact (`timing.source = ball_contact`); otherwise it keeps
peak hand speed (serves: highest hand) and the report says so. Contact point relative to the
hips (height, distance, and "in front" only when side-on) is a **2D image-plane** quantity shown
as an unvalidated observation. One rule uses the contact's side: an auto-detected stroke the swing
rules left `unclassified` is labelled forehand when the ball at contact is at least 0.8 torso
lengths to the hitting-arm side of the hip centre, backhand when as far on the other side (which
side is which comes from the shoulders' image x-difference, median over ±1 s). It abstains when
the ball is closer to the hips or the shoulders are within 0.25 torso lengths of each other
(side-on), gives confidence 0.45, never changes a label the user set, and is undone on a rebuild
without the ball or without a contact. With the player facing away it relies on the pose model
keeping left and right landmarks on the right body side. Recovering strokes from the ball when the arm
is hidden exists but is off (it produced false strokes). Phases: the forward swing starts when
hand speed last dropped below 30 % of its peak; preparation is before, follow-through after
the timestamp.

### 4b. Court and movement (`court.py`)
Bright thin low-saturation pixels → Hough lines → two direction families → every assignment of
two detected lines per family to standard court lines gives a homography, scored by how many
court-line samples land on line pixels. Accepted only with ≥ 5 court lines supported and a clear
margin over differently-placed alternatives; otherwise **unable to assess**. With a mapping:
foot midpoint on the court plane (2D), zone at each stroke, position 1.5 s later, sideways
direction changes, and camera view from the camera's position relative to the player. Hops
(both ankles and hips rising ≥ 0.035 torso lengths and returning within 0.4 s) are found from
pose alone. All of this is unvalidated and none of it feeds a finding.

Timestamps are ball contact only where the report says so.

## 5. Measurements (method in brackets)
Peak hand speed [relative]; deepest knee bend per leg while loading [larger of 2D image
angle and single-camera 3D estimate]; elbow angle at peak [3D estimate, 2D also shown];
trunk tilt from vertical [2D]; serve reach height [relative]; follow-through travel past
mid-chest [3D estimate]; knee offset from hip–ankle line [2D, only when the hip line is
within 30° of the image plane and the knee is bent ≥ 20°]; shoulder-line rotation [3D estimate].
A stroke with < 80 % tracked frames is not measured. Pose measurements stay anchored to the
pose-based timestamp even when the stroke is timed at ball contact. Joint-angle-over-time plots
(`angles.py`) show 2D and single-camera-3D series separately and leave gaps where a joint was
below the visibility cut-off.

## 6. Rules
| Rule | Strokes | Fires when | Health-related evidence |
|---|---|---|---|
| Limited knee bend before serve | serve | both knees < 30° | Elliott 2003 |
| Pronounced sideways trunk tilt | serve, frontal view only | tilt > 35° | Campbell 2013 |
| Knee inward of foot line | all, frontal view only | offset > 8 % leg length | Hewett 2005 / Krosshaug 2016 (mixed) |
| Upright legs while loading | forehand, backhand | both knees < 25° | none — technique only |
| Short follow-through | forehand, backhand | travel < 0.15 / 0.3 shoulder widths | none |
| Trunk tilted at the hit | forehand, backhand | tilt > 30° | none |
| Hand-speed change | any with ≥ 6 strokes | last third vs first third differs ≥ 15 % | none; cause not attributed |

"Recurring" = at least 3 strokes and ≥ 60 % of assessable strokes. A health-related
interpretation is attached only via an entry in `evidence.py`.

## 7. Detection confidence
`min(landmark visibility, tracking continuity, view suitability, method cap) × (0.7 + 0.3 × support)`
where support is 0.6 / 0.8 / 1.0 for 1 / 2 / 3+ strokes and the method cap is 0.75 for
single-camera 3D estimates. ≥ 0.75 high, ≥ 0.55 moderate, ≥ 0.4 low, below → not reported.
It describes how clearly something was seen. It is not an injury probability.

## 8. Fixed content, review status and the optional LLM
Drills (`drills.py`) are fixed text attached to each cue and shown as "Not yet coach-reviewed"
until a coach signs them off. Review status of evidence, rule wording, pain guidance and drills
comes only from sign-offs recorded in reviewer mode (`review.py`), tied to a fingerprint of the
wording. The optional plain-language summary (`llm.py`) sends the findings JSON alone to a
language model and discards the output if it contains a number not in that JSON, anything that
looks like a citation, or diagnostic or risk wording. It never feeds back into the report.
