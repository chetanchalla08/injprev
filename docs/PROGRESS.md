# Progress log

Short dated notes on what changed. Details and numbers are in VALIDATION.md.

## 2026-11-22

- Swings the arm rules leave unlabelled are now labelled from the side the ball was on at contact (PR #2).
- Backhands with a hidden backswing are labelled from the part of the swing seen before the hit (PR #3).
- Still open on p12_rally_front: 4 undetected backhands, 2 false serves, about 9 false swings.
- Tests for the hidden-backswing rules now run from four camera angles, and for left-handers (PR #4).

## 2026-11-23

- New test: no backhand is claimed when the hand stays hidden right up to the hit.
- VALIDATION next steps now list the small-player problems (undetected backhands, false serves, false swings).
- Main reached 30 commits; aiming for a steady count each day.
- Filming tip for better results: the player fills at least a third of the frame height, at 60 fps.
- Next up: the two false serves on p12_rally_front (108.6 s and 119.8 s).

## 2026-11-24

- Fast hand movement that is mostly up and down (ready position, recovery) is no longer a swing (PR #6). Tune false swings 21 -> 15.
- Forehands whose take-back is hidden behind the racket head are labelled forehand with low confidence (PR #6).
- p12_rally_front now has 4 unlabelled swings instead of 7: two false swings (116.2 s, 127.3 s) and two forehands the pose cannot read (88.7 s, 101.2 s).
- Left-handed tests added for both new rules, and the up-and-down test now runs from six camera angles (PR #7 and this batch).
- Next up: a clip of the same player filmed closer, added to held-out, to see how much of the small-player trouble is down to size.

## 2026-11-25

- No pipeline changes today; this entry tracks what is still open.
- p12_rally_front still has two false swings (116.2 s, 127.3 s).
- p12_rally_front still has two forehands the pose cannot read (88.7 s, 101.2 s).
- The four undetected backhands on p12_rally_front are still open.
- The two false serves on p12_rally_front (108.6 s and 119.8 s) are still open.
- Tune false swings stand at 15 after the up-and-down fix (PR #6).
- Filming reminder: the player should fill at least a third of the frame height, at 60 fps.
- Next up: film the same player closer and add the clip to held-out.
