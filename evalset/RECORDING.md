# Recording footage for the evaluation set

Your own clips fill the gaps licensed footage can't:
- modern recreational players
- left-handers
- side-on and behind views at 60 fps
- the hard cases on purpose

This page is the checklist for a recording session.

## Before you film: consent

- Ask each person filmed for **written consent** to be recorded and for the clips to be used
  to test the app. A text or email that says so is enough. Record who agreed and when.
- **Anyone under 18 needs a parent or guardian's consent.**
- Tell them the footage stays on your machine (it is never committed to git), and that you
  will delete their clips if they ask. To delete: remove `evalset/clips/<id>/`, its label file
  and its manifest entry.
- Give each person a stable id (`p10`, `p11`, …) and keep the name-to-id list **outside the
  repo**. Only the id goes into `manifest.json`.

## Camera

- Phone in landscape, **60 fps** (1080p60 is fine; use slow motion only if it records at
  real 120/240 fps).
- **Fixed:** a tripod, or propped against the fence or a bag. No handheld panning.
- **Whole body in frame for the whole stroke, racket included.** The player should fill
  about half the frame height. Step back rather than crop the feet or the finish.
- One player per clip where possible. If others are in view, note where the player is.

## Angles

Film each player from at least two of these, as separate clips.

| Angle | Where the phone goes | `camera_view` |
|---|---|---|
| Side-on | On the side fence, level with the baseline | `side` |
| Behind | Behind the baseline, centred, at chest height | `behind` |
| High behind | On the back fence, raised (phone clamp or on a bench) | `behind` (note "high") |
| Front | Across the net, facing the player | `front` |
| Oblique | Back corner, about 45° | `oblique` |

## What to hit (about 10 of each per player per angle)

1. Forehands from a feed or rally.
2. Backhands from a feed or rally. Note one- or two-handed.
3. Serves, first and second.
4. **Hard cases:**
   - **one-handed backhands on high balls** (feed them high, to shoulder height), if the
     player hits one-handed
   - **serves with a low or barely visible toss arm** (ask them to keep the toss low or
     tucked)
   - **smashes and overheads**, from lobs
   - **slice backhands**
   - a few **volleys**

Say the stroke out loud before each one, or keep a quick list. It makes labelling much faster
and more accurate.

## Who to film

Aim for variety rather than many clips of one person:
- different levels (beginner, club, advanced)
- left- and right-handed
- one- and two-handed backhands
- women and men
- juniors (with guardian consent) and older players

Five different people filmed once beats one person filmed five times.

## After the session

1. Copy the clips to your computer and tell Claude:
   - who is in each clip (by id)
   - their level
   - left- or right-handed
   - one- or two-handed backhand
2. Each clip is added with `python -m app.evaltools.add` (see `README.md`), **with its split
   chosen before scoring**: a new person goes wholly into tune or held-out.
3. Labels are made from the footage and marked non-expert until a coach checks them in the
   app's Label tab.
