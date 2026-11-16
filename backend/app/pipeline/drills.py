"""Drill suggestions tied to coaching cues.

FIXED CONTENT written by the developers from common coaching practice. None of it has been
reviewed by a qualified coach yet; every drill is shown with that status until a coach
signs it off in reviewer mode. Drills are technique practice, not exercise prescription or
rehabilitation, and none is offered as a way to treat or prevent pain.
"""
from __future__ import annotations

NOT_REVIEWED = "Not yet coach-reviewed"

DRILLS: dict[str, dict] = {
    "serve_sit_and_reach": {
        "id": "serve_sit_and_reach", "name": "Toss, sit, reach (no ball strike)",
        "focus": "Feeling the knees bend as the toss arm goes up.",
        "steps": ["Stand in your serve stance without a ball.", "Lift the toss arm and let both knees bend at the same time; pause for a moment.", "Push up from the legs and reach to your contact point with a slow swing.", "Repeat 8-10 times, then do the same with a real toss, catching the ball instead of hitting it."],
        "caution": "Stop if it is uncomfortable; this is a slow rehearsal, not a strength exercise.",
    },  # fmt: skip
    "serve_half_speed_legs_first": {
        "id": "serve_half_speed_legs_first", "name": "Half-speed serves, legs first",
        "focus": "Starting the serve from the ground rather than the arm.",
        "steps": ["Serve at about half your normal speed from the service line.", "On each serve, say 'down' as the knees bend and 'up' as you drive to the ball.", "Move back to the baseline once the rhythm feels the same every time."],
        "caution": "",
    },  # fmt: skip
    "toss_target": {
        "id": "toss_target", "name": "Toss to a target",
        "focus": "A toss that lands slightly in front of the lead foot instead of drifting sideways.",
        "steps": ["Lay your racket on the ground with the head just in front of your lead foot, pointing to the net post.", "Toss without swinging and let the ball drop.", "Aim for the ball to land on or just beyond the racket head; adjust until 7 of 10 do."],
        "caution": "",
    },  # fmt: skip
    "knee_over_toes_shadow": {
        "id": "knee_over_toes_shadow", "name": "Loading in front of a mirror or camera",
        "focus": "Seeing where the knee points as you load.",
        "steps": ["Stand facing a mirror or film yourself from the front.", "Take your ready position and turn to load for a forehand or backhand in slow motion.", "Check that the bent knee points the same way as the toes of that foot; reset and repeat on both sides."],
        "caution": "This is a visual check of a movement habit. It is not a treatment for knee pain; see a clinician if you have symptoms.",
    },  # fmt: skip
    "low_ready_feeds": {
        "id": "low_ready_feeds", "name": "Low-ready hand feeds",
        "focus": "Getting lower before the ball arrives.",
        "steps": ["A partner drops or tosses balls gently to your hitting side.", "Start each ball from a deliberately low ready position with the knees bent.", "Stay low through the turn, then push up as you swing. 3 sets of 8."],
        "caution": "",
    },  # fmt: skip
    "bounce_hit_rhythm": {
        "id": "bounce_hit_rhythm", "name": "Bounce-hit with a knee bend",
        "focus": "Timing the leg bend to the bounce.",
        "steps": ["Rally or take feeds at an easy pace.", "Say 'bounce' as the ball lands and bend the knees on that word.", "Say 'hit' at contact and rise through the shot."],
        "caution": "",
    },  # fmt: skip
    "finish_and_hold": {
        "id": "finish_and_hold", "name": "Finish and hold",
        "focus": "Letting the swing run to a full finish.",
        "steps": ["Hit easy feeds or shadow-swing.", "After each swing, freeze at the finish for two seconds.", "Check the finish: hand by the opposite shoulder on a forehand; out and up on the hitting side on a backhand.", "Only then recover for the next ball."],
        "caution": "",
    },  # fmt: skip
    "swing_through_targets": {
        "id": "swing_through_targets", "name": "Deep-target rally",
        "focus": "Swinging through the ball rather than stopping at it.",
        "steps": ["Place a marker a racket length inside the far baseline.", "Rally or take feeds aiming to land the ball past the service line every time.", "Count how many of 10 land deep without the swing getting shorter."],
        "caution": "",
    },  # fmt: skip
    "adjustment_steps": {
        "id": "adjustment_steps", "name": "Small steps to the ball",
        "focus": "Moving the feet so the upper body does not have to reach.",
        "steps": ["Take feeds that land a step or two away from you.", "Use at least three small adjustment steps before each hit.", "Aim to make contact with the chest up and the ball at a comfortable distance from the body."],
        "caution": "",
    },  # fmt: skip
}

RULE_DRILLS: dict[str, tuple[str, ...]] = {
    "serve_knee_bend": ("serve_sit_and_reach", "serve_half_speed_legs_first"),
    "serve_trunk_lean": ("toss_target",),
    "knee_alignment": ("knee_over_toes_shadow",),
    "ground_knee_bend": ("low_ready_feeds", "bounce_hit_rhythm"),
    "short_follow_through": ("finish_and_hold", "swing_through_targets"),
    "ground_trunk_lean": ("adjustment_steps",),
}


def for_rule(rule_id: str) -> list[dict]:
    return [{**DRILLS[d], "review_status": NOT_REVIEWED} for d in RULE_DRILLS.get(rule_id, ())]
