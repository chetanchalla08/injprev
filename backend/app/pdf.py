"""Structured PDF export of one session, laid out for a coach or clinician to read or file.

Built on request from the stored report and pose data; nothing is written to disk. The same
separation as on screen is kept: Observation / Interpretation / Evidence / Action, each
measurement with its method (2D image plane vs single-camera 3D estimate), and angle plots
that leave gaps where tracking was poor.
"""
from __future__ import annotations

import time

from fpdf import FPDF

from . import angles
from .pipeline.signals import PoseTrack

METHOD = {"image_plane": "2D image angle", "estimated_3d": "single-camera 3D estimate", "relative": "relative to body size", "image_plane+estimated_3d": "larger of 2D and 3D estimate"}
SUBS = {"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "≥": ">=", "≤": "<=", "…": "...", "×": "x", "•": "-", "→": "->"}
COLORS = {"image_plane": (31, 92, 64), "estimated_3d": (185, 88, 47)}


def _t(s) -> str:
    s = str(s if s is not None else "")
    for a, b in SUBS.items():
        s = s.replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")


class _Doc(FPDF):
    def __init__(self, title: str) -> None:
        super().__init__(format="A4")
        self.doc_title = title
        self.set_auto_page_break(True, margin=16)
        self.set_margins(16, 16, 16)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(110)
        self.cell(0, 4, _t(f"{self.doc_title} - technique analysis and education only; not a diagnosis, not clinically validated - page {self.page_no()}"), align="C")
        self.set_text_color(0)

    def hd(self, text: str, size: int = 12) -> None:
        self.ln(2)
        self.set_font("Helvetica", "B", size)
        self.multi_cell(0, size * 0.5, _t(text), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def para(self, text: str, size: float = 9, style: str = "", color: int = 0) -> None:
        self.set_font("Helvetica", style, size)
        self.set_text_color(color)
        self.multi_cell(0, size * 0.48, _t(text), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0)

    def field(self, label: str, text: str) -> None:
        if not text:
            return
        self.set_font("Helvetica", "B", 8.5)
        self.cell(30, 4.2, _t(label), new_x="RIGHT", new_y="TOP")
        self.set_font("Helvetica", "", 8.5)
        self.multi_cell(0, 4.2, _t(text), new_x="LMARGIN", new_y="NEXT")

    def table(self, head: list[str], rows: list[list], widths: list[float]) -> None:
        self.set_font("Helvetica", "B", 7.5)
        for h_, w in zip(head, widths):
            self.cell(w, 5, _t(h_), border="B")
        self.ln()
        self.set_font("Helvetica", "", 7.5)
        for r in rows:
            if self.get_y() > 270:
                self.add_page()
            for v, w in zip(r, widths):
                self.cell(w, 4.6, _t(v)[: int(w * 0.75)])
            self.ln()
        self.ln(1)


def _plot(pdf: _Doc, data: dict, entry: dict, x: float, y: float, w: float, h: float, marks: dict[str, float]) -> None:
    """One angle-over-time panel. Lines break where a value is missing."""
    t = data["t"]
    vals = [v for m in entry["methods"].values() for v in m if v is not None]
    pdf.set_draw_color(200)
    pdf.rect(x, y, w, h)
    pdf.set_font("Helvetica", "B", 7)
    pdf.text(x, y - 1, _t(f"{entry['label']} ({entry['note']}) - tracked {entry['tracked_frac']:.0%} of frames"))
    if not vals or len(t) < 2:
        pdf.set_font("Helvetica", "", 7)
        pdf.text(x + 3, y + h / 2, "Unable to assess: the joints needed were not tracked in this stroke.")
        return
    lo, hi = min(vals) - 5, max(vals) + 5
    X = lambda tt: x + (tt - t[0]) / (t[-1] - t[0]) * w  # noqa: E731
    Y = lambda v: y + h - (v - lo) / (hi - lo) * h  # noqa: E731
    pdf.set_font("Helvetica", "", 6)
    pdf.text(x - 7, y + 2, f"{hi:.0f}")
    pdf.text(x - 7, y + h, f"{lo:.0f}")
    pdf.text(x, y + h + 3, f"{t[0]:.2f} s")
    pdf.text(x + w - 8, y + h + 3, f"{t[-1]:.2f} s")
    for name, tt in marks.items():
        if tt is not None and t[0] <= tt <= t[-1]:
            pdf.set_draw_color(120)
            pdf.set_dash_pattern(dash=1, gap=1)
            pdf.line(X(tt), y, X(tt), y + h)
            pdf.set_dash_pattern()
            pdf.text(X(tt) + 0.5, y + 3, _t(name))
    for i, v in enumerate(entry["visibility"]):  # shade stretches below the visibility cut-off
        if v < data["min_visibility"] and i + 1 < len(t):
            pdf.set_fill_color(235)
            pdf.rect(X(t[i]), y, max(X(t[i + 1]) - X(t[i]), 0.2), h, style="F")
    for m, series in entry["methods"].items():
        pdf.set_draw_color(*COLORS[m])
        pdf.set_line_width(0.35)
        for i in range(len(series) - 1):
            if series[i] is not None and series[i + 1] is not None:
                pdf.line(X(t[i]), Y(series[i]), X(t[i + 1]), Y(series[i + 1]))
    pdf.set_line_width(0.2)
    pdf.set_draw_color(0)


def build_pdf(*, title: str, created_at: float, intake: dict, report: dict, track: PoseTrack | None, comments: list[dict] | None = None, max_plots: int = 6) -> bytes:
    r = report
    pdf = _Doc(_t(title))
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 8, _t(title), new_x="LMARGIN", new_y="NEXT")
    pdf.para(f"Recorded session of {time.strftime('%d %b %Y', time.gmtime(created_at))} - report generated {r['generated_at']} - exported {time.strftime('%d %b %Y', time.gmtime())}", 8, color=90)
    if r["meta"].get("demo"):
        pdf.para("DEMO DATA: a computer-drawn stick figure with scripted movements. Nothing here was measured from a real player.", 9, "B")
    pdf.ln(1)
    pdf.para("Not clinically validated. " + r["disclaimer"], 8.5, "B")
    pdf.para(r["validation_status"], 8, color=60)

    pdf.hd("1. Summary")
    pdf.para(r["summary"]["headline"] + " " + r["summary"]["detail"])
    for i in r["quality"]["issues"]:
        pdf.para(f"- {i['message']} {i['suggestion']}", 8, color=60)

    pdf.hd("2. Method and tracking quality")
    tq, m = r["quality"]["tracking"], r["meta"]
    for k, v in (
        ("Video", f"{m['width']}x{m['height']}, {m['fps']:.0f} fps, {m['duration_s']:.1f} s"),
        ("Pose model", "MediaPipe Pose Landmarker, 33 landmarks, one camera" + (f". {r['pose_note']}" if r.get("pose_note") else "")),
        ("Player tracked", f"{tq['tracked_frac']:.0%} of frames; player height {tq['player_height_frac']:.0%} of the frame; knee/ankle visibility {tq['leg_visibility']}"),
        ("Camera view", f"{tq['camera_view']} (from {tq.get('camera_view_basis', 'body orientation')})"),
        ("Stroke timing", (r.get("ball") or {}).get("note", "")),
        ("Court", "Located from court lines." if (r.get("court") or {}).get("status") == "ok" else "Unable to assess position on court. " + (r.get("court") or {}).get("reason", "")),
        ("2D angles", "Measured in the image plane; they change with camera position, and a limb pointing at the camera looks straighter than it is."),
        ("3D estimates", "From the pose model's single-camera body prior. Not motion capture; error in tennis postures is unknown."),
    ):
        pdf.field(k, v)

    pdf.hd("3. Self-reported background (not verified; not used by the video analysis)")
    d = intake.get("discomfort") or {}
    for k, v in (
        ("Player", f"{intake.get('handedness', '?')}-handed, {str(intake.get('skill_level', '')).replace('_', ' ')}, {str(intake.get('backhand_style', '')).replace('_', ' ')} backhand"),
        ("Playing load", f"{str(intake.get('playing_frequency') or 'not given').replace('_', ' ')}; recent change: {intake.get('workload_change') or 'not given'}"),
        ("Discomfort", ("Yes - " + (", ".join(x.replace("_", " ") for x in d.get("regions", [])) or "area not given") + (f'. "{d.get("notes")}"' if d.get("notes") else "")) if d.get("current") else "None reported"),
        ("Injury history", intake.get("injury_history") or "Not given"),
    ):
        pdf.field(k, v)
    for n in r.get("context_notes", []):
        pdf.ln(1)
        pdf.para(n["title"], 8.5, "B")
        pdf.para(n["body"], 8.5)
        pdf.para(n["review_note"], 7.5, "I", 90)

    pdf.hd("4. Findings")
    groups = (("Observed patterns", "observed"), ("Could not be assessed in this recording", "unable_to_assess"), ("Checked, nothing notable", "not_observed"))
    for head, status in groups:
        fs = [f for f in r["findings"] if f["status"] == status]
        if not fs:
            continue
        pdf.para(head, 10, "B")
        for f in fs:
            if pdf.get_y() > 250:
                pdf.add_page()
            pdf.ln(1)
            pdf.para(f"{f['title']}  [{f['stroke_type']} / {f['body_region'].replace('_', ' ')} / {f['category'].replace('_', ' ')}]", 9, "B")
            pdf.field("Observation", f["observation"] + (f" ({f['support']['fired']} of {f['support']['assessed']} assessable; {f['support']['total']} detected.)" if status == "observed" else ""))
            if f.get("occurrences"):
                pdf.field("At", ", ".join(f"{o['time'] if o['time'] is not None else o['start']:.1f} s" for o in f["occurrences"]))
            if f.get("interpretation"):
                pdf.field("Interpretation", "Possible interpretation, not a finding about this person's body. " + f["interpretation"] + (" " + f["user_context"] if f.get("user_context") else ""))
            if f.get("evidence"):
                pdf.field("Evidence", f["evidence"]["strength"] + ".")
                for ref in f["evidence"]["refs"]:
                    pdf.set_x(46)
                    pdf.para(f"{ref['citation']} doi:{ref['doi']}. Studied: {ref['population']} Found: {ref['summary']} Limits: {ref['limits']} [{ref['review_status']}]", 7.5, color=50)
            a = f["action"]
            pdf.field("Action", " ".join(x for x in (a.get("cue") and f"Cue: {a['cue']}", a.get("recording") and f"Recording: {a['recording']}" if status != "not_observed" else None, a.get("professional")) if x))
            if f.get("drills") and status == "observed":
                pdf.field("Drills", "; ".join(f"{dr['name']} [{dr['review_status']}]" for dr in f["drills"]))
            if f.get("limitations") and status != "not_observed":
                pdf.field("Limits", " ".join(f["limitations"]))
            c = f.get("confidence")
            if c:
                pdf.field("Detection conf.", f"{c['level']} ({c['score']}): " + "; ".join(f"{x['name']} {x['value']}" for x in c["factors"]) + ". How clearly it was seen, not a probability of injury.")

    pdf.add_page()
    pdf.hd("5. Strokes")
    get = lambda s, k: next((x for x in s["measurements"] if x["key"] == k), None)  # noqa: E731
    val = lambda x, unit="": "n/a" if not x or not x["assessable"] else f"{x['value']:.0f}{unit}" if x["unit"] == "deg" else f"{x['value']:.1f}"  # noqa: E731
    rows = []
    for s in r["strokes"]:
        src = {"ball_contact": "ball contact", "peak_hand_speed": "peak hand speed", "highest_hand": "highest hand"}.get((s.get("timing") or {}).get("source"), "")
        rows.append([f"{s['peak']:.2f}", s["type"], src, "manual" if s["source"] == "manual" else f"{s['label_confidence']:.2f}", val(get(s, "peak_hand_speed")),
                     f"{val(get(s, 'knee_bend_dom'))}/{val(get(s, 'knee_bend_non'))}", val(get(s, "trunk_lean")), val(get(s, "elbow_angle_peak")), val(get(s, "contact_height")),
                     f"{s['quality']['tracked_frac']:.0%}"])  # fmt: skip
    pdf.table(["Time s", "Stroke", "Timed at", "Label conf.", "Hand speed", "Knee bend", "Trunk tilt", "Elbow", "Contact ht", "Tracked"], rows, [14, 22, 26, 18, 18, 20, 18, 14, 16, 14])
    pdf.para("Hand speed: torso lengths/s, image plane. Knee bend: hitting-arm side / other leg, degrees, larger of 2D and 3D estimate. Trunk tilt: degrees from vertical, 2D. Elbow: degrees, single-camera 3D estimate. "
          "Contact height: torso lengths above the hips, 2D, only when ball contact was found.", 7.5, color=60)  # fmt: skip
    if r.get("stroke_summary"):
        pdf.hd("Across strokes", 10)
        rows = [[st, mm["label"], f"{mm['median']}", f"{mm['min']} - {mm['max']}", str(mm["n"]), METHOD.get(mm["method"], mm["method"])] for st, v in r["stroke_summary"].items() for mm in v["metrics"].values()]
        pdf.table(["Stroke", "Measurement", "Median", "Range", "n", "Method"], rows, [20, 78, 16, 24, 8, 34])

    if track is not None and r["strokes"]:
        shown = [s for s in r["strokes"] if s["quality"]["assessable"]][:max_plots]
        for s in shown:
            pdf.add_page()
            data = angles.stroke_series(track, s["start"], s["end"])
            pdf.hd(f"6. Joint angles over time - {s['type']} at {s['peak']:.2f} s")
            pdf.para(f"Green = 2D image-plane angle. Orange = single-camera 3D estimate. Grey bands and breaks in a line = the joints were below the visibility cut-off ({data['min_visibility']}); nothing is drawn rather than a guess. "
                  f"Dashed line = stroke timestamp ({(s.get('timing') or {}).get('label', 'pose-based')}).", 7.5, color=60)  # fmt: skip
            y = pdf.get_y() + 6
            for e in data["series"]:
                _plot(pdf, data, e, 24, y, 160, 34, {"timestamp": s["peak"]})
                y += 46
        if len([s for s in r["strokes"] if s["quality"]["assessable"]]) > max_plots:
            pdf.para(f"Plots are included for the first {max_plots} assessable strokes; the app shows any stroke on request.", 8, "I")

    pdf.add_page()
    pdf.hd("7. Not assessed")
    for n in r["not_assessed"]:
        pdf.para(f"- {n['topic']}. {n['reason']}", 8.5)
    if comments:
        pdf.hd("8. Comments pinned to the video")
        for c in sorted(comments, key=lambda c: c["t"]):
            pdf.para(f"{c['t']:.1f} s - {c.get('author') or 'unsigned'}: {c['body']}", 8.5)
    return bytes(pdf.output())
