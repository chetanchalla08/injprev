import { useState } from 'react'
import { AnglePlot } from './Tools'
import { cap, fmtTime } from '../api'
import type { Finding, Measurement, Ref, Session, Stroke } from '../api'

type View = 'player' | 'coach' | 'clinician'
type Edit = { id?: string; start: number; end: number; type: string }
interface Props { session: Session; view: View; activeStroke: string | null; onShow: (start: number, end: number, strokeId?: string) => void; getTime: () => number; onEdit: (s: Edit[]) => Promise<unknown>; readOnly?: boolean }

const TYPES = ['forehand', 'backhand', 'serve', 'unclassified']
const METHOD: Record<string, string> = {
  image_plane: '2D image angle', estimated_3d: 'single-camera 3D estimate', relative: 'relative to body size', 'image_plane+estimated_3d': 'larger of 2D and 3D estimate',
}
const CONF_TEXT = { high: 'Seen clearly', moderate: 'Seen fairly clearly', low: 'Seen, but tracking was weak', insufficient: 'Not clear enough' }
const plural = (n: number, w: string) => `${n} ${w}${n === 1 ? '' : 's'}`

function Refs({ refs }: { refs: Ref[] }) {
  return (
    <ul className="refs">
      {refs.map((r) => (
        <li key={r.id}>
          <p>{r.citation} <a href={`https://doi.org/${r.doi}`} target="_blank" rel="noreferrer">doi:{r.doi}</a></p>
          <p className="muted small"><strong>Who was studied:</strong> {r.population}</p>
          <p className="muted small"><strong>What it found:</strong> {r.summary}</p>
          <p className="muted small"><strong>Limits:</strong> {r.limits}</p>
          <p className="muted small"><em>{r.review_status}.</em></p>
        </li>
      ))}
    </ul>
  )
}

function ConfidenceBadge({ f, detail }: { f: Finding; detail: boolean }) {
  if (!f.confidence) return null
  const c = f.confidence
  return (
    <details className="conf" open={detail && undefined}>
      <summary><span className={`meter m-${c.level}`} aria-hidden><i /><i /><i /></span> Detection confidence: {c.level} <span className="muted">— {CONF_TEXT[c.level].toLowerCase()}</span></summary>
      <p className="muted small">{c.explanation}</p>
      <table className="mini">
        <tbody>{c.factors.map((x) => <tr key={x.name}><th>{x.name}</th><td>{x.value}</td><td className="muted">{x.note}</td></tr>)}</tbody>
      </table>
    </details>
  )
}

function FindingCard({ f, view, onShow }: { f: Finding; view: View; onShow: Props['onShow'] }) {
  const health = f.category === 'health_related' && f.status === 'observed'
  const full = view !== 'player'
  return (
    <article className={`finding s-${f.status}`}>
      <header>
        <div className="tags">
          <span className={`tag k-${f.stroke_type}`}>{cap(f.stroke_type)}</span>
          <span className="tag">{cap(f.body_region)}</span>
          {f.status === 'observed' && <span className="tag soft">{f.recurring ? 'Recurring' : f.support.fired === 1 ? 'Seen once' : 'Seen sometimes'} · {f.support.fired} of {f.support.assessed}</span>}
          {f.status === 'unable_to_assess' && <span className="tag hollow">Unable to assess</span>}
          {f.status === 'not_observed' && <span className="tag hollow">Checked · not seen</span>}
        </div>
        <h4>{f.title}</h4>
      </header>
      <dl>
        <dt>Observed</dt>
        <dd>{f.observation}{f.status === 'observed' && full && f.support.assessed < f.support.total ? ` (${f.support.total - f.support.assessed} further ${f.stroke_type}${f.support.total - f.support.assessed === 1 ? '' : 's'} could not be assessed.)` : ''}</dd>
        {f.occurrences.length > 0 && (
          <>
            <dt>Video</dt>
            <dd className="occ no-print">{f.occurrences.map((o) => <button key={o.stroke_id} className="chip play" onClick={() => onShow(o.start, o.end, o.stroke_id)}>▶ {fmtTime(o.time ?? o.start)}</button>)}</dd>
          </>
        )}
        {f.action.cue && <><dt>Try</dt><dd className="cue">{f.action.cue}</dd></>}
        {f.interpretation && (
          <>
            <dt>What it might mean</dt>
            <dd>
              {health && <span className="tag hollow">Possible interpretation, not a finding about your body</span>} {f.interpretation}
              {f.user_context && <p className="note">{f.user_context}</p>}
            </dd>
          </>
        )}
        {f.evidence && (
          <>
            <dt>Evidence</dt>
            <dd>
              <strong>{f.evidence.strength}.</strong>{' '}
              {full ? <Refs refs={f.evidence.refs} /> : <details><summary>{plural(f.evidence.refs.length, 'reference')}</summary><Refs refs={f.evidence.refs} /></details>}
            </dd>
          </>
        )}
        {f.status === 'observed' && (f.drills?.length ?? 0) > 0 && (
          <>
            <dt>Drills</dt>
            <dd>{f.drills!.map((d) => (
              <details key={d.id}><summary>{d.name} <span className="tag hollow">{d.review_status}</span></summary>
                <p className="muted small">{d.focus}</p><ol>{d.steps.map((x) => <li key={x}>{x}</li>)}</ol>{d.caution && <p className="muted small">{d.caution}</p>}</details>
            ))}</dd>
          </>
        )}
        {f.action.recording && (f.status !== 'not_observed') && <><dt>Recording tip</dt><dd>{f.action.recording}</dd></>}
        {f.action.professional && <><dt>Further review</dt><dd>{f.action.professional}</dd></>}
        {(full || f.status === 'observed') && f.limitations.length > 0 && (
          <>
            <dt>Limits</dt>
            <dd>{full ? <ul>{f.limitations.map((l) => <li key={l}>{l}</li>)}</ul> : <details><summary>What this can't tell you</summary><ul>{f.limitations.map((l) => <li key={l}>{l}</li>)}</ul></details>}</dd>
          </>
        )}
      </dl>
      <ConfidenceBadge f={f} detail={view === 'clinician'} />
    </article>
  )
}

function mval(m: Measurement, both: boolean) {
  if (!m.assessable) return <span className="muted" title={m.reason}>n/a</span>
  const unit = m.unit === 'deg' ? '°' : ''
  const v = m.unit === 'deg' ? Math.round(m.value!) : m.value!.toFixed(1)
  const ex = m.extra
  return <>{v}{unit}{both && ex.image_plane != null && ex.estimated_3d != null && <span className="muted small"> (2D {ex.image_plane}° · 3D est. {ex.estimated_3d}°)</span>}</>
}

function StrokeEditor({ strokes, getTime, onEdit, onShow, activeStroke, clinician, readOnly, sessionId }: { strokes: Stroke[]; getTime: () => number; onEdit: Props['onEdit']; onShow: Props['onShow']; activeStroke: string | null; clinician: boolean; readOnly?: boolean; sessionId: string }) {
  const [plot, setPlot] = useState<string | null>(null)
  const [draft, setDraft] = useState<{ start: number | null; end: number | null; type: string }>({ start: null, end: null, type: 'forehand' })
  const [busy, setBusy] = useState(false)
  const [open, setOpen] = useState<string | null>(null)
  const base = (): Edit[] => strokes.map((s) => ({ id: s.id, start: s.start, end: s.end, type: s.type }))
  const send = (list: Edit[]) => { setBusy(true); onEdit(list).finally(() => setBusy(false)) }
  const get = (s: Stroke, k: string) => s.measurements.find((m) => m.key === k)
  const valid = draft.start !== null && draft.end !== null && draft.end - draft.start >= 0.2 && draft.end - draft.start <= 6
  return (
    <section className="card">
      <h3>Strokes</h3>
      <p className="muted small">Labels are automatic and can be wrong. Fix a label, remove a false detection or add a missed stroke, and the measurements and findings are recalculated from the saved pose data.</p>
      {strokes.length > 0 && (
        <div className="scroll">
          <table className="strokes">
            <thead><tr><th>Time</th><th>Timed at</th><th>Stroke</th><th>Label confidence</th><th>Hand speed</th><th>Knee bend</th><th>Trunk tilt</th><th /></tr></thead>
            <tbody>
              {strokes.map((s) => {
                const kd = get(s, 'knee_bend_dom'), kn = get(s, 'knee_bend_non'), sp = get(s, 'peak_hand_speed'), tl = get(s, 'trunk_lean')
                return [
                  <tr key={s.id} className={activeStroke === s.id ? 'active' : ''}>
                    <td><button className="chip play no-print" onClick={() => onShow(s.start, s.end, s.id)}>▶ {fmtTime(s.peak)}</button><span className="print-only">{fmtTime(s.peak)}</span></td>
                    <td className="muted small" title={s.timing?.label}>{s.timing?.source === 'ball_contact' ? 'ball contact' : s.timing?.source === 'highest_hand' ? 'highest hand' : 'peak hand speed'}</td>
                    <td>
                      <select value={s.type} disabled={busy || readOnly} aria-label="Stroke type" onChange={(e) => send(base().map((x) => (x.id === s.id ? { ...x, type: e.target.value } : x)))}>
                        {TYPES.map((t) => <option key={t} value={t}>{t === 'unclassified' ? 'Unlabelled' : cap(t)}</option>)}
                      </select>
                    </td>
                    <td title={s.label_basis}>{s.source === 'manual' ? 'Set by you' : s.label_confidence >= 0.7 ? 'High' : s.label_confidence >= 0.5 ? 'Moderate' : 'Low'}</td>
                    {s.quality.assessable ? (
                      <>
                        <td>{sp ? mval(sp, false) : '–'}</td>
                        <td>{kd && kn ? <>{mval(kd, false)} / {mval(kn, false)}</> : '–'}</td>
                        <td>{tl ? mval(tl, false) : '–'}</td>
                      </>
                    ) : <td colSpan={3} className="muted">{s.quality.reason}</td>}
                    <td className="no-print">
                      <button className="link" onClick={() => setOpen(open === s.id ? null : s.id)}>{open === s.id ? 'Less' : 'Detail'}</button>{' '}
                      {clinician && !readOnly && s.quality.assessable && <><button className="link" onClick={() => setPlot(plot === s.id ? null : s.id)}>{plot === s.id ? 'Hide plot' : 'Angles'}</button>{' '}</>}
                      {!readOnly && <button className="icon" disabled={busy} aria-label="Remove stroke" title="Remove" onClick={() => send(base().filter((x) => x.id !== s.id))}>✕</button>}
                    </td>
                  </tr>,
                  open === s.id && (
                    <tr key={s.id + 'd'} className="detail">
                      <td colSpan={8}>
                        <p className="muted small"><strong>Timestamp:</strong> {s.timing?.label}{s.timing?.basis ? ` (${s.timing.basis}; unvalidated)` : ''}.
                          {s.phases && <> <strong>Phases:</strong> preparation {fmtTime(s.phases.preparation[0])}–{fmtTime(s.phases.preparation[1])}, forward swing {s.phases.forward_swing_s}s to the {s.phases.boundary}, follow-through to {fmtTime(s.phases.follow_through[1])}. {s.phases.note}</>}
                          {s.movement && <> <strong>On court (unvalidated):</strong> {s.movement.zone}{s.movement.recovery ? `; ${s.movement.recovery.after_s}s later ${s.movement.recovery.zone}, having moved ${s.movement.recovery.moved_m} m` : ''}.</>}</p>
                        <p className="muted small">{s.label_basis} Tracked in {Math.round(s.quality.tracked_frac * 100)}% of frames; arm visibility {s.quality.arm_visibility}, leg visibility {s.quality.leg_visibility}.</p>
                        <table className="mini">
                          <thead><tr><th>Measurement</th><th>Value</th><th>Method</th>{clinician && <th>Visibility</th>}</tr></thead>
                          <tbody>
                            {s.measurements.map((m) => (
                              <tr key={m.key}><th>{m.label}</th><td>{m.assessable ? <>{mval(m, true)} {m.unit !== 'deg' && <span className="muted small">{m.unit}</span>}</> : <span className="muted">Unable to assess — {m.reason}</span>}</td>
                                <td className="muted">{METHOD[m.method] || m.method}</td>{clinician && <td className="muted">{m.visibility ?? '–'}</td>}</tr>
                            ))}
                          </tbody>
                        </table>
                      </td>
                    </tr>
                  ),
                  plot === s.id && <tr key={s.id + 'p'} className="detail"><td colSpan={8}><AnglePlot id={sessionId} strokeId={s.id} /></td></tr>,
                ]
              })}
            </tbody>
          </table>
        </div>
      )}
      <p className="muted small">Hand speed is in torso lengths per second (image plane). Knee bend is hitting-arm side leg / other leg, 0° = straight.</p>
      {!readOnly && <div className="add no-print">
        <strong>Add a stroke:</strong> pause the video at the start of the swing preparation and at the end of the follow-through.
        <div className="row">
          <button className="btn ghost" onClick={() => setDraft({ ...draft, start: getTime() })}>Set start {draft.start !== null && `(${fmtTime(draft.start)})`}</button>
          <button className="btn ghost" onClick={() => setDraft({ ...draft, end: getTime() })}>Set end {draft.end !== null && `(${fmtTime(draft.end)})`}</button>
          <select value={draft.type} onChange={(e) => setDraft({ ...draft, type: e.target.value })} aria-label="New stroke type">{TYPES.slice(0, 3).map((t) => <option key={t} value={t}>{cap(t)}</option>)}</select>
          <button className="btn" disabled={!valid || busy} onClick={() => { send([...base(), { start: draft.start!, end: draft.end!, type: draft.type }]); setDraft({ start: null, end: null, type: draft.type }) }}>Add</button>
        </div>
        {draft.start !== null && draft.end !== null && !valid && <p className="muted small">A stroke needs to be between 0.2 and 6 seconds, with the end after the start.</p>}
      </div>}
    </section>
  )
}

export default function ReportView({ session, view, activeStroke, onShow, getTime, onEdit, readOnly }: Props) {
  const r = session.report!
  const intake = session.intake
  const observed = r.findings.filter((f) => f.status === 'observed')
  const unable = r.findings.filter((f) => f.status === 'unable_to_assess')
  const clear = r.findings.filter((f) => f.status === 'not_observed')
  const byId = Object.fromEntries(r.findings.map((f) => [f.id, f]))
  const groups = (list: Finding[], key: (f: Finding) => string) => {
    const m = new Map<string, Finding[]>()
    list.forEach((f) => m.set(key(f), [...(m.get(key(f)) || []), f]))
    return [...m.entries()]
  }
  const tq = r.quality.tracking

  return (
    <div className={`report v-${view}`}>
      {r.meta.demo && <div className="banner"><strong>Demo data.</strong> This session is a computer-drawn stick figure with scripted movements, included only to show how the report works. Nothing here was measured from a real player.</div>}
      {view === 'clinician' && (
        <div className="banner">
          <strong>Not clinically validated.</strong> This view shows more detail, not more certainty. Angles are from one uncalibrated consumer video using a general-purpose pose model; thresholds are developer heuristics; evidence summaries have not been reviewed by a clinician. Use it as a prompt for your own assessment, not as a measurement instrument.
        </div>
      )}

      <section className="card summary">
        <h2>{r.summary.headline}</h2>
        <p>{r.summary.detail}</p>
        {r.quality.issues.length > 0 && (
          <div className="quality">
            <h4>Recording limitations</h4>
            <ul>{r.quality.issues.map((i) => <li key={i.code}>{i.message} <span className="muted">{i.suggestion}</span></li>)}</ul>
          </div>
        )}
        {r.ball && r.ball.note && <p className="muted small"><strong>Stroke timing:</strong> {r.ball.note}</p>}
        <p className="muted small">{r.disclaimer}</p>
      </section>

      {r.context_notes.map((n) => (
        <section key={n.kind} className="card note-card">
          <h3>{n.title}</h3>
          <p>{n.body}</p>
          {n.refs.length > 0 && <details><summary>References</summary><Refs refs={n.refs} /></details>}
          {view !== 'player' && <p className="muted small">{n.review_note}</p>}
        </section>
      ))}

      {view === 'clinician' && (
        <section className="card">
          <h3>Self-reported background</h3>
          <table className="mini">
            <tbody>
              <tr><th>Hand / level / backhand</th><td>{cap(intake.handedness)} · {cap(intake.skill_level)} · {cap(intake.backhand_style)}</td></tr>
              <tr><th>Playing frequency</th><td>{intake.playing_frequency ? cap(intake.playing_frequency) : 'Not given'}</td></tr>
              <tr><th>Recent change in load</th><td>{intake.workload_change ? cap(intake.workload_change) : 'Not given'}</td></tr>
              <tr><th>Current discomfort</th><td>{intake.discomfort.current ? `Yes — ${intake.discomfort.regions.map(cap).join(', ') || 'area not given'}${intake.discomfort.notes ? `. “${intake.discomfort.notes}”` : ''}` : 'None reported'}</td></tr>
              <tr><th>Injury history</th><td>{intake.injury_history || 'Not given'}</td></tr>
            </tbody>
          </table>
          <p className="muted small">Entered by the player; not verified and not used by the video analysis.</p>
        </section>
      )}

      {view === 'player' && !r.summary.no_strokes && (
        <section className="card">
          <h3>What to work on first</h3>
          {r.priorities.length === 0 ? (
            <p className="muted">Nothing stood out in the checks this recording allowed. That is not the same as a clean bill of health for your technique: see what couldn't be checked below.</p>
          ) : (
            <ol className="priorities">
              {r.priorities.map((p) => {
                const f = byId[p.finding_id]
                return (
                  <li key={p.finding_id}>
                    <span className={`tag k-${p.stroke_type}`}>{cap(p.stroke_type)}</span>
                    <strong>{p.title}</strong>
                    <p className="cue">{p.cue}</p>
                    {f?.occurrences[0] && <button className="chip play no-print" onClick={() => onShow(f.occurrences[0].start, f.occurrences[0].end, f.occurrences[0].stroke_id)}>▶ Show me</button>}
                  </li>
                )
              })}
            </ol>
          )}
        </section>
      )}

      {view !== 'player' && <StrokeEditor strokes={r.strokes} getTime={getTime} onEdit={onEdit} onShow={onShow} activeStroke={activeStroke} clinician={view === 'clinician'} readOnly={readOnly} sessionId={session.id} />}
      {view === 'player' && r.summary.no_strokes && (
        <section className="card"><h3>No strokes found</h3><p>Either there were no forehands, backhands or serves in the clip, or the hitting arm wasn't visible enough to detect them. Switch to the <strong>Coach</strong> tab to mark strokes yourself, or record again using the filming tips.</p></section>
      )}

      {observed.length > 0 && (
        <section>
          <h3 className="sec">{view === 'player' ? 'What we saw' : 'Observed patterns'}</h3>
          {groups(observed, (f) => (view === 'clinician' ? cap(f.body_region) : cap(f.stroke_type))).map(([g, list]) => (
            <div key={g} className="group">
              <h4 className="group-h">{g}</h4>
              {list.map((f) => <FindingCard key={f.id} f={f} view={view} onShow={onShow} />)}
            </div>
          ))}
        </section>
      )}

      {view !== 'player' && Object.keys(r.stroke_summary).length > 0 && (
        <section className="card">
          <h3>Across strokes</h3>
          <div className="scroll">
            <table className="mini wide-t">
              <thead><tr><th>Stroke</th><th>Measurement</th><th>Median</th><th>Range</th><th>n</th><th>Method</th></tr></thead>
              <tbody>
                {Object.entries(r.stroke_summary).flatMap(([st, v]) => Object.entries(v.metrics).map(([k, m], i) => (
                  <tr key={st + k}><th>{i === 0 ? `${cap(st)} (${v.count})` : ''}</th><td>{m.label}</td><td>{m.median}{m.unit === 'deg' ? '°' : ''}</td><td>{m.min}–{m.max}</td><td>{m.n}</td><td className="muted">{METHOD[m.method] || m.method}</td></tr>
                )))}
              </tbody>
            </table>
          </div>
          <p className="muted small">A wide range across the same stroke can reflect inconsistency, different incoming balls, or tracking noise. Values are specific to this camera position.</p>
        </section>
      )}

      {(unable.length > 0 || view !== 'player') && (
        <section>
          <h3 className="sec">Couldn't be assessed in this recording</h3>
          {unable.map((f) => <FindingCard key={f.id} f={f} view={view} onShow={onShow} />)}
          <div className="card">
            <h4>Not assessed by this version at all</h4>
            <ul className="plain">{r.not_assessed.map((n) => <li key={n.topic}><strong>{n.topic}.</strong> <span className="muted">{n.reason}</span></li>)}</ul>
          </div>
        </section>
      )}

      {clear.length > 0 && (
        view === 'player' ? (
          <details className="card"><summary>{plural(clear.length, 'check')} where nothing notable was seen</summary>{clear.map((f) => <p key={f.id}><span className={`tag k-${f.stroke_type}`}>{cap(f.stroke_type)}</span> {f.observation}</p>)}</details>
        ) : (
          <section><h3 className="sec">Checked, nothing notable</h3>{clear.map((f) => <FindingCard key={f.id} f={f} view={view} onShow={onShow} />)}</section>
        )
      )}

      {view === 'clinician' && (
        <section className="card">
          <h3>Method and tracking quality</h3>
          <table className="mini">
            <tbody>
              <tr><th>Pose model</th><td>MediaPipe Pose Landmarker, 33 landmarks, single camera</td></tr>
              <tr><th>Frames with player tracked</th><td>{Math.round(tq.tracked_frac * 100)}%</td></tr>
              <tr><th>Player height in frame</th><td>{Math.round(tq.player_height_frac * 100)}% of image height</td></tr>
              <tr><th>Knee/ankle visibility</th><td>{tq.leg_visibility} (model's own 0–1 estimate)</td></tr>
              <tr><th>Camera view</th><td>{cap(tq.camera_view)}{tq.pelvis_azimuth_median !== null ? ` (median hip-line angle to image plane ${tq.pelvis_azimuth_median}°)` : ''}</td></tr>
              <tr><th>Camera view judged from</th><td>{tq.camera_view_basis || 'body orientation'}</td></tr>
              <tr><th>Ball tracking</th><td>{r.ball ? (r.ball.status === 'off' ? 'Off' : `Contact found for ${r.ball.found} of ${r.ball.strokes} strokes (unvalidated)`) : 'Not run'}</td></tr>
              <tr><th>Court position</th><td>{r.court?.status === 'ok' ? `Court located from lines (${(r.court.lines_supported || []).length} lines); positions are unvalidated` : `Unable to assess. ${r.court?.reason || ''}`}</td></tr>
              {r.hops && <tr><th>Hops before strokes</th><td>{r.hops.times.length} found (unvalidated). <span className="muted">{r.hops.note}</span></td></tr>}
              {r.movement && <tr><th>Movement</th><td>About {r.movement.distance_m} m covered, {r.movement.direction_changes.length} sideways changes of direction. <span className="muted">{r.movement.note}</span></td></tr>}
              {r.tracking_events && r.tracking_events.cuts.length > 0 && <tr><th>Scene cuts</th><td>{r.tracking_events.cuts.map((c) => `${c}s`).join(', ')}</td></tr>}
              {r.pose_note && <tr><th>Pose note</th><td>{r.pose_note}</td></tr>}
              <tr><th>Video</th><td>{r.meta.width}×{r.meta.height}, {r.meta.fps} fps, brightness {String(r.quality.video.metrics.mean_brightness)}, sharpness {String(r.quality.video.metrics.sharpness)}</td></tr>
            </tbody>
          </table>
          <ul className="plain small muted">
            <li><strong>2D image angles</strong> are projections and change with camera position; a limb pointing toward the camera looks straighter than it is.</li>
            <li><strong>Single-camera 3D estimates</strong> come from the pose model's learned body prior. They are not motion capture and their error in tennis postures is unknown.</li>
            <li><strong>Timestamps</strong> mark ball contact only where the table says so; otherwise peak hitting-hand speed (serves: highest hand), which is near contact but is not contact.</li>
            <li>{r.validation_status}</li>
          </ul>
        </section>
      )}
    </div>
  )
}
