import { useEffect, useState } from 'react'
import { api, cap, fmtDate, fmtTime } from '../api'
import type { AnnotationSet, Comment, Config, Drawing, Finding, LabelSet, LabelStroke, Profile, RuleReview, Segment, Session, Share, StrokeSeries, Summary } from '../api'

const err = (set: (s: string) => void) => (e: unknown) => set((e as Error).message)
export const JOINTS = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
const VIEWS = ['side', 'behind', 'front', 'oblique', 'far', 'unknown']
const ROLES: [string, string][] = [['coach', 'Coach'], ['physiotherapist', 'Physiotherapist'], ['sports_physician', 'Sports physician'], ['other_clinician', 'Other clinician']]

/* ---- re-acquisition: never analyse a stretch that may be a different person until the user says so ---- */
export function ConfirmPlayer({ segments, onShow, onDecide }: { segments: Segment[]; onShow: (s: Segment | null) => void; onDecide: (id: number, d: 'confirm' | 'reject') => Promise<unknown> }) {
  const [busy, setBusy] = useState(false)
  const asking = segments.filter((s) => s.status === 'needs_confirmation')
  if (!asking.length) return null
  return (
    <section className="card note-card no-print">
      <h3>Check the player</h3>
      <p className="muted small">After a cut, or after the player left the picture, someone who may be the same player was found. These parts of the video are left out of the analysis unless you confirm. If you are not sure, reject.</p>
      <ul className="plain">
        {asking.map((s) => (
          <li key={s.id} className="row">
            <button className="chip play" onClick={() => onShow(s)}>▶ {fmtTime(s.start)}–{fmtTime(s.end)}</button>
            <span className="muted small">after {s.after === 'cut' ? 'a scene cut' : 'leaving the frame'} · {s.similarity === null ? 'clothing could not be compared' : `clothing match ${Math.round(s.similarity * 100)}%`}</span>
            <span className="spacer" />
            {s.decision ? <span className="tag hollow">{cap(s.decision)}</span> : <span className="tag st-failed">Waiting for you</span>}
            <button className="btn" disabled={busy || s.decision === 'confirmed'} onClick={() => { setBusy(true); onDecide(s.id, 'confirm').finally(() => { setBusy(false); onShow(null) }) }}>Same player</button>
            <button className="btn ghost" disabled={busy || s.decision === 'rejected'} onClick={() => { setBusy(true); onDecide(s.id, 'reject').finally(() => { setBusy(false); onShow(null) }) }}>Different person</button>
          </li>
        ))}
      </ul>
    </section>
  )
}

/* ---- coach: comments, drawings, profile, share links ---- */
export function CoachPanel({ s, cfg, comments, drawings, getTime, seek, reload }: { s: Session; cfg: Config | null; comments: Comment[]; drawings: Drawing[]; getTime: () => number; seek: (t: number) => void; reload: () => void }) {
  const [body, setBody] = useState('')
  const [author, setAuthor] = useState(localStorage.getItem('author') || '')
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [shares, setShares] = useState<Share[]>([])
  const [fresh, setFresh] = useState<Share | null>(null)
  const [days, setDays] = useState(cfg?.share_ttl_days ?? 7)
  const [error, setError] = useState('')
  const loadShares = () => api.shares(s.id).then(setShares).catch(() => {})
  useEffect(() => { api.profiles().then(setProfiles).catch(() => {}); loadShares() }, [s.id]) // eslint-disable-line
  const add = () => { localStorage.setItem('author', author); api.addComment(s.id, { t: getTime(), body, author }).then(() => { setBody(''); reload() }).catch(err(setError)) }
  const link = fresh ? `${window.location.origin}${fresh.path}` : ''
  return (
    <section className="card no-print">
      <h3>Coach notes</h3>
      {error && <p className="alert">{error}</p>}
      <label>Player profile
        <select value={s.profile_id || ''} onChange={(e) => api.setProfile(s.id, e.target.value || null).then(reload).catch(err(setError))}>
          <option value="">Not assigned</option>
          {profiles.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
        </select>
      </label>
      <p className="muted small">Profiles live in this browser only (there are no accounts). Create them on the Sessions page.</p>
      <h4>Comments pinned to the video</h4>
      <ul className="plain">
        {comments.map((c) => (
          <li key={c.id} className="row"><button className="chip play" onClick={() => seek(c.t)}>▶ {fmtTime(c.t)}</button><span>{c.author && <strong>{c.author}: </strong>}{c.body}</span><span className="spacer" />
            <button className="icon" aria-label="Delete comment" onClick={() => api.delComment(s.id, c.id).then(reload)}>✕</button></li>
        ))}
        {drawings.map((d) => (
          <li key={d.id} className="row"><button className="chip play" onClick={() => seek(d.t)}>✎ {fmtTime(d.t)}</button><span className="muted">Drawing ({d.shapes.length} {d.shapes.length === 1 ? 'shape' : 'shapes'}){d.note ? ` — ${d.note}` : ''}</span><span className="spacer" />
            <button className="icon" aria-label="Delete drawing" onClick={() => api.delDrawing(s.id, d.id).then(reload)}>✕</button></li>
        ))}
      </ul>
      <div className="row">
        <input className="inline" placeholder="Your name" maxLength={60} value={author} onChange={(e) => setAuthor(e.target.value)} style={{ maxWidth: 140 }} />
        <input className="inline grow" placeholder="Comment at the current video time" maxLength={1000} value={body} onChange={(e) => setBody(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && body.trim() && add()} />
        <button className="btn" disabled={!body.trim()} onClick={add}>Pin comment</button>
      </div>
      <p className="muted small">To draw, use the Line / Angle / Freehand buttons under the video.</p>
      <h4>Share this session (read-only)</h4>
      <p className="muted small">Anyone with the link can watch the video and read the report, comments and drawings until it expires or you revoke it. They cannot change anything or see your other sessions. Discomfort and injury-history details you entered are never included.</p>
      <div className="row">
        <label className="check">Expires in <select className="inline" value={days} onChange={(e) => setDays(Number(e.target.value))}>{[1, 3, 7, 14, 30].filter((d) => d <= (cfg?.share_max_ttl_days ?? 30)).map((d) => <option key={d} value={d}>{d} day{d > 1 ? 's' : ''}</option>)}</select></label>
        <button className="btn" onClick={() => api.addShare(s.id, days).then((x) => { setFresh(x); loadShares() }).catch(err(setError))}>Create link</button>
      </div>
      {fresh && <div className="note"><strong>Copy this link now.</strong> It is not stored and cannot be shown again.<br /><input className="inline grow" readOnly value={link} onFocus={(e) => e.target.select()} />
        <button className="btn ghost" onClick={() => navigator.clipboard?.writeText(link)}>Copy</button></div>}
      <ul className="plain">
        {shares.map((x) => (
          <li key={x.id} className="row"><span>Link created {fmtDate(x.created_at)}</span><span className="muted small">{x.revoked_at ? 'revoked' : x.active ? `expires ${fmtDate(x.expires_at)}` : 'expired'}</span><span className="spacer" />
            {x.active && <button className="btn ghost" onClick={() => api.revokeShare(s.id, x.id).then(() => { loadShares(); if (fresh?.id === x.id) setFresh(null) })}>Revoke</button>}</li>
        ))}
      </ul>
    </section>
  )
}

/* ---- optional LLM rewording, shown only when the server has it enabled ---- */
export function SummaryBox({ id }: { id: string }) {
  const [s, setS] = useState<Summary | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => { api.summary(id).then(setS).catch(() => {}) }, [id])
  if (!s?.enabled) return null
  return (
    <section className="card">
      <h3>In plain words <span className="tag hollow">AI rewording</span></h3>
      {s.summary ? <>{s.summary.text.split('\n\n').map((p, i) => <p key={i}>{p}</p>)}<p className="muted small">{s.summary.notice}</p></>
        : <p className="muted small">An AI language model can reword the findings below into a few plain paragraphs. It is sent the findings only (no video, no health details), and its text is thrown away if it contains any number or reference that is not in the findings.</p>}
      {error && <p className="alert">{error}</p>}
      <button className="btn no-print" disabled={busy} onClick={() => { setBusy(true); setError(''); api.makeSummary(id).then(setS).catch(err(setError)).finally(() => setBusy(false)) }}>{busy ? 'Writing…' : s.summary ? 'Reword again' : 'Reword the findings'}</button>
    </section>
  )
}

/* ---- clinician: joint angle over time for one stroke ---- */
const MCOL: Record<string, string> = { image_plane: 'var(--green)', estimated_3d: 'var(--clay)' }
export function AnglePlot({ id, strokeId }: { id: string; strokeId: string }) {
  const [d, setD] = useState<StrokeSeries | null>(null)
  const [error, setError] = useState('')
  useEffect(() => { setD(null); api.series(id, strokeId).then(setD).catch(err(setError)) }, [id, strokeId])
  if (error) return <p className="alert">{error}</p>
  if (!d) return <p className="muted small">Loading angles…</p>
  const W = 640, H = 150, P = 34
  const t0 = d.t[0], t1 = d.t[d.t.length - 1] || t0 + 1
  const X = (t: number) => P + ((t - t0) / (t1 - t0 || 1)) * (W - P - 6)
  return (
    <div className="plots">
      <p className="muted small"><span className="key" style={{ background: MCOL.image_plane }} /> 2D image-plane angle <span className="key" style={{ background: MCOL.estimated_3d }} /> single-camera 3D estimate <span className="key grey" /> joints below the visibility cut-off ({d.min_visibility}) · dashed line = stroke timestamp ({d.timing?.source === 'ball_contact' ? 'ball contact' : 'pose-based, not contact'})</p>
      {d.series.map((s) => {
        const vals = Object.values(s.methods).flat().filter((v): v is number => v !== null)
        const lo = Math.min(...vals) - 5, hi = Math.max(...vals) + 5
        const Y = (v: number) => 8 + (1 - (v - lo) / (hi - lo || 1)) * (H - 30)
        const path = (a: (number | null)[]) => a.map((v, i) => (v === null ? '' : `${i === 0 || a[i - 1] === null ? 'M' : 'L'}${X(d.t[i]).toFixed(1)},${Y(v).toFixed(1)}`)).join(' ')
        return (
          <figure key={s.key}>
            <figcaption><strong>{s.label}</strong> <span className="muted small">{s.note} · tracked in {Math.round(s.tracked_frac * 100)}% of frames</span></figcaption>
            <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${s.label} over time`}>
              <rect x={P} y={8} width={W - P - 6} height={H - 30} fill="none" stroke="var(--line)" />
              {s.visibility.map((v, i) => v < d.min_visibility && i + 1 < d.t.length ? <rect key={i} x={X(d.t[i])} y={8} width={Math.max(X(d.t[i + 1]) - X(d.t[i]), 1)} height={H - 30} fill="var(--grey-soft)" /> : null)}
              {vals.length === 0 ? <text x={W / 2} y={H / 2} textAnchor="middle" className="svg-muted">Unable to assess: the joints needed were not tracked in this stroke</text> : <>
                {Object.entries(s.methods).map(([m, a]) => <path key={m} d={path(a)} fill="none" stroke={MCOL[m]} strokeWidth={2} />)}
                <text x={2} y={16} className="svg-muted">{Math.round(hi)}°</text><text x={2} y={H - 22} className="svg-muted">{Math.round(lo)}°</text>
              </>}
              <line x1={X(d.stroke.peak)} x2={X(d.stroke.peak)} y1={8} y2={H - 22} stroke="var(--muted)" strokeDasharray="4 3" />
              <text x={P} y={H - 6} className="svg-muted">{t0.toFixed(2)} s</text><text x={W - 6} y={H - 6} textAnchor="end" className="svg-muted">{t1.toFixed(2)} s</text>
            </svg>
          </figure>
        )
      })}
      <p className="muted small">{d.gap_note} {d.method_notes.image_plane} {d.method_notes.estimated_3d}</p>
    </div>
  )
}

/* ---- evaluation: label strokes for the test set ---- */
export function LabelTool({ id, getTime, seek }: { id: string; getTime: () => number; seek: (t: number) => void }) {
  const [l, setL] = useState<LabelSet | null>(null)
  const [d, setD] = useState<{ start: number | null; contact: number | null; end: number | null; type: string; contact_kind: string }>({ start: null, contact: null, end: null, type: 'forehand', contact_kind: 'contact' })
  const [msg, setMsg] = useState('')
  useEffect(() => { api.labels(id).then(setL).catch(err(setMsg)) }, [id])
  if (!l) return <p className="muted">Loading…</p>
  const set = (p: Partial<LabelSet>) => setL({ ...l, ...p })
  const ok = d.start !== null && d.contact !== null && d.end !== null && d.start <= d.contact && d.contact <= d.end
  const save = (next: LabelSet) => api.setLabels(id, next).then((x) => { setL(x); setMsg('Saved.') }).catch(err(setMsg))
  const mark = (k: 'start' | 'contact' | 'end') => setD({ ...d, [k]: Math.round(getTime() * 1000) / 1000 })
  return (
    <section className="card">
      <h3>Label strokes for the evaluation set</h3>
      <p className="muted small">Mark every forehand, backhand and serve in this video, whether or not the app found it. Step frame by frame to the moment of contact. Labels are judged against the app's detections by the scoring command; don't copy the app's timings. Mark volleys, slices and smashes as “other”.</p>
      <div className="grid2 tight">
        <label>Your name<input className="inline" value={l.labeller} maxLength={80} onChange={(e) => set({ labeller: e.target.value })} /></label>
        <label>Role or qualification<input className="inline" value={l.labeller_role} maxLength={60} placeholder="e.g. LTA level 3 coach" onChange={(e) => set({ labeller_role: e.target.value })} /></label>
        <label>Camera view<select value={l.camera_view} onChange={(e) => set({ camera_view: e.target.value })}>{VIEWS.map((v) => <option key={v} value={v}>{cap(v)}</option>)}</select></label>
        <label className="check"><input type="checkbox" checked={l.expert} onChange={(e) => set({ expert: e.target.checked })} /> I am a qualified tennis coach</label>
      </div>
      <table className="mini wide-t"><thead><tr><th>Stroke</th><th>Start</th><th>Contact</th><th>End</th><th>Timed at</th><th /></tr></thead>
        <tbody>{l.strokes.map((s, i) => (
          <tr key={i}><td>{cap(s.type)}</td><td>{fmtTime(s.start)}</td><td><button className="chip play" onClick={() => seek(s.contact)}>▶ {s.contact.toFixed(2)}</button></td><td>{fmtTime(s.end)}</td><td className="muted">{s.contact_kind === 'contact' ? 'ball contact' : 'swing peak (no ball seen)'}</td>
            <td><button className="icon" aria-label="Remove label" onClick={() => save({ ...l, strokes: l.strokes.filter((_, j) => j !== i) })}>✕</button></td></tr>
        ))}</tbody></table>
      <div className="add">
        <div className="row">
          <button className="btn ghost" onClick={() => mark('start')}>Start {d.start !== null && `(${d.start.toFixed(2)})`}</button>
          <button className="btn ghost" onClick={() => mark('contact')}>Contact {d.contact !== null && `(${d.contact.toFixed(2)})`}</button>
          <button className="btn ghost" onClick={() => mark('end')}>End {d.end !== null && `(${d.end.toFixed(2)})`}</button>
          <select value={d.type} onChange={(e) => setD({ ...d, type: e.target.value })} aria-label="Stroke type">{['forehand', 'backhand', 'serve', 'other'].map((t) => <option key={t} value={t}>{cap(t)}</option>)}</select>
          <select value={d.contact_kind} onChange={(e) => setD({ ...d, contact_kind: e.target.value })} aria-label="What the contact time marks"><option value="contact">Ball contact seen</option><option value="peak">No ball / contact not visible</option></select>
          <button className="btn" disabled={!ok || !l.labeller.trim()} onClick={() => { save({ ...l, strokes: [...l.strokes, d as LabelStroke].sort((a, b) => a.contact - b.contact) }); setD({ ...d, start: null, contact: null, end: null }) }}>Add label</button>
        </div>
        {!l.labeller.trim() && <p className="muted small">Enter your name first: every label file records who made it.</p>}
      </div>
      <div className="row">
        <button className="btn" disabled={!l.labeller.trim()} onClick={() => save(l)}>Save</button>
        <a className="btn ghost" href={`/api/sessions/${id}/labels/export`}>Export label file</a>
        <span className="muted small">{msg}</span>
      </div>
      <p className="muted small">The exported file goes in <code>evalset/labels/</code> next to the clip; see <code>evalset/README.md</code>.</p>
    </section>
  )
}

/* ---- evaluation: click joints on frames to check the pose model ---- */
export function AnnotateTool({ id, frame, pending, setPending, seekFrame }: { id: string; frame: () => number; pending: Record<string, number[]>; setPending: (p: Record<string, number[]>) => void; seekFrame: (f: number) => void }) {
  const [a, setA] = useState<AnnotationSet | null>(null)
  const [rep, setRep] = useState<Record<string, unknown> | null>(null)
  const [msg, setMsg] = useState('')
  const [csv, setCsv] = useState<Record<string, unknown> | null>(null)
  const [offset, setOffset] = useState(0)
  useEffect(() => { api.annotations(id).then(setA).catch(err(setMsg)) }, [id])
  if (!a) return <p className="muted">Loading…</p>
  const next = JOINTS.find((j) => !(j in pending))
  const saveFrame = () => {
    const f = frame()
    const frames = [...a.frames.filter((x) => x.frame !== f), { frame: f, joints: pending }].sort((x, y) => x.frame - y.frame)
    api.setAnnotations(id, { ...a, frames }).then((x) => { setA(x); setPending({}); setMsg(`Frame ${f} saved.`) }).catch(err(setMsg))
  }
  const pos = rep?.position_error_by_joint as Record<string, { n: number; median_px: number; median_torso_lengths: number }> | undefined
  const ang = rep?.image_plane_angle_error as Record<string, { n: number; mean_signed_deg: number; median_abs_deg: number }> | undefined
  const mc = csv?.angles as Record<string, { reference_samples: number; methods: Record<string, { n: number; mean_signed_deg?: number; mean_abs_deg?: number; rmse_deg?: number }> }> | undefined
  return (
    <section className="card">
      <h3>Check joint positions and angles</h3>
      <p className="muted small">Pause on a frame, turn the skeleton off, then click each joint centre in the order asked. Use frames from different moments of the stroke. No accuracy figure exists until someone does this; nothing here is filled in automatically.</p>
      <label>Your name<input className="inline" value={a.annotator} maxLength={80} onChange={(e) => setA({ ...a, annotator: e.target.value })} /></label>
      <p>{next ? <>Click the <strong>{next.replace('_', ' ')}</strong> <span className="muted small">(the player's own left/right) · {Object.keys(pending).length} of {JOINTS.length} placed</span></> : <strong>All joints placed.</strong>}</p>
      <div className="row">
        <button className="btn ghost" disabled={!next} onClick={() => next && setPending({ ...pending, [next]: null as unknown as number[] })}>Skip (hidden)</button>
        <button className="btn ghost" disabled={!Object.keys(pending).length} onClick={() => { const k = Object.keys(pending); const p = { ...pending }; delete p[k[k.length - 1]]; setPending(p) }}>Undo</button>
        <button className="btn" disabled={!a.annotator.trim() || Object.values(pending).filter(Boolean).length < 3} onClick={saveFrame}>Save this frame</button>
        <span className="muted small">{msg}</span>
      </div>
      <p className="muted small">Annotated frames: {a.frames.length === 0 ? 'none yet' : a.frames.map((f) => <button key={f.frame} className="chip" onClick={() => seekFrame(f.frame)}>{f.frame}</button>)}</p>
      <div className="row">
        <button className="btn" onClick={() => api.annotationReport(id).then(setRep).catch(err(setMsg))}>Error report</button>
        <a className="btn ghost" href={`/api/sessions/${id}/annotations/export`}>Export annotations</a>
      </div>
      {rep && (rep.available ? <>
        <p className="muted small">{String(rep.frames_compared)} of {String(rep.frames_annotated)} frames compared · camera view: {String(rep.camera_view)} · annotator: {String(rep.annotator)}</p>
        <table className="mini"><thead><tr><th>Joint</th><th>n</th><th>Median error (px)</th><th>Median error (torso lengths)</th></tr></thead>
          <tbody>{Object.entries(pos || {}).map(([k, v]) => <tr key={k}><th>{k.replace('_', ' ')}</th><td>{v.n}</td><td>{v.median_px}</td><td>{v.median_torso_lengths}</td></tr>)}</tbody></table>
        <table className="mini"><thead><tr><th>2D angle</th><th>n</th><th>Mean signed error</th><th>Median absolute error</th></tr></thead>
          <tbody>{Object.entries(ang || {}).map(([k, v]) => <tr key={k}><th>{k.replace(/_/g, ' ')}</th><td>{v.n}</td><td>{v.mean_signed_deg}°</td><td>{v.median_abs_deg}°</td></tr>)}</tbody></table>
        <p className="muted small">{String(rep.note)}</p>
      </> : <p className="note">{String(rep.reason)}</p>)}
      <h4>Motion-capture reference</h4>
      <p className="muted small">Upload a CSV with a <code>time_s</code> column and any of <code>knee_flexion_left</code>, <code>knee_flexion_right</code>, <code>elbow_angle_left</code>, <code>elbow_angle_right</code>, <code>trunk_lean</code> (degrees). The file is used for this comparison only and is not stored.</p>
      <div className="row">
        <label className="check">Time offset (s) <input className="inline" type="number" step="0.01" value={offset} onChange={(e) => setOffset(Number(e.target.value))} style={{ width: 90 }} /></label>
        <input type="file" accept=".csv,text/csv" onChange={(e) => { const f = e.target.files?.[0]; if (f) f.text().then((t) => api.mocap(id, t, offset)).then(setCsv).catch(err(setMsg)) }} />
      </div>
      {mc && <table className="mini"><thead><tr><th>Angle</th><th>Method</th><th>n</th><th>Mean signed</th><th>Mean abs</th><th>RMSE</th></tr></thead>
        <tbody>{Object.entries(mc).flatMap(([k, v]) => Object.entries(v.methods).map(([m, e]) => <tr key={k + m}><th>{k.replace(/_/g, ' ')}</th><td>{m === 'image_plane' ? '2D image plane' : 'single-camera 3D estimate'}</td><td>{e.n}</td><td>{e.mean_signed_deg ?? '–'}</td><td>{e.mean_abs_deg ?? '–'}</td><td>{e.rmse_deg ?? '–'}</td></tr>))}</tbody></table>}
      {csv && <p className="muted small">Camera view: {String(csv.camera_view)}. {String(csv.note)}</p>}
    </section>
  )
}

/* ---- reviewer: is each flagged clip right, and is the cue appropriate? ---- */
export function ReviewFindings({ s, onShow }: { s: Session; onShow: (a: number, b: number, id?: string) => void }) {
  const [rows, setRows] = useState<RuleReview[]>([])
  const [who, setWho] = useState(localStorage.getItem('reviewer') || '')
  const [role, setRole] = useState(localStorage.getItem('reviewer_role') || 'coach')
  const [msg, setMsg] = useState('')
  const load = () => api.reviews(s.id).then(setRows).catch(err(setMsg))
  useEffect(() => { load() }, [s.id]) // eslint-disable-line
  const observed = s.report!.findings.filter((f) => f.status === 'observed')
  const mine = (f: Finding, stroke: string | null) => rows.find((r) => r.finding_id === f.id && r.stroke_id === stroke && r.reviewer.toLowerCase() === who.toLowerCase())
  const put = (f: Finding, stroke: string | null, patch: Partial<RuleReview>) => {
    const old = mine(f, stroke)
    localStorage.setItem('reviewer', who); localStorage.setItem('reviewer_role', role)
    api.putReview(s.id, { finding_id: f.id, stroke_id: stroke, verdict: old?.verdict ?? 'unsure', cue_ok: old?.cue_ok ?? null, comment: old?.comment ?? '', ...patch, reviewer: who, role }).then(load).catch(err(setMsg))
  }
  return (
    <section className="card">
      <h3>Review the findings</h3>
      <p className="muted small">For each flagged clip: watch it, then say whether the stated observation is true. For each rule: say whether the cue is appropriate and comment on the wording. Your answers are stored with your name and are used to decide which rules to keep; they do not change this report.</p>
      <div className="grid2 tight">
        <label>Your name<input className="inline" value={who} maxLength={80} onChange={(e) => setWho(e.target.value)} /></label>
        <label>Role<select value={role} onChange={(e) => setRole(e.target.value)}>{ROLES.map(([v, t]) => <option key={v} value={v}>{t}</option>)}</select></label>
      </div>
      {msg && <p className="alert">{msg}</p>}
      {observed.length === 0 && <p className="muted">No patterns were flagged in this session, so there is nothing to review here.</p>}
      {observed.map((f) => {
        const whole = mine(f, null)
        return (
          <article key={f.id} className="finding">
            <header><div className="tags"><span className={`tag k-${f.stroke_type}`}>{cap(f.stroke_type)}</span></div><h4>{f.title}</h4></header>
            <p>{f.observation}</p>
            {f.action.cue && <p className="cue">{f.action.cue}</p>}
            <table className="mini"><tbody>
              {f.occurrences.map((o) => {
                const r = mine(f, o.stroke_id)
                return <tr key={o.stroke_id}><td><button className="chip play" onClick={() => onShow(o.start, o.end, o.stroke_id)}>▶ {fmtTime(o.time ?? o.start)}</button></td><td className="muted">measured {o.value}</td>
                  <td><div className="seg small">{(['correct', 'incorrect', 'unsure'] as const).map((v) => <button key={v} disabled={who.trim().length < 3} className={r?.verdict === v ? 'on' : ''} onClick={() => put(f, o.stroke_id, { verdict: v })}>{v === 'correct' ? 'Observation is true' : v === 'incorrect' ? 'Not true' : 'Unsure'}</button>)}</div></td></tr>
              })}
            </tbody></table>
            <div className="row">
              <span className="small">Is the cue appropriate?</span>
              <div className="seg small">{([[true, 'Yes'], [false, 'No']] as const).map(([v, t]) => <button key={t} disabled={who.trim().length < 3} className={whole?.cue_ok === v ? 'on' : ''} onClick={() => put(f, null, { cue_ok: v })}>{t}</button>)}</div>
              <input className="inline grow" placeholder="Comment on the wording (saved when you leave the box)" maxLength={1000} defaultValue={whole?.comment || ''} key={whole?.id || f.id} onBlur={(e) => e.target.value !== (whole?.comment || '') && who.trim().length >= 3 && put(f, null, { comment: e.target.value })} />
            </div>
          </article>
        )
      })}
      <div className="row"><a className="btn ghost" href="/api/review/export">Export all my reviews</a><a className="btn ghost" href="#/review">Evidence and wording sign-off</a></div>
    </section>
  )
}
