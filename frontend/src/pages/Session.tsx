import { useCallback, useEffect, useRef, useState } from 'react'
import { api, fmtDate } from '../api'
import type { Comment, Config, Drawing, Segment, Session } from '../api'
import Player from '../components/Player'
import type { PlayerHandle } from '../components/Player'
import ReportView from '../components/Report'
import { AnnotateTool, CoachPanel, ConfirmPlayer, JOINTS, LabelTool, ReviewFindings, SummaryBox } from '../components/Tools'

const ACTIVE = new Set(['queued_prepare', 'preparing', 'queued_analyze', 'analyzing'])
type View = 'player' | 'coach' | 'clinician' | 'label' | 'annotate' | 'review'
const TABS: [View, string][] = [['player', 'Player'], ['coach', 'Coach'], ['clinician', 'Clinician'], ['label', 'Label'], ['annotate', 'Annotate'], ['review', 'Review']]

export default function SessionPage({ id, shareToken }: { id?: string; shareToken?: string }) {
  const [s, setS] = useState<Session | null>(null)
  const [cfg, setCfg] = useState<Config | null>(null)
  const [error, setError] = useState('')
  const [view, setView] = useState<View>('player')
  const [activeStroke, setActiveStroke] = useState<string | null>(null)
  const [comments, setComments] = useState<Comment[]>([])
  const [drawings, setDrawings] = useState<Drawing[]>([])
  const [box, setBox] = useState<number[] | null>(null)
  const [pending, setPending] = useState<Record<string, number[]>>({})
  const player = useRef<PlayerHandle>(null)
  const shared = !!shareToken
  const base = shared ? `/api/shared/${shareToken}` : `/api/sessions/${id}`

  const load = useCallback(() => (shared ? api.shared(shareToken!) : api.get(id!)).then((x) => { setS(x); if (shared) { setComments(x.comments || []); setDrawings(x.drawings || []) } }).catch((e) => setError((e as Error).message)), [id, shared, shareToken])
  const loadNotes = useCallback(() => { if (!shared && id) { api.comments(id).then(setComments).catch(() => {}); api.drawings(id).then(setDrawings).catch(() => {}) } }, [id, shared])
  useEffect(() => { load(); loadNotes(); if (!shared) api.config().then(setCfg).catch(() => {}) }, [load, loadNotes, shared])
  useEffect(() => {
    if (!s || !ACTIVE.has(s.status)) return
    const t = setTimeout(load, 1000)
    return () => clearTimeout(t)
  }, [s, load])

  const remove = async () => {
    if (!s || !confirm('Delete this session? The video, landmark data, report, comments, drawings, labels and share links will be removed.')) return
    await api.remove(s.id)
    window.location.hash = '#/'
  }
  const act = (p: Promise<unknown>) => p.then(load).catch((e) => setError((e as Error).message))

  if (error && !s) return <div className="narrow"><div className="card"><h2>{shared ? 'Link not available' : 'Session not available'}</h2><p>{error}</p><p className="muted">{shared ? 'Ask the person who sent it for a new link.' : 'It may have been deleted, expired, or been created in a different browser.'}</p>{!shared && <a className="btn" href="#/">Back to sessions</a>}</div></div>
  if (!s) return <div className="narrow"><p className="muted">Loading…</p></div>

  if (ACTIVE.has(s.status)) {
    const phase = s.status.includes('prepare') || s.status === 'preparing' ? 1 : 2
    return (
      <div className="narrow">
        <div className="card center">
          <h2>{phase === 1 ? 'Checking your video' : 'Analysing your strokes'}</h2>
          <p className="muted">Step {phase} of 2 · {s.stage || 'Queued'}</p>
          <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(s.progress * 100)}><div style={{ width: `${Math.max(3, Math.round(s.progress * 100))}%` }} /></div>
          <p className="muted small">Pose estimation runs on every frame, so a one-minute clip can take a couple of minutes. You can leave this page; the session will be in your list.</p>
          <button className="link" onClick={remove}>Cancel and delete</button>
        </div>
      </div>
    )
  }

  if (s.status === 'failed') {
    return (
      <div className="narrow">
        <div className="card">
          <h2>We couldn't analyse this video</h2>
          <p>{s.error?.message}</p>
          {s.error?.suggestion && <p className="note">{s.error.suggestion}</p>}
          {error && <p className="alert">{error}</p>}
          <div className="row">
            {(s.error?.code === 'internal_error' || s.error?.code === 'model_missing') && <button className="btn primary" onClick={() => act(api.retry(s.id))}>Try again</button>}
            {(s.error?.code === 'tracking_lost' || s.error?.code === 'tracking_poor') && s.candidates.length > 1 && <button className="btn primary" onClick={() => setS({ ...s, status: 'awaiting_player' })}>Pick the player again</button>}
            <a className="btn" href="#/">Upload a different clip</a>
            <button className="btn ghost" onClick={remove}>Delete</button>
          </div>
        </div>
      </div>
    )
  }

  if (s.status === 'awaiting_player') {
    return (
      <div className="narrow wide">
        <div className="card">
          <h2>Who should we follow?</h2>
          <p className="muted">More than one person is visible. Tap the player you want analysed.</p>
          {error && <p className="alert">{error}</p>}
          <div className="picker">
            <img src={`/api/sessions/${s.id}/candidates.jpg`} alt="A frame from your video with each detected person outlined" />
            {s.candidates.map((c, i) => (
              <button key={i} style={{ left: `${c.box[0] * 100}%`, top: `${c.box[1] * 100}%`, width: `${(c.box[2] - c.box[0]) * 100}%`, height: `${(c.box[3] - c.box[1]) * 100}%` }}
                onClick={() => act(api.choosePlayer(s.id, i))} aria-label={`Player ${i + 1}`}><span>{i + 1}</span></button>
            ))}
          </div>
          <p className="muted small">If the player you want isn't outlined, they were too small or hidden in this frame; a closer clip will work better.</p>
        </div>
      </div>
    )
  }

  const r = s.report!
  const show = (start: number, end: number, strokeId?: string) => { if (strokeId) setActiveStroke(strokeId); player.current?.playRange(Math.max(0, start), end) }
  const showSeg = (g: Segment | null) => { setBox(g?.box ?? null); if (g) { player.current?.pause(); player.current?.seek(g.box_t) } }
  const reportView = view === 'label' || view === 'annotate' || view === 'review' ? 'coach' : view
  const nextJoint = JOINTS.find((j) => !(j in pending))
  const marks = view === 'annotate' ? Object.entries(pending).filter(([, p]) => p).map(([k, p]) => ({ x: p[0], y: p[1], label: k.replace('_', ' ') })) : undefined
  return (
    <div className="report-page">
      {shared && <div className="banner calm"><strong>Shared, read-only view.</strong> Someone sent you this session. You can watch and read but not change anything{s.shared_until ? `; the link stops working on ${fmtDate(s.shared_until)}` : ''}. Self-reported health details are not included.</div>}
      <div className="row between head-row">
        <div>
          <h1>{s.title}</h1>
          <p className="muted">{fmtDate(s.created_at)} · {Math.round(r.meta.duration_s)} s at {r.meta.fps} fps · {r.handedness}-handed{s.expires_at ? ` · auto-deletes ${fmtDate(s.expires_at)}` : ''}</p>
        </div>
        {!shared && <div className="row no-print">
          <a className="btn ghost" href={`/api/sessions/${s.id}/report.pdf`}>PDF report</a>
          <a className="btn ghost" href={`/api/sessions/${s.id}/export`}>Download data</a>
          {s.candidates.length > 1 && <button className="btn ghost" onClick={() => setS({ ...s, status: 'awaiting_player' })}>Change player</button>}
          <button className="btn ghost" onClick={remove}>Delete</button>
        </div>}
      </div>
      <div className="layout">
        <div className="sticky">
          <Player ref={player} base={base} fps={r.meta.fps} duration={r.meta.duration_s} strokes={r.strokes} handedness={r.handedness} activeStroke={activeStroke} comments={comments} drawings={drawings} box={box} marks={marks}
            onSaveDrawing={!shared && view === 'coach' ? (t, shapes, note) => api.addDrawing(s.id, { t, shapes, note, author: localStorage.getItem('author') || '' }).then(loadNotes) : undefined}
            onStageClick={view === 'annotate' && nextJoint ? (x, y) => setPending({ ...pending, [nextJoint]: [x, y] }) : undefined}
            onSelectStroke={(sid) => { const st = r.strokes.find((x) => x.id === sid); if (st) show(st.start, st.end, sid) }} />
        </div>
        <div>
          <div className="tabs no-print" role="tablist" aria-label="Report detail">
            {TABS.filter(([v]) => !shared || ['player', 'coach', 'clinician'].includes(v)).map(([v, label]) => (
              <button key={v} role="tab" aria-selected={view === v} className={view === v ? 'on' : ''} onClick={() => { setView(v); setPending({}) }}>{label}</button>
            ))}
          </div>
          {error && <p className="alert">{error}</p>}
          {!shared && r.tracking_events && <ConfirmPlayer segments={r.tracking_events.segments} onShow={showSeg} onDecide={(seg, d) => api.decideSegment(s.id, seg, d).then(setS).catch((e) => setError((e as Error).message))} />}
          {view === 'label' && <LabelTool id={s.id} getTime={() => player.current?.time() ?? 0} seek={(t) => player.current?.seek(t)} />}
          {view === 'annotate' && <AnnotateTool id={s.id} frame={() => player.current?.frame() ?? 0} pending={pending} setPending={setPending} seekFrame={(f) => player.current?.seek(f / r.meta.fps)} />}
          {view === 'review' && <ReviewFindings s={s} onShow={show} />}
          {(view === 'player' || view === 'coach' || view === 'clinician') && <>
            {!shared && view === 'player' && <SummaryBox id={s.id} />}
            {!shared && view === 'coach' && <CoachPanel s={s} cfg={cfg} comments={comments} drawings={drawings} getTime={() => player.current?.time() ?? 0} seek={(t) => player.current?.seek(t)} reload={() => { load(); loadNotes() }} />}
            {shared && comments.length > 0 && view !== 'player' && <section className="card"><h3>Comments</h3><ul className="plain">{comments.map((c) => <li key={c.id}><button className="chip play" onClick={() => player.current?.seek(c.t)}>▶ {c.t.toFixed(1)}s</button> {c.author && <strong>{c.author}: </strong>}{c.body}</li>)}</ul></section>}
            <ReportView session={s} view={reportView} activeStroke={activeStroke} onShow={show} getTime={() => player.current?.time() ?? 0} readOnly={shared}
              onEdit={(strokes) => { setError(''); return api.setStrokes(s.id, strokes).then(setS).catch((e) => setError((e as Error).message)) }} />
          </>}
        </div>
      </div>
    </div>
  )
}
