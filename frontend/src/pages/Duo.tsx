import { useEffect, useRef, useState } from 'react'
import { api, cap, fmtDate, fmtTime } from '../api'
import type { Session, Stroke } from '../api'

const src = (s?: Stroke) => (s?.timing?.source === 'ball_contact' ? 'ball contact' : s?.timing?.source === 'highest_hand' ? 'highest hand position' : 'peak hand speed')

export default function Duo({ a, b }: { a: string; b: string }) {
  const [sa, setSa] = useState<Session | null>(null)
  const [sb, setSb] = useState<Session | null>(null)
  const [ia, setIa] = useState('')
  const [ib, setIb] = useState('')
  const [u, setU] = useState(-0.8)
  const [rate, setRate] = useState(0.25)
  const [playing, setPlaying] = useState(false)
  const [error, setError] = useState('')
  const va = useRef<HTMLVideoElement>(null), vb = useRef<HTMLVideoElement>(null)
  useEffect(() => {
    api.get(a).then((x) => { setSa(x); setIa(x.report?.strokes[0]?.id || '') }).catch((e) => setError((e as Error).message))
    api.get(b).then((x) => { setSb(x); const st = x.report?.strokes || []; setIb((a === b ? st[1] : st[0])?.id || st[0]?.id || '') }).catch((e) => setError((e as Error).message))
  }, [a, b])
  const A = sa?.report?.strokes.find((s) => s.id === ia), B = sb?.report?.strokes.find((s) => s.id === ib)
  const lo = A && B ? -Math.min(A.peak - A.start, B.peak - B.start, 1.5) : -1, hi = A && B ? Math.min(A.end - A.peak, B.end - B.peak, 1.5) : 1
  const place = (x: number) => { if (va.current && A) va.current.currentTime = A.peak + x; if (vb.current && B) vb.current.currentTime = B.peak + x }
  useEffect(() => { setPlaying(false); setU(lo); place(lo) }, [ia, ib, sa, sb]) // eslint-disable-line
  useEffect(() => {
    if (!playing || !A || !B) return
    const x = va.current, y = vb.current
    if (!x || !y) return
    x.playbackRate = y.playbackRate = rate
    x.play().catch(() => {}); y.play().catch(() => {})
    let raf = 0
    const tick = () => {
      const cur = x.currentTime - A.peak
      if (cur >= hi) { place(lo); setU(lo) } else {
        if (Math.abs(y.currentTime - B.peak - cur) > 0.05) y.currentTime = B.peak + cur  // keep the second clip locked to the first
        setU(cur)
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => { cancelAnimationFrame(raf); x.pause(); y.pause() }
  }, [playing, rate, A, B, lo, hi]) // eslint-disable-line
  if (error) return <div className="narrow"><div className="card"><h2>Can't show these</h2><p>{error}</p><a className="btn" href="#/">Back</a></div></div>
  if (!sa?.report || !sb?.report) return <div className="narrow"><p className="muted">Loading…</p></div>
  const pick = (s: Session, v: string, set: (x: string) => void) => (
    <select value={v} onChange={(e) => set(e.target.value)} aria-label={`Stroke from ${s.title}`}>{s.report!.strokes.map((x) => <option key={x.id} value={x.id}>{cap(x.type)} at {fmtTime(x.peak)}</option>)}</select>
  )
  const sameView = sa.report.quality.tracking.camera_view === sb.report.quality.tracking.camera_view
  const sameTiming = A?.timing?.source === B?.timing?.source
  return (
    <div className="narrow wide">
      <h1>Two strokes side by side</h1>
      <div className={`banner ${sameView && sameTiming ? 'calm' : ''}`}>
        The clips are lined up at each stroke's timestamp: <strong>{src(A)}</strong> on the left, <strong>{src(B)}</strong> on the right.
        {!sameTiming && ' These are not the same kind of moment, so the alignment is approximate.'}
        {A?.timing?.source !== 'ball_contact' || B?.timing?.source !== 'ball_contact' ? ' Where the ball was not tracked the timestamp is near contact, not contact.' : ''}
        {!sameView && ` The camera views differ (${sa.report.quality.tracking.camera_view} vs ${sb.report.quality.tracking.camera_view}), so the same movement will look different.`}
      </div>
      {(!A || !B) ? <div className="card"><p className="muted">Both sessions need at least one stroke.</p></div> : <>
        <div className="grid2">
          {([[sa, A, va, ia, setIa], [sb, B, vb, ib, setIb]] as const).map(([s, st, ref, v, set], k) => (
            <div className="card" key={k}>
              <div className="row between"><strong><a href={`#/s/${s.id}`}>{s.title}</a></strong>{pick(s, v, set)}</div>
              <p className="muted small">{fmtDate(s.created_at)} · {s.report!.quality.tracking.camera_view} · {s.report!.meta.fps} fps</p>
              <div className="stage"><video ref={ref} src={`/api/sessions/${s.id}/video#t=${st.peak}`} muted playsInline preload="auto" /></div>
            </div>
          ))}
        </div>
        <div className="card">
          <div className="controls">
            <button className="btn primary" onClick={() => setPlaying(!playing)}>{playing ? 'Pause' : 'Play both'}</button>
            <button className="btn ghost" onClick={() => { setPlaying(false); const x = Math.max(lo, u - 0.04); setU(x); place(x) }}>◀ step</button>
            <button className="btn ghost" onClick={() => { setPlaying(false); const x = Math.min(hi, u + 0.04); setU(x); place(x) }}>step ▶</button>
            <button className="btn ghost" onClick={() => { setPlaying(false); setU(0); place(0) }}>Go to timestamp</button>
            <span className="time">{u >= 0 ? '+' : ''}{u.toFixed(2)} s</span>
            <span className="spacer" />
            <div className="seg small" role="group" aria-label="Playback speed">{[0.1, 0.25, 0.5, 1].map((r) => <button key={r} className={rate === r ? 'on' : ''} onClick={() => setRate(r)}>{r}×</button>)}</div>
          </div>
          <input type="range" min={lo} max={hi} step={0.01} value={u} aria-label="Position relative to the stroke timestamp" style={{ width: '100%' }} onChange={(e) => { setPlaying(false); const x = Number(e.target.value); setU(x); place(x) }} />
          <p className="muted small">0 is the stroke timestamp in both clips. Clips recorded at different frame rates will not step in exactly matching frames.</p>
        </div>
      </>}
    </div>
  )
}
