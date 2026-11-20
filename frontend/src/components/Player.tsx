import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react'
import { api, fmtTime } from '../api'
import type { Comment, Drawing, Overlay, Shape, Stroke } from '../api'

export interface PlayerHandle { playRange: (start: number, end: number) => void; seek: (t: number) => void; time: () => number; pause: () => void; frame: () => number; el: () => HTMLVideoElement | null }

// pairs of MediaPipe landmark ids
const BONES: [number, number][] = [[11, 12], [11, 23], [12, 24], [23, 24], [11, 13], [13, 15], [12, 14], [14, 16], [23, 25], [25, 27], [24, 26], [26, 28], [27, 29], [29, 31], [27, 31], [28, 30], [30, 32], [28, 32]]
const RIGHT_ARM = new Set(['12-14', '14-16'])
const LEFT_ARM = new Set(['11-13', '13-15'])
const COLORS: Record<string, string> = { yellow: '#ffe14d', red: '#ff5a4d', blue: '#58b7ff', white: '#ffffff' }
const NEED = { line: 2, angle: 3, free: 400 }

function angleDeg(p: number[][], aspect: number) {
  const v = (a: number[], b: number[]) => [(a[0] - b[0]) * aspect, a[1] - b[1]]
  const [a, b] = [v(p[0], p[1]), v(p[2], p[1])]
  const c = (a[0] * b[0] + a[1] * b[1]) / (Math.hypot(a[0], a[1]) * Math.hypot(b[0], b[1]) || 1)
  return (Math.acos(Math.max(-1, Math.min(1, c))) * 180) / Math.PI
}

function drawShapes(g: CanvasRenderingContext2D, shapes: Shape[], w: number, h: number) {
  for (const s of shapes) {
    if (s.points.length < 1) continue
    g.strokeStyle = g.fillStyle = COLORS[s.color || 'yellow']
    g.lineWidth = Math.max(2, w / 300); g.lineCap = 'round'; g.lineJoin = 'round'; g.setLineDash([]); g.globalAlpha = 1
    g.beginPath()
    s.points.forEach((p, i) => (i ? g.lineTo(p[0] * w, p[1] * h) : g.moveTo(p[0] * w, p[1] * h)))
    g.stroke()
    if (s.type !== 'free') for (const p of s.points) { g.beginPath(); g.arc(p[0] * w, p[1] * h, 3, 0, 7); g.fill() }
    if (s.type === 'angle' && s.points.length === 3) {
      g.font = `600 ${Math.max(13, w / 45)}px system-ui`
      const txt = `${Math.round(angleDeg(s.points, w / h))}° (2D)`
      g.lineWidth = 3; g.strokeStyle = '#000'; g.strokeText(txt, s.points[1][0] * w + 8, s.points[1][1] * h - 8); g.fillText(txt, s.points[1][0] * w + 8, s.points[1][1] * h - 8)
    }
  }
}

interface Props {
  base: string; fps: number; duration: number; strokes: Stroke[]; handedness: string; activeStroke: string | null; onSelectStroke: (id: string) => void
  comments?: Comment[]; drawings?: Drawing[]; onSaveDrawing?: (t: number, shapes: Shape[], note: string) => Promise<unknown>
  box?: number[] | null; onStageClick?: (x: number, y: number) => void; marks?: { x: number; y: number; label: string }[]; onTime?: (t: number) => void; compact?: boolean
}

const Player = forwardRef<PlayerHandle, Props>(function Player({ base, fps, duration, strokes, handedness, activeStroke, onSelectStroke, comments = [], drawings = [], onSaveDrawing, box, onStageClick, marks, onTime, compact }, ref) {
  const video = useRef<HTMLVideoElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const overlay = useRef<Overlay | null>(null)
  const loop = useRef<{ start: number; end: number } | null>(null)
  const [show, setShow] = useState(true)
  const [rate, setRate] = useState(1)
  const [playing, setPlaying] = useState(false)
  const [t, setT] = useState(0)
  const [looping, setLooping] = useState(false)
  const [overlayError, setOverlayError] = useState(false)
  const [tool, setTool] = useState<'' | 'line' | 'angle' | 'free'>('')
  const [color, setColor] = useState('yellow')
  const [draft, setDraft] = useState<Shape[]>([])
  const [note, setNote] = useState('')
  const live = useRef({ show, draft, drawings, box, marks, tool, color, current: null as Shape | null })
  live.current = { ...live.current, show, draft, drawings, box, marks, tool, color }

  const draw = useCallback(() => {
    const v = video.current, c = canvas.current, o = overlay.current, L = live.current
    if (!v || !c) return
    const w = v.clientWidth, h = v.clientHeight, dpr = window.devicePixelRatio || 1
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); c.style.width = `${w}px`; c.style.height = `${h}px` }
    const g = c.getContext('2d')!
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    g.clearRect(0, 0, w, h)
    if (o && L.show) {
      const idx = Math.min(o.frames.length - 1, Math.max(0, Math.round((v.currentTime * o.fps) / o.stride)))
      const f = o.frames[idx]
      if (f) {
        const at = (id: number) => { const k = o.points.indexOf(id) * 3; return { x: (f[k] / 1000) * w, y: (f[k + 1] / 1000) * h, v: f[k + 2] } }
        const hit = handedness === 'left' ? LEFT_ARM : RIGHT_ARM
        g.lineCap = 'round'
        for (const [a, b] of BONES) {
          const p = at(a), q = at(b)
          const weak = Math.min(p.v, q.v) < 5
          g.globalAlpha = weak ? 0.35 : 0.95
          g.setLineDash(weak ? [4, 5] : [])
          g.strokeStyle = hit.has(`${a}-${b}`) ? '#ffb454' : '#e9fbe5'
          g.lineWidth = Math.max(2, w / 360)
          g.beginPath(); g.moveTo(p.x, p.y); g.lineTo(q.x, q.y); g.stroke()
        }
        g.setLineDash([])
        for (const id of o.points) {
          if (id === 0 || id > 28) continue
          const p = at(id)
          g.globalAlpha = p.v < 5 ? 0.4 : 1
          g.fillStyle = '#123524'; g.strokeStyle = '#e9fbe5'; g.lineWidth = 1.5
          g.beginPath(); g.arc(p.x, p.y, Math.max(3, w / 260), 0, Math.PI * 2); g.fill(); g.stroke()
        }
      }
      const b = o.ball?.[idx]
      if (b) { g.globalAlpha = 1; g.strokeStyle = '#d5e44b'; g.lineWidth = 2; g.beginPath(); g.arc((b[0] / 1000) * w, (b[1] / 1000) * h, Math.max(6, w / 110), 0, 7); g.stroke() }
      g.globalAlpha = 1
    }
    if (L.box) { g.setLineDash([6, 4]); g.strokeStyle = '#d5e44b'; g.lineWidth = 3; g.strokeRect(L.box[0] * w, L.box[1] * h, (L.box[2] - L.box[0]) * w, (L.box[3] - L.box[1]) * h); g.setLineDash([]) }
    for (const d of L.drawings) if (Math.abs(d.t - v.currentTime) <= 0.4) drawShapes(g, d.shapes, w, h)
    drawShapes(g, L.current ? [...L.draft, L.current] : L.draft, w, h)
    for (const m of L.marks || []) {
      g.fillStyle = '#ff5a4d'; g.strokeStyle = '#fff'; g.lineWidth = 1.5
      g.beginPath(); g.arc(m.x * w, m.y * h, 5, 0, 7); g.fill(); g.stroke()
      g.font = '600 11px system-ui'; g.fillStyle = '#fff'; g.strokeStyle = '#000'; g.lineWidth = 3; g.strokeText(m.label, m.x * w + 7, m.y * h + 4); g.fillText(m.label, m.x * w + 7, m.y * h + 4)
    }
  }, [handedness])

  useEffect(() => {
    api.overlayAt(base).then((o) => { overlay.current = o; draw() }).catch(() => setOverlayError(true))
  }, [base, draw])

  useEffect(() => {
    const v = video.current
    if (!v) return
    let raf = 0, alive = true
    const tick = () => {
      if (!alive) return
      if (loop.current && v.currentTime >= loop.current.end) v.currentTime = loop.current.start
      draw()
      setT(v.currentTime)
      onTime?.(v.currentTime)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    const ro = new ResizeObserver(draw)
    ro.observe(v)
    return () => { alive = false; cancelAnimationFrame(raf); ro.disconnect() }
  }, [draw, onTime])

  useEffect(() => { if (video.current) video.current.playbackRate = rate }, [rate])

  const seek = (time: number) => { const v = video.current; if (v) v.currentTime = Math.max(0, Math.min(duration, time)) }
  const stopLoop = () => { loop.current = null; setLooping(false) }
  useImperativeHandle(ref, () => ({
    seek: (time) => { stopLoop(); seek(time) },
    time: () => video.current?.currentTime ?? 0,
    frame: () => Math.round((video.current?.currentTime ?? 0) * fps),
    pause: () => video.current?.pause(),
    el: () => video.current,
    playRange: (start, end) => {
      const v = video.current
      if (!v) return
      loop.current = { start, end }; setLooping(true)
      v.currentTime = start
      if (rate === 1) setRate(0.5)
      v.play().catch(() => {})
      v.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
    },
  }))

  const toggle = () => { const v = video.current; if (!v) return; if (v.paused) v.play().catch(() => {}); else v.pause() }
  const step = (n: number) => { const v = video.current; if (!v) return; v.pause(); stopLoop(); seek(v.currentTime + n / fps) }
  const pos = (e: React.PointerEvent) => { const r = canvas.current!.getBoundingClientRect(); return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))] }
  const interactive = !!tool || !!onStageClick

  const down = (e: React.PointerEvent) => {
    const p = pos(e)
    if (onStageClick && !tool) return onStageClick(p[0], p[1])
    if (!tool) return
    video.current?.pause()
    const L = live.current
    if (tool === 'free') { L.current = { type: 'free', points: [p], color }; (e.target as Element).setPointerCapture(e.pointerId); return }
    const cur = L.current && L.current.type === tool ? L.current : { type: tool, points: [], color }
    cur.points = [...cur.points, p]
    if (cur.points.length >= NEED[tool]) { setDraft((d) => [...d, cur]); L.current = null } else L.current = cur
    draw()
  }
  const move = (e: React.PointerEvent) => {
    const L = live.current
    if (tool === 'free' && L.current && L.current.points.length < NEED.free) { L.current.points = [...L.current.points, pos(e)]; draw() }
  }
  const up = () => {
    const L = live.current
    if (tool === 'free' && L.current) { if (L.current.points.length > 1) setDraft((d) => [...d, L.current!]); L.current = null }
  }
  const save = async () => {
    if (!onSaveDrawing || !draft.length) return
    await onSaveDrawing(video.current?.currentTime ?? 0, draft, note)
    setDraft([]); setNote(''); setTool('')
  }

  return (
    <div className="player">
      <div className="stage">
        <video ref={video} src={`${base}/video`} playsInline muted preload="auto" onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onClick={toggle} />
        <canvas ref={canvas} aria-hidden={!interactive} className={interactive ? 'draw' : ''} onPointerDown={down} onPointerMove={move} onPointerUp={up} />
      </div>
      <div className="timeline" onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); stopLoop(); seek(((e.clientX - r.left) / r.width) * duration) }} role="slider" aria-label="Video position" aria-valuemin={0} aria-valuemax={duration} aria-valuenow={t} tabIndex={0}
        onKeyDown={(e) => { if (e.key === 'ArrowRight') step(1); if (e.key === 'ArrowLeft') step(-1) }}>
        {strokes.map((s) => (
          <button key={s.id} className={`seg-mark k-${s.type} ${activeStroke === s.id ? 'active' : ''}`} style={{ left: `${(s.start / duration) * 100}%`, width: `${Math.max(((s.end - s.start) / duration) * 100, 0.8)}%` }}
            title={`${s.type} at ${fmtTime(s.peak)}`} aria-label={`${s.type} at ${fmtTime(s.peak)}`} onClick={(e) => { e.stopPropagation(); onSelectStroke(s.id) }} />
        ))}
        {comments.map((c) => <span key={c.id} className="pin" style={{ left: `${(c.t / duration) * 100}%` }} title={`${fmtTime(c.t)} ${c.author ? c.author + ': ' : ''}${c.body}`} />)}
        {drawings.map((d) => <span key={d.id} className="pin draw-pin" style={{ left: `${(d.t / duration) * 100}%` }} title={`Drawing at ${fmtTime(d.t)}${d.note ? ': ' + d.note : ''}`} />)}
        <div className="head" style={{ left: `${(t / duration) * 100}%` }} />
      </div>
      <div className="controls no-print">
        <button className="btn" onClick={toggle}>{playing ? 'Pause' : 'Play'}</button>
        <button className="btn ghost" onClick={() => step(-1)} aria-label="Back one frame">◀ frame</button>
        <button className="btn ghost" onClick={() => step(1)} aria-label="Forward one frame">frame ▶</button>
        <span className="time">{fmtTime(t)} / {fmtTime(duration)}</span>
        <span className="spacer" />
        {looping && <button className="btn ghost" onClick={stopLoop}>Stop looping clip</button>}
        <div className="seg small" role="group" aria-label="Playback speed">
          {[0.25, 0.5, 1].map((r) => <button key={r} className={rate === r ? 'on' : ''} onClick={() => setRate(r)}>{r}×</button>)}
        </div>
        <label className="check"><input type="checkbox" checked={show} onChange={(e) => setShow(e.target.checked)} /> Skeleton</label>
      </div>
      {onSaveDrawing && (
        <div className="controls drawbar no-print">
          <span className="muted small">Draw on this frame:</span>
          <div className="seg small" role="group" aria-label="Drawing tool">
            {(['line', 'angle', 'free'] as const).map((k) => <button key={k} className={tool === k ? 'on' : ''} onClick={() => { live.current.current = null; setTool(tool === k ? '' : k) }}>{k === 'free' ? 'Freehand' : k === 'line' ? 'Line' : 'Angle'}</button>)}
          </div>
          {tool && <div className="seg small" role="group" aria-label="Colour">{Object.keys(COLORS).map((c) => <button key={c} className={color === c ? 'on' : ''} onClick={() => setColor(c)} aria-label={c}><i className="swatch" style={{ background: COLORS[c] }} /></button>)}</div>}
          {draft.length > 0 && <>
            <input className="inline" placeholder="Note (optional)" maxLength={300} value={note} onChange={(e) => setNote(e.target.value)} />
            <button className="btn" onClick={save}>Save drawing</button>
            <button className="btn ghost" onClick={() => setDraft(draft.slice(0, -1))}>Undo</button>
          </>}
          {tool === 'angle' && <span className="muted small">Click three points; the middle one is the joint. The angle is a 2D picture angle, not a joint measurement.</span>}
        </div>
      )}
      {!compact && <p className="legend muted small">
        <span className="dot k-forehand" /> forehand <span className="dot k-backhand" /> backhand <span className="dot k-serve" /> serve <span className="dot k-unclassified" /> unlabelled
        · orange = hitting arm · dashed = the model was unsure where that limb was · yellow ring = tracked ball{overlayError ? ' · skeleton data could not be loaded' : ''}
      </p>}
    </div>
  )
})
export default Player
