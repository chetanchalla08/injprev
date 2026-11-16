import { useEffect, useState } from 'react'
import { api, cap, fmtDate } from '../api'
import type { Profile, Progress } from '../api'

const KEYS: [string, string][] = [['peak_hand_speed', 'Hand speed'], ['knee_bend_dom', 'Knee bend (hitting side)'], ['trunk_lean', 'Trunk tilt'], ['follow_through_travel', 'Follow-through'], ['reach_height', 'Serve reach']]

export default function ProgressPage() {
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [pid, setPid] = useState('')
  const [p, setP] = useState<Progress | null>(null)
  const [stroke, setStroke] = useState('serve')
  useEffect(() => { api.profiles().then(setProfiles).catch(() => {}) }, [])
  useEffect(() => { api.progress(pid || undefined).then(setP).catch(() => {}) }, [pid])
  if (!p) return <div className="narrow"><p className="muted">Loading…</p></div>
  const rows = p.sessions.filter((s) => s.stroke_summary[stroke])
  return (
    <div className="narrow wide">
      <h1>Progress across sessions</h1>
      <div className={`banner ${p.mixed_views ? '' : 'calm'}`}>{p.caveat}</div>
      <div className="card">
        <div className="row">
          <label className="check">Player <select className="inline" value={pid} onChange={(e) => setPid(e.target.value)}><option value="">All sessions in this browser</option>{profiles.map((x) => <option key={x.id} value={x.id}>{x.name}</option>)}</select></label>
          <div className="seg small">{['serve', 'forehand', 'backhand'].map((s) => <button key={s} className={stroke === s ? 'on' : ''} onClick={() => setStroke(s)}>{cap(s)}</button>)}</div>
        </div>
        {rows.length === 0 ? <p className="muted">No finished sessions with {stroke}s yet.</p> : (
          <div className="scroll">
            <table className="mini wide-t cmp">
              <thead><tr><th>Session</th><th>Camera view</th><th>Strokes</th>{KEYS.map(([k, t]) => <th key={k}>{t}</th>)}<th>Patterns flagged</th></tr></thead>
              <tbody>
                {rows.map((s, i) => {
                  const g = s.stroke_summary[stroke]
                  const sameAsPrev = i > 0 && rows[i - 1].camera_view === s.camera_view
                  return (
                    <tr key={s.id}>
                      <th><a href={`#/s/${s.id}`}>{s.title}</a><br /><span className="muted small">{fmtDate(s.created_at)} · {s.fps} fps</span></th>
                      <td>{s.camera_view}{i > 0 && !sameAsPrev && <><br /><span className="tag st-failed">view changed</span></>}</td>
                      <td>{g.count}</td>
                      {KEYS.map(([k]) => { const m = g.metrics[k]; return <td key={k}>{m ? <>{m.median}{m.unit === 'deg' ? '°' : ''} <span className="muted small">({m.min}–{m.max}, n={m.n})</span></> : <span className="muted">–</span>}</td> })}
                      <td>{s.observed.filter((o) => o.stroke_type === stroke).map((o) => <div key={o.rule} className="small">{o.title} <span className="muted">{o.fired}/{o.assessed}</span></div>)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
        <p className="muted small">Medians with range and number of strokes. Hand speed is in torso lengths per second; angles in degrees. Rows marked “view changed” were filmed from a different camera view than the row above, so a difference there may be the camera, not the player. With fewer than three strokes a median means very little.</p>
      </div>
    </div>
  )
}
