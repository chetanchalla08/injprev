import { useEffect, useState } from 'react'
import { api, cap, fmtDate } from '../api'
import type { Comparison } from '../api'

export default function Compare({ a, b }: { a: string; b: string }) {
  const [c, setC] = useState<Comparison | null>(null)
  const [error, setError] = useState('')
  useEffect(() => { api.compare(a, b).then(setC).catch((e) => setError((e as Error).message)) }, [a, b])
  if (error) return <div className="narrow"><div className="card"><h2>Can't compare these</h2><p>{error}</p><a className="btn" href="#/">Back</a></div></div>
  if (!c) return <div className="narrow"><p className="muted">Loading…</p></div>
  const [x, y] = [...c.sessions].sort((p, q) => p.created_at - q.created_at)
  const strokes = [...new Set([...Object.keys(x.stroke_summary), ...Object.keys(y.stroke_summary)])].filter((s) => s !== 'unclassified')
  return (
    <div className="narrow wide">
      <h1>Session comparison</h1>
      <div className={`banner ${c.comparable ? 'calm' : ''}`}>{c.caveat} A difference here is a prompt to look at the video, not proof that technique changed.</div>
      <div className="card">
        <div className="scroll">
          <table className="mini wide-t cmp">
            <thead>
              <tr><th /><th><a href={`#/s/${x.id}`}>{x.title}</a><br /><span className="muted">{fmtDate(x.created_at)} · {x.camera_view} · {x.fps} fps</span></th>
                <th><a href={`#/s/${y.id}`}>{y.title}</a><br /><span className="muted">{fmtDate(y.created_at)} · {y.camera_view} · {y.fps} fps</span></th><th>Difference</th></tr>
            </thead>
            <tbody>
              {strokes.flatMap((st) => {
                const sx = x.stroke_summary[st], sy = y.stroke_summary[st]
                const keys = [...new Set([...Object.keys(sx?.metrics || {}), ...Object.keys(sy?.metrics || {})])]
                return [
                  <tr key={st} className="grp"><th>{cap(st)}</th><td>{sx?.count ?? 0} strokes</td><td>{sy?.count ?? 0} strokes</td><td /></tr>,
                  ...keys.map((k) => {
                    const mx = sx?.metrics[k], my = sy?.metrics[k]
                    const m = mx || my!
                    const u = m.unit === 'deg' ? '°' : ''
                    const few = (mx?.n ?? 0) < 3 || (my?.n ?? 0) < 3
                    return (
                      <tr key={st + k}><th className="sub">{m.label}</th>
                        <td>{mx ? `${mx.median}${u} (${mx.min}–${mx.max}, n=${mx.n})` : <span className="muted">not assessed</span>}</td>
                        <td>{my ? `${my.median}${u} (${my.min}–${my.max}, n=${my.n})` : <span className="muted">not assessed</span>}</td>
                        <td>{mx && my ? <>{my.median - mx.median > 0 ? '+' : ''}{(my.median - mx.median).toFixed(1)}{u}{few && <span className="muted small"> · few strokes</span>}</> : '–'}</td></tr>
                    )
                  }),
                ]
              })}
            </tbody>
          </table>
        </div>
        <p className="muted small">Values are medians with the range and number of strokes. With fewer than three strokes in either session a difference is very weak evidence of change.</p>
      </div>
      <div className="grid2">
        {[x, y].map((s) => (
          <div className="card" key={s.id}>
            <h3>{s.title}: patterns observed</h3>
            {s.observed.length === 0 ? <p className="muted">None in the checks that recording allowed.</p> : <ul className="plain">{s.observed.map((o) => <li key={o.title + o.stroke_type}><span className={`tag k-${o.stroke_type}`}>{cap(o.stroke_type)}</span> {o.title} <span className="muted">· {o.support.fired} of {o.support.assessed}</span></li>)}</ul>}
          </div>
        ))}
      </div>
    </div>
  )
}
