import { useEffect, useState } from 'react'
import { api, cap } from '../api'
import type { ReviewItem } from '../api'

const ROLES: [string, string][] = [['physiotherapist', 'Physiotherapist'], ['sports_physician', 'Sports physician'], ['other_clinician', 'Other clinician'], ['coach', 'Coach']]
const KINDS: [string, string][] = [['evidence', 'Evidence entries'], ['rule', 'Rules: wording, cues and interpretations'], ['text', 'Fixed guidance text'], ['drill', 'Drills']]

export default function Review() {
  const [items, setItems] = useState<ReviewItem[]>([])
  const [enabled, setEnabled] = useState(false)
  const [me, setMe] = useState({ code: '', reviewer: localStorage.getItem('reviewer') || '', role: 'physiotherapist', credentials: '' })
  const [notes, setNotes] = useState<Record<string, string>>({})
  const [msg, setMsg] = useState('')
  const load = () => api.reviewItems().then((r) => { setItems(r.items); setEnabled(r.reviewer_mode) }).catch((e) => setMsg((e as Error).message))
  useEffect(() => { load() }, [])
  const sign = (it: ReviewItem, decision: string) => {
    setMsg('')
    api.signoff({ ...me, target: it.id, decision, hash: it.hash, comment: notes[it.id] || '' }).then(() => { localStorage.setItem('reviewer', me.reviewer); load() }).catch((e) => setMsg((e as Error).message))
  }
  const ready = enabled && me.code && me.reviewer.trim().length >= 3
  const done = items.filter((i) => i.status.state === 'approved').length
  return (
    <div className="narrow wide">
      <h1>Reviewer mode</h1>
      <div className="banner">Nothing here is marked as reviewed until a named reviewer signs it off on this page. A sign-off applies to the exact wording shown; if the wording is edited later, the item goes back to “review needed”. Health-related items need a clinician; coaching items can be signed off by a coach.</div>
      <div className="card">
        <h3>Who is reviewing</h3>
        {!enabled && <p className="alert">Sign-off is switched off on this server: the operator has not set a reviewer code (REVIEWER_CODE). You can read every item below, but nothing can be signed off.</p>}
        <div className="grid2 tight">
          <label>Reviewer code (from the operator)<input className="inline" type="password" autoComplete="off" value={me.code} onChange={(e) => setMe({ ...me, code: e.target.value })} /></label>
          <label>Full name<input className="inline" value={me.reviewer} maxLength={80} onChange={(e) => setMe({ ...me, reviewer: e.target.value })} /></label>
          <label>Role<select value={me.role} onChange={(e) => setMe({ ...me, role: e.target.value })}>{ROLES.map(([v, t]) => <option key={v} value={v}>{t}</option>)}</select></label>
          <label>Registration or qualification<input className="inline" value={me.credentials} maxLength={120} placeholder="e.g. HCPC PH12345" onChange={(e) => setMe({ ...me, credentials: e.target.value })} /></label>
        </div>
        <p className="muted small">{done} of {items.length} items currently approved. Your name, role and registration are shown next to anything you sign off. To review whether the rules fire correctly on real clips, open a session and use its Review tab.</p>
        {msg && <p className="alert" role="alert">{msg}</p>}
      </div>
      {KINDS.map(([kind, title]) => (
        <section key={kind}>
          <h3 className="sec">{title}</h3>
          {items.filter((i) => i.kind === kind).map((it) => (
            <article key={it.id} className="finding">
              <header><div className="tags"><span className={`tag ${it.status.state === 'approved' ? 'soft' : it.status.state === 'unreviewed' ? 'hollow' : 'st-failed'}`}>{it.status.label}</span><span className="tag hollow">Needs: {it.needs}</span></div><h4>{it.title}</h4></header>
              <dl>{Object.entries(it.text).map(([k, v]) => <div key={k} style={{ display: 'contents' }}><dt>{cap(k)}</dt><dd>{Array.isArray(v) ? <ul>{v.map((x) => <li key={x}>{x}</li>)}</ul> : k === 'doi' ? <a href={`https://doi.org/${v}`} target="_blank" rel="noreferrer">{v}</a> : v}</dd></div>)}</dl>
              {it.status.signoffs.map((so) => (
                <p key={so.id} className="muted small">{so.date}: {so.reviewer} ({cap(so.role)}{so.credentials ? `, ${so.credentials}` : ''}) — {so.decision === 'approved' ? 'approved' : 'requested changes'}{so.current ? '' : ' (an earlier wording)'}{so.comment ? `: “${so.comment}”` : ''}{' '}
                  {ready && <button className="link" onClick={() => api.withdraw(so.id, me.code).then(load).catch((e) => setMsg((e as Error).message))}>withdraw</button>}</p>
              ))}
              <div className="row no-print">
                <input className="inline grow" placeholder="Comment (required when requesting changes)" maxLength={2000} value={notes[it.id] || ''} onChange={(e) => setNotes({ ...notes, [it.id]: e.target.value })} />
                <button className="btn" disabled={!ready} onClick={() => sign(it, 'approved')}>Approve as written</button>
                <button className="btn ghost" disabled={!ready || !(notes[it.id] || '').trim()} onClick={() => sign(it, 'changes_requested')}>Request changes</button>
              </div>
            </article>
          ))}
        </section>
      ))}
    </div>
  )
}
