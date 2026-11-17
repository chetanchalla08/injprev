import { useEffect, useRef, useState } from 'react'
import { api, cap, fmtDate } from '../api'
import type { Config, Intake, Profile, SessionSummary } from '../api'

const REGIONS = ['shoulder', 'elbow', 'wrist', 'lower_back', 'knee', 'ankle', 'other']
const GOALS = ['serve', 'forehand', 'backhand', 'consistency', 'power', 'general']
const SPORTS = [
  { id: 'tennis', name: 'Tennis', icon: '🎾', ready: true, blurb: 'Forehands, backhands and serves' },
  { id: 'pickleball', name: 'Pickleball', icon: '🏓', ready: false, blurb: '' },
  { id: 'golf', name: 'Golf', icon: '⛳', ready: false, blurb: '' },
]
const STATUS: Record<string, string> = {
  queued_prepare: 'Queued', preparing: 'Checking video', awaiting_player: 'Pick the player', queued_analyze: 'Queued', analyzing: 'Analysing', complete: 'Ready', failed: 'Needs attention',
}
const EMPTY: Intake = {
  handedness: 'right', skill_level: 'intermediate', backhand_style: 'two_handed', goals: [], playing_frequency: '', workload_change: '',
  discomfort: { current: false, regions: [], notes: '' }, injury_history: '',
}

function Choice({ label, value, options, onChange }: { label: string; value: string; options: [string, string][]; onChange: (v: string) => void }) {
  return (
    <fieldset className="choice">
      <legend>{label}</legend>
      <div className="seg">
        {options.map(([v, text]) => (
          <button type="button" key={v} className={value === v ? 'on' : ''} aria-pressed={value === v} onClick={() => onChange(v)}>{text}</button>
        ))}
      </div>
    </fieldset>
  )
}

export default function Home() {
  const [cfg, setCfg] = useState<Config | null>(null)
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [file, setFile] = useState<File | null>(null)
  const [intake, setIntake] = useState<Intake>(EMPTY)
  const [more, setMore] = useState(false)
  const [busy, setBusy] = useState<number | null>(null)
  const [error, setError] = useState('')
  const [picked, setPicked] = useState<string[]>([])
  const [drag, setDrag] = useState(false)
  const input = useRef<HTMLInputElement>(null)
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [profile, setProfile] = useState('')
  const [newName, setNewName] = useState('')
  const loadProfiles = () => api.profiles().then(setProfiles).catch(() => {})

  const refresh = () => api.list().then(setSessions).catch(() => {})
  useEffect(() => {
    api.config().then(setCfg).catch(() => setError('Cannot reach the server. Is the backend running?'))
    refresh(); loadProfiles()
    const t = setInterval(refresh, 4000)
    return () => clearInterval(t)
  }, [])

  const set = (patch: Partial<Intake>) => setIntake((i) => ({ ...i, ...patch }))
  const toggle = (list: string[], v: string) => (list.includes(v) ? list.filter((x) => x !== v) : [...list, v])

  const choose = (f: File | undefined) => {
    setError('')
    if (!f || !cfg) return
    const ext = '.' + (f.name.split('.').pop() || '').toLowerCase()
    if (!cfg.extensions.includes(ext)) return setError(`That file type isn't supported. Use ${cfg.extensions.join(', ')}.`)
    if (f.size > cfg.max_upload_mb * 1024 * 1024) return setError(`That file is ${(f.size / 1048576).toFixed(0)} MB; the limit is ${cfg.max_upload_mb} MB. Trim the clip and try again.`)
    setFile(f)
  }

  const submit = async () => {
    if (!file) return
    setBusy(0); setError('')
    try {
      const { id } = await api.upload(file, intake, '', setBusy, profile)
      window.location.hash = `#/s/${id}`
    } catch (e) {
      setError((e as Error).message); setBusy(null)
    }
  }

  const remove = async (s: SessionSummary) => {
    if (!confirm(`Delete "${s.title}"? The video, landmark data, report, comments, drawings, labels and share links will be removed.`)) return
    await api.remove(s.id)
    setPicked((p) => p.filter((x) => x !== s.id))
    refresh()
  }

  return (
    <div className="home">
      <section className="hero">
        <h1>See what your strokes are actually doing.</h1>
        <p>Upload a clip of yourself hitting. You get your forehands, backhands and serves marked on the video, a few things worth working on, and a plain account of what the camera could and couldn't show.</p>
        <div className="sports" role="group" aria-label="Sport">
          {SPORTS.map((s) => (
            <div key={s.id} className={`sport ${s.ready ? 'on' : 'soon'}`} aria-disabled={!s.ready}>
              <span className="sport-icon" aria-hidden>{s.icon}</span>
              <span className="sport-text"><strong>{s.name}</strong><span className="muted small">{s.ready ? s.blurb : 'Coming soon'}</span></span>
              {!s.ready && <span className="tag hollow">Soon</span>}
            </div>
          ))}
        </div>
      </section>

      <div className="grid2">
        <section className="card">
          <h2>New session</h2>
          <div
            className={`drop ${drag ? 'over' : ''} ${file ? 'has' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); choose(e.dataTransfer.files[0]) }}
            onClick={() => input.current?.click()}
            role="button" tabIndex={0}
            onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
          >
            <input ref={input} type="file" accept={cfg?.extensions.join(',') || 'video/*'} hidden onChange={(e) => choose(e.target.files?.[0])} />
            {file ? (
              <><strong>{file.name}</strong><span className="muted">{(file.size / 1048576).toFixed(1)} MB · click to change</span></>
            ) : (
              <><strong>Choose a video</strong><span className="muted">or drop it here{cfg ? ` · up to ${cfg.max_upload_mb} MB and ${Math.round(cfg.max_duration_s / 60)} minutes` : ''}</span></>
            )}
          </div>

          <p className="muted small">Or <a href="#/capture">record with this device's camera</a>, with framing checks before you start.</p>
          {profiles.length > 0 && (
            <label>Player profile (optional)
              <select value={profile} onChange={(e) => { setProfile(e.target.value); const p = profiles.find((x) => x.id === e.target.value); if (p) set({ handedness: p.handedness, backhand_style: p.backhand_style, skill_level: p.skill_level }) }}>
                <option value="">None</option>{profiles.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
              </select>
            </label>
          )}
          <Choice label="Playing hand" value={intake.handedness} onChange={(v) => set({ handedness: v })} options={[['right', 'Right'], ['left', 'Left']]} />
          <Choice label="Level" value={intake.skill_level} onChange={(v) => set({ skill_level: v })} options={[['beginner', 'Beginner'], ['intermediate', 'Intermediate'], ['advanced', 'Advanced']]} />
          <Choice label="Backhand" value={intake.backhand_style} onChange={(v) => set({ backhand_style: v })} options={[['two_handed', 'Two-handed'], ['one_handed', 'One-handed']]} />
          <fieldset className="choice">
            <legend>What do you want to work on?</legend>
            <div className="chips">
              {GOALS.map((g) => (
                <button type="button" key={g} className={`chip ${intake.goals.includes(g) ? 'on' : ''}`} aria-pressed={intake.goals.includes(g)} onClick={() => set({ goals: toggle(intake.goals, g) })}>{cap(g)}</button>
              ))}
            </div>
          </fieldset>

          <button type="button" className="link" onClick={() => setMore(!more)} aria-expanded={more}>{more ? '− Hide' : '+ Add'} optional background (playing load, discomfort)</button>
          {more && (
            <div className="optional">
              <p className="muted small">Optional. This is shown in the report for you, your coach or your clinician. It does not change what is measured, and the app cannot assess pain.</p>
              <label>How often do you play?
                <select value={intake.playing_frequency} onChange={(e) => set({ playing_frequency: e.target.value })}>
                  <option value="">Prefer not to say</option><option value="less_than_weekly">Less than weekly</option><option value="1_2_per_week">1–2 times a week</option>
                  <option value="3_4_per_week">3–4 times a week</option><option value="5_plus_per_week">5 or more times a week</option>
                </select>
              </label>
              <label>Has that changed in the last month?
                <select value={intake.workload_change} onChange={(e) => set({ workload_change: e.target.value })}>
                  <option value="">Prefer not to say</option><option value="none">About the same</option><option value="increased">Playing more</option><option value="decreased">Playing less</option>
                </select>
              </label>
              <label className="check"><input type="checkbox" checked={intake.discomfort.current} onChange={(e) => set({ discomfort: { ...intake.discomfort, current: e.target.checked } })} /> I currently have pain or discomfort when playing</label>
              {intake.discomfort.current && (
                <>
                  <div className="chips">
                    {REGIONS.map((r) => (
                      <button type="button" key={r} className={`chip ${intake.discomfort.regions.includes(r) ? 'on' : ''}`} aria-pressed={intake.discomfort.regions.includes(r)}
                        onClick={() => set({ discomfort: { ...intake.discomfort, regions: toggle(intake.discomfort.regions, r) } })}>{cap(r)}</button>
                    ))}
                  </div>
                  <label>Anything you want noted (optional)
                    <textarea maxLength={500} rows={2} value={intake.discomfort.notes} onChange={(e) => set({ discomfort: { ...intake.discomfort, notes: e.target.value } })} />
                  </label>
                </>
              )}
              <label>Past injuries relevant to tennis (optional)
                <textarea maxLength={500} rows={2} value={intake.injury_history} onChange={(e) => set({ injury_history: e.target.value })} />
              </label>
            </div>
          )}

          {error && <p className="alert" role="alert">{error}</p>}
          {busy !== null ? (
            <div className="progress" aria-label="Uploading"><div style={{ width: `${Math.round(busy * 100)}%` }} /><span>Uploading… {Math.round(busy * 100)}%</span></div>
          ) : (
            <button className="btn primary wide" disabled={!file} onClick={submit}>Upload and analyse</button>
          )}
          <p className="muted small">Your video stays on this server, is not used for training, and is deleted {cfg && cfg.retention_days > 0 ? `after ${cfg.retention_days} days` : 'when you delete it'}. <a href="#/privacy">Details</a></p>
        </section>

        <section className="card guide">
          <h2>Filming for a useful result</h2>
          <ul className="ticks">
            <li><strong>Whole body in frame</strong>, head to feet, for the entire swing. The player should fill at least a third of the picture height.</li>
            <li><strong>Camera still</strong> on a tripod or propped on the fence, at about hip height.</li>
            <li><strong>Side-on</strong> shows knee bend and the swing path best. <strong>From behind or in front</strong> is needed for sideways trunk tilt and knee-over-foot checks. One clip cannot do both.</li>
            <li><strong>60 fps</strong> if your phone offers it, in daylight or full court lighting.</li>
            <li><strong>One player</strong> is simplest. If others are in shot you'll be asked which one to follow.</li>
            <li>A handful of the same stroke (five serves, say) lets the report separate a habit from a one-off.</li>
          </ul>
          <h3>What this version looks at</h3>
          <p className="muted">Forehands, backhands and serves: leg loading, trunk tilt, follow-through, and knee position when the view allows. The ball is tracked where it can be seen, but contact point, wrist position and footwork are not validated and are not used for findings.</p>
        </section>
      </div>

      <section className="card">
        <div className="row between">
          <h2>Your sessions</h2>
          <div className="row">
            {picked.length === 2 && <a className="btn" href={`#/compare/${picked[0]}/${picked[1]}`}>Compare numbers</a>}
            {picked.length >= 1 && <a className="btn" href={`#/duo/${picked[0]}/${picked[1] || picked[0]}`}>Strokes side by side</a>}
          </div>
        </div>
        {sessions.length === 0 ? (
          <p className="muted">Nothing here yet. Sessions you upload from this browser appear here.</p>
        ) : (
          <ul className="sessions">
            {sessions.map((s) => (
              <li key={s.id}>
                <label className="pick" title="Select two finished sessions to compare">
                  <input type="checkbox" disabled={s.status !== 'complete' || (picked.length >= 2 && !picked.includes(s.id))} checked={picked.includes(s.id)} onChange={() => setPicked(toggle(picked, s.id))} />
                  <span className="sr">Select {s.title} for comparison</span>
                </label>
                <a href={`#/s/${s.id}`} className="s-main">
                  <strong>{s.title}</strong>
                  <span className="muted">{fmtDate(s.created_at)}{s.duration_s ? ` · ${Math.round(s.duration_s)} s` : ''}{s.summary ? ` · ${s.summary}` : ''}</span>
                </a>
                <span className={`tag st-${s.status}`}>{STATUS[s.status] || s.status}</span>
                <button className="icon" onClick={() => remove(s)} aria-label={`Delete ${s.title}`} title="Delete">✕</button>
              </li>
            ))}
          </ul>
        )}
        {cfg?.demo_enabled && <p className="muted small"><button className="link" onClick={() => api.demo().then(({ id }) => { window.location.hash = `#/s/${id}` })}>Create a demo session</button> using a synthetic stick figure (labelled as demo; not real analysis).</p>}
        {sessions.some((s) => s.status === 'complete') && <p className="muted small">Tick two finished sessions to compare their numbers or replay two strokes side by side; tick one to compare two strokes from the same session.</p>}
      </section>

      <section className="card">
        <h2>Player profiles</h2>
        <p className="muted small">For coaches with several players. Profiles group sessions for the <a href="#/progress">progress view</a>. They are kept in this browser only: there are no accounts, so they do not follow you to another device.</p>
        <ul className="sessions">
          {profiles.map((p) => (
            <li key={p.id}><span className="s-main"><strong>{p.name}</strong><span className="muted">{cap(p.handedness)}-handed · {cap(p.backhand_style)} backhand · {p.sessions} session{p.sessions === 1 ? '' : 's'}</span></span>
              <button className="icon" aria-label={`Delete profile ${p.name}`} title="Delete profile (its sessions are kept)" onClick={() => confirm(`Delete the profile "${p.name}"? Its sessions are kept, unassigned.`) && api.delProfile(p.id).then(() => { loadProfiles(); refresh() })}>✕</button></li>
          ))}
        </ul>
        <div className="row">
          <input className="inline" placeholder="Player name" maxLength={60} value={newName} onChange={(e) => setNewName(e.target.value)} />
          <button className="btn" disabled={!newName.trim()} onClick={() => api.addProfile({ name: newName, handedness: intake.handedness, backhand_style: intake.backhand_style, skill_level: intake.skill_level }).then(() => { setNewName(''); loadProfiles() })}>Add profile</button>
          <span className="muted small">Uses the hand, level and backhand chosen above.</span>
        </div>
      </section>
    </div>
  )
}
