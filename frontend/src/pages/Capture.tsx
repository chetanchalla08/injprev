import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import type { Config, Framing, Intake } from '../api'

const BASE: Intake = { handedness: 'right', skill_level: 'intermediate', backhand_style: 'two_handed', goals: [], playing_frequency: '', workload_change: '', discomfort: { current: false, regions: [], notes: '' }, injury_history: '' }

export default function Capture() {
  const video = useRef<HTMLVideoElement>(null)
  const stream = useRef<MediaStream | null>(null)
  const rec = useRef<MediaRecorder | null>(null)
  const chunks = useRef<Blob[]>([])
  const prev = useRef<Uint8ClampedArray | null>(null)
  const [cfg, setCfg] = useState<Config | null>(null)
  const [error, setError] = useState('')
  const [f, setF] = useState<Framing | null>(null)
  const [steady, setSteady] = useState<boolean | null>(null)
  const [recording, setRecording] = useState(false)
  const [secs, setSecs] = useState(0)
  const [busy, setBusy] = useState<number | null>(null)
  const [intake, setIntake] = useState(BASE)
  const [facing, setFacing] = useState<'environment' | 'user'>('environment')

  useEffect(() => { api.config().then(setCfg).catch(() => {}) }, [])
  useEffect(() => {
    let dead = false
    if (!navigator.mediaDevices?.getUserMedia) { setError('This browser cannot use the camera here. Camera access needs a secure (https) page or localhost. You can still record with your camera app and upload the file.'); return }
    navigator.mediaDevices.getUserMedia({ video: { facingMode: facing, width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 60 } }, audio: false })
      .then((s) => { if (dead) return s.getTracks().forEach((t) => t.stop()); stream.current = s; if (video.current) { video.current.srcObject = s; video.current.play().catch(() => {}) } })
      .catch(() => setError('Camera access was refused or no camera was found. Allow the camera for this site, or record with your camera app and upload the file.'))
    return () => { dead = true; stream.current?.getTracks().forEach((t) => t.stop()); stream.current = null }
  }, [facing])

  // framing guidance: steadiness is judged here from consecutive frames; the rest from a snapshot checked on the server (not stored)
  useEffect(() => {
    if (recording) return
    const c = document.createElement('canvas')
    const tiny = document.createElement('canvas'); tiny.width = 64; tiny.height = 36
    const fast = setInterval(() => {
      const v = video.current
      if (!v || !v.videoWidth) return
      const g = tiny.getContext('2d', { willReadFrequently: true })!
      g.drawImage(v, 0, 0, 64, 36)
      const d = g.getImageData(0, 0, 64, 36).data
      if (prev.current) { let sum = 0; for (let i = 0; i < d.length; i += 4) sum += Math.abs(d[i + 1] - prev.current[i + 1]); setSteady(sum / (64 * 36) < 9) }
      prev.current = d
    }, 400)
    const slow = setInterval(() => {
      const v = video.current
      if (!v || !v.videoWidth) return
      c.width = 480; c.height = Math.round((480 * v.videoHeight) / v.videoWidth)
      c.getContext('2d')!.drawImage(v, 0, 0, c.width, c.height)
      c.toBlob((b) => b && api.framing(b).then(setF).catch(() => {}), 'image/jpeg', 0.7)
    }, 1500)
    return () => { clearInterval(fast); clearInterval(slow) }
  }, [recording])

  useEffect(() => {
    if (!recording) return
    const t = setInterval(() => setSecs((s) => { if (cfg && s + 1 >= cfg.max_duration_s) stop(); return s + 1 }), 1000)
    return () => clearInterval(t)
  }, [recording, cfg]) // eslint-disable-line

  const start = () => {
    if (!stream.current) return
    const type = ['video/mp4', 'video/webm;codecs=vp9', 'video/webm'].find((t) => MediaRecorder.isTypeSupported(t)) || ''
    chunks.current = []
    const r = new MediaRecorder(stream.current, { mimeType: type || undefined, videoBitsPerSecond: 8_000_000 })
    r.ondataavailable = (e) => e.data.size && chunks.current.push(e.data)
    r.onstop = async () => {
      const ext = r.mimeType.includes('mp4') ? 'mp4' : 'webm'
      const file = new File(chunks.current, `capture-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-')}.${ext}`, { type: r.mimeType })
      if (cfg && file.size > cfg.max_upload_mb * 1048576) return setError(`The recording is larger than the ${cfg.max_upload_mb} MB limit. Record a shorter clip.`)
      setBusy(0)
      try { const { id } = await api.upload(file, intake, '', setBusy); window.location.hash = `#/s/${id}` } catch (e) { setError((e as Error).message); setBusy(null) }
    }
    rec.current = r; r.start(1000); setSecs(0); setRecording(true)
  }
  const stop = () => { rec.current?.state === 'recording' && rec.current.stop(); setRecording(false) }
  const checks = [...(f?.checks || []), { key: 'steady', ok: steady, message: steady === null ? 'Checking steadiness…' : steady ? 'Camera is steady.' : 'The camera is moving. Put it on a tripod or prop it against the fence.' }]
  const ready = !!f?.ready && steady === true

  return (
    <div className="narrow wide">
      <h1>Record with this device</h1>
      {error && <p className="alert" role="alert">{error} <a href="#/">Upload a file instead</a></p>}
      <div className="grid2">
        <div>
          <div className="stage"><video ref={video} muted playsInline />{recording && <span className="rec">● REC {Math.floor(secs / 60)}:{String(secs % 60).padStart(2, '0')}</span>}</div>
          <div className="controls">
            {busy !== null ? <div className="progress grow" aria-label="Uploading"><div style={{ width: `${Math.round(busy * 100)}%` }} /><span>Uploading… {Math.round(busy * 100)}%</span></div>
              : recording ? <button className="btn primary" onClick={stop}>Stop and analyse</button>
                : <><button className="btn primary" disabled={!stream.current && !f} onClick={start}>{ready ? 'Start recording' : 'Record anyway'}</button>
                  <button className="btn ghost" onClick={() => setFacing(facing === 'user' ? 'environment' : 'user')}>Switch camera</button></>}
          </div>
        </div>
        <div className="card">
          <h3>Before you record</h3>
          <ul className="checks">{checks.map((c) => <li key={c.key} className={c.ok === true ? 'ok' : c.ok === false ? 'bad' : 'wait'}>{c.message}</li>)}</ul>
          {!f && !error && <p className="muted small">Waiting for the first check…</p>}
          <p className="muted small">While you set up, a small snapshot is sent to this server about every second and a half to check the framing. Snapshots are looked at and discarded, never stored. Checking stops when recording starts. The recording itself then goes through the normal upload and analysis.</p>
          <fieldset className="choice"><legend>Playing hand</legend><div className="seg">{[['right', 'Right'], ['left', 'Left']].map(([v, t]) => <button type="button" key={v} className={intake.handedness === v ? 'on' : ''} onClick={() => setIntake({ ...intake, handedness: v })}>{t}</button>)}</div></fieldset>
          <fieldset className="choice"><legend>Backhand</legend><div className="seg">{[['two_handed', 'Two-handed'], ['one_handed', 'One-handed']].map(([v, t]) => <button type="button" key={v} className={intake.backhand_style === v ? 'on' : ''} onClick={() => setIntake({ ...intake, backhand_style: v })}>{t}</button>)}</div></fieldset>
          <p className="muted small">Up to {cfg ? Math.round(cfg.max_duration_s / 60) : 3} minutes. Browser recording is usually 30 fps; your phone's own camera app at 60 fps gives a better result if you can upload that instead.</p>
        </div>
      </div>
    </div>
  )
}
