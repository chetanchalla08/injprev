import { useEffect, useState } from 'react'
import { api } from '../api'
import type { Config } from '../api'

export default function Privacy() {
  const [cfg, setCfg] = useState<Config | null>(null)
  const [count, setCount] = useState<number | null>(null)
  const [done, setDone] = useState(false)
  useEffect(() => { api.config().then(setCfg); api.list().then((l) => setCount(l.length)) }, [])
  const wipe = async () => {
    if (!confirm('Delete every session, video, report, note and profile stored for this browser? This cannot be undone.')) return
    await api.removeAll()
    setDone(true); setCount(0)
  }
  return (
    <div className="narrow">
      <h1>Your data</h1>
      <div className="card prose">
        <h3>What is stored</h3>
        <ul>
          <li>A converted copy of your video (no audio). The original file is discarded as soon as the copy is made.</li>
          <li>Body-landmark positions for the player you selected, and the report built from them.</li>
          <li>The details you entered, including anything you chose to say about discomfort or injury history.</li>
          <li>Anything you add: comments, drawings, stroke labels, joint annotations, reviews of findings, player profiles, and a hashed record of each share link.</li>
          <li>If the optional AI rewording is switched on and you ask for it: the reworded text. To make it, the findings (not the video, and not your health details) are sent to the AI provider.</li>
        </ul>
        <h3>Encryption</h3>
        <p>{cfg?.encryption_at_rest ? 'Encryption at rest is on: videos, landmark data, reports and the details and notes you enter are stored encrypted with a key held by the server. This protects against someone obtaining the storage or a backup without the key. It does not protect against someone who controls the running server, and it is not end-to-end encryption.' : 'Encryption at rest is off on this server: files and the database are stored as they are. The operator can turn it on by setting a key.'}</p>
        <h3>How long</h3>
        <p>{cfg && cfg.retention_days > 0 ? `Each session is deleted automatically ${cfg.retention_days} days after upload.` : 'Sessions are kept until you delete them.'} You can delete any session, or everything, at any time. Deletion removes the video, the landmark data, the report and everything attached to the session (comments, drawings, labels, annotations, reviews, AI rewording, share links). PDF reports are generated when you ask and are not kept on the server. “Delete all my data” also removes your player profiles.</p>
        <h3>Who can see it</h3>
        <p>There are no accounts in this version. Sessions are tied to a random key held in a cookie in this browser; without it the server will not return the video or report. If you clear cookies you lose access, and the data is removed when the retention period ends.</p>
        <h3>Share links</h3>
        <p>A share link gives read-only access to one session to anyone who has it, until it expires or you revoke it. The link itself is not stored (only a fingerprint of it), so it cannot be recovered; create a new one if you lose it. Details you entered about discomfort or injury history are never shown through a link.</p>
        <h3>Recording with the camera</h3>
        <p>While you set up a recording, small snapshots are sent to the server to check framing. They are checked and discarded, not stored.</p>
        <h3>What it is not used for</h3>
        <p>Your uploads are not used to train models. The video analysis runs entirely on the server running this app. {cfg?.llm_summary ? 'The optional AI rewording, only when you press its button, sends the written findings to an outside AI provider.' : 'Nothing is sent to an outside service.'}</p>
        <h3>Reviewer sign-offs</h3>
        <p>If a coach or clinician signs off wording in reviewer mode, their name, role and registration are stored and shown next to what they approved. Those records are not removed when sessions are deleted.</p>
        <h3>What this is not</h3>
        <p>These are sensible precautions, not a certification. This app makes no claim of compliance with HIPAA, GDPR or any other regulation.</p>
      </div>
      <div className="card">
        <h3>Delete everything</h3>
        <p className="muted">{count === null ? '' : `${count} session${count === 1 ? '' : 's'} stored for this browser.`}</p>
        {done ? <p>All data for this browser has been deleted.</p> : <button className="btn danger" onClick={wipe} disabled={!count}>Delete all my data</button>}
      </div>
    </div>
  )
}
