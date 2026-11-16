export interface Ref { id: string; citation: string; doi: string; population: string; summary: string; limits: string; review_status: string }
export interface Confidence { level: 'high' | 'moderate' | 'low' | 'insufficient'; score: number; explanation: string; factors: { name: string; value: number; note: string }[] }
export interface Occurrence { stroke_id: string; start: number; end: number; time: number | null; value: number }
export interface Finding {
  id: string; rule: string; status: 'observed' | 'not_observed' | 'unable_to_assess'; category: 'technique' | 'health_related'
  stroke_type: string; body_region: string; title: string; observation: string; interpretation: string | null
  evidence: { strength: string; refs: Ref[] } | null
  action: { cue: string | null; recording: string | null; professional: string | null }
  limitations: string[]; occurrences: Occurrence[]; support: { fired: number; assessed: number; total: number }
  recurring: boolean; confidence: Confidence | null; user_context: string | null
  drills?: Drill[]; wording_review?: string
}
export interface Drill { id: string; name: string; focus: string; steps: string[]; caution: string; review_status: string }
export interface Measurement { key: string; label: string; value: number | null; unit: string; method: string; time: number | null; visibility: number | null; assessable: boolean; reason: string; extra: Record<string, number | null> }
export interface Stroke {
  id: string; type: string; start: number; end: number; peak: number; source: 'auto' | 'manual'; label_confidence: number; label_basis: string
  quality: { tracked_frac: number; arm_visibility: number; leg_visibility: number; pelvis_azimuth_median: number | null; assessable: boolean; reason: string }
  measurements: Measurement[]
  timing?: { source: string; label: string; pose_time: number | null; confidence: number | null; basis: string | null }
  phases?: { preparation: number[]; forward_swing: number[]; follow_through: number[]; forward_swing_s: number; boundary: string; note: string } | null
  movement?: { position: number[]; zone: string; recovery: { after_s: number; zone: string; moved_m: number; toward_centre_m: number } | null } | null
}
export interface Segment { id: number; kind: string; status: 'auto' | 'needs_confirmation'; after: string | null; confidence: number; similarity: number | null; box: number[] | null; box_t: number; start: number; end: number; decision?: string | null }
export interface Issue { code: string; message: string; suggestion: string }
export interface Metric { label: string; unit: string; method: string; n: number; median: number; min: number; max: number }
export type StrokeSummary = Record<string, { count: number; assessable: number; metrics: Record<string, Metric> }>
export interface Report {
  disclaimer: string; generated_at: string; handedness: string; validation_status: string
  meta: { fps: number; width: number; height: number; duration_s: number; n_frames: number; demo?: boolean }
  quality: { issues: Issue[]; video: { metrics: Record<string, number | string | null> }; tracking: { tracked_frac: number; player_height_frac: number; leg_visibility: number; pelvis_azimuth_median: number | null; camera_view: string; camera_view_basis?: string } }
  summary: { headline: string; detail: string; counts: Record<string, number>; no_strokes: boolean }
  priorities: { finding_id: string; stroke_type: string; title: string; cue: string }[]
  findings: Finding[]; strokes: Stroke[]; stroke_summary: StrokeSummary
  not_assessed: { topic: string; reason: string }[]
  context_notes: { kind: string; title: string; body: string; refs: Ref[]; review_note: string }[]
  ball?: { status: string; found: number; strokes: number; note: string }
  court?: { status: string; reason: string; lines_supported?: string[] }
  movement?: { distance_m: number; direction_changes: number[]; note: string; validated: boolean } | null
  hops?: { times: number[]; before_stroke_s: Record<string, number | null>; note: string; validated: boolean }
  tracking_events?: { cuts: number[]; segments: Segment[]; reacquire: boolean } | null
  pose_note?: string | null
}
export interface Intake {
  handedness: string; skill_level: string; backhand_style: string; goals: string[]; playing_frequency: string; workload_change: string
  discomfort: { current: boolean; regions: string[]; notes: string }; injury_history: string
}
export interface SessionSummary {
  id: string; title: string; created_at: number; status: string; stage: string; progress: number
  error: { code: string; message: string; suggestion: string } | null; expires_at: number | null; duration_s: number | null; summary: string | null
  profile_id?: string | null; camera_view?: string | null; demo?: boolean
}
export interface Session extends SessionSummary { intake: Intake; report: Report | null; candidates: { box: number[] }[]; selected: number | null; quality: { issues: Issue[] } | null
  read_only?: boolean; shared_until?: number; comments?: Comment[]; drawings?: Drawing[] }
export interface Config { max_upload_mb: number; max_duration_s: number; retention_days: number; extensions: string[]; supported_strokes: string[]; pose_model: string; demo_enabled: boolean
  encryption_at_rest: boolean; reviewer_mode: boolean; llm_summary: boolean; share_ttl_days: number; share_max_ttl_days: number }
export interface Overlay { fps: number; stride: number; points: number[]; frames: (number[] | null)[]; ball?: (number[] | null)[] }
export interface Comment { id: string; t: number; body: string; author: string; created_at: number }
export interface Shape { type: 'line' | 'angle' | 'free'; points: number[][]; color?: string }
export interface Drawing { id: string; t: number; shapes: Shape[]; note: string; author: string }
export interface Profile { id: string; name: string; handedness: string; backhand_style: string; skill_level: string; notes: string; sessions: number }
export interface Share { id: string; created_at: number; expires_at: number; revoked_at: number | null; active: boolean; token?: string; path?: string }
export interface LabelStroke { type: string; start: number; contact: number; end: number; contact_kind: string }
export interface LabelSet { labeller: string; labeller_role: string; expert: boolean; camera_view: string; notes: string; strokes: LabelStroke[]; handedness?: string; backhand_style?: string }
export interface AnnotationSet { annotator: string; frames: { frame: number; joints: Record<string, number[]> }[] }
export interface SeriesEntry { key: string; label: string; note: string; unit: string; visibility: number[]; tracked_frac: number; methods: Record<string, (number | null)[]> }
export interface StrokeSeries { t: number[]; min_visibility: number; series: SeriesEntry[]; method_notes: Record<string, string>; gap_note: string; stroke: { id: string; type: string; start: number; end: number; peak: number }; timing?: Stroke['timing'] }
export interface ReviewStatus { state: string; label: string; signoffs: { id: string; reviewer: string; role: string; credentials: string; decision: string; comment: string; date: string; current: boolean }[] }
export interface ReviewItem { id: string; kind: string; title: string; needs: string; text: Record<string, string | string[]>; hash: string; status: ReviewStatus }
export interface RuleReview { id: string; finding_id: string; stroke_id: string | null; verdict: string; cue_ok: boolean | null; comment: string; reviewer: string; role: string }
export interface Progress { caveat: string; mixed_views: boolean; views: string[]; sessions: { id: string; title: string; created_at: number; camera_view: string; fps: number; stroke_summary: StrokeSummary; observed: { rule: string; title: string; stroke_type: string; fired: number; assessed: number }[] }[] }
export interface Framing { ready: boolean; people: number; height_frac: number | null; checks: { key: string; ok: boolean | null; message: string }[] }
export interface Summary { enabled: boolean; summary: { text: string; model: string; generated_at: string; notice: string } | null }
export interface Comparison {
  comparable: boolean; caveat: string
  sessions: { id: string; title: string; created_at: number; camera_view: string; fps: number; stroke_summary: StrokeSummary; observed: { title: string; stroke_type: string; support: { fired: number; assessed: number } }[] }[]
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, { credentials: 'same-origin', ...init })
  if (!r.ok) {
    let msg = `Request failed (${r.status})`
    try { const j = await r.json(); if (typeof j.detail === 'string') msg = j.detail } catch { /* keep default */ }
    throw new Error(msg)
  }
  return (r.status === 204 ? undefined : await r.json()) as T
}
const json = (method: string, body: unknown): RequestInit => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

export const api = {
  config: () => req<Config>('/api/config'),
  demo: () => req<{ id: string }>('/api/demo', { method: 'POST' }),
  list: () => req<SessionSummary[]>('/api/sessions'),
  get: (id: string) => req<Session>(`/api/sessions/${id}`),
  overlay: (id: string) => req<Overlay>(`/api/sessions/${id}/overlay`),
  choosePlayer: (id: string, candidate: number) => req(`/api/sessions/${id}/player`, json('POST', { candidate })),
  retry: (id: string) => req(`/api/sessions/${id}/retry`, { method: 'POST' }),
  remove: (id: string) => req<void>(`/api/sessions/${id}`, { method: 'DELETE' }),
  removeAll: () => req<void>('/api/me', { method: 'DELETE' }),
  setStrokes: (id: string, strokes: { id?: string; start: number; end: number; type: string }[]) => req<Session>(`/api/sessions/${id}/strokes`, json('PUT', { strokes })),
  compare: (a: string, b: string) => req<Comparison>(`/api/compare?a=${a}&b=${b}`),
  shared: (token: string) => req<Session>(`/api/shared/${token}`),
  overlayAt: (base: string) => req<Overlay>(`${base}/overlay`),
  comments: (id: string) => req<Comment[]>(`/api/sessions/${id}/comments`),
  addComment: (id: string, c: { t: number; body: string; author: string }) => req<Comment>(`/api/sessions/${id}/comments`, json('POST', c)),
  delComment: (id: string, cid: string) => req<void>(`/api/sessions/${id}/comments/${cid}`, { method: 'DELETE' }),
  drawings: (id: string) => req<Drawing[]>(`/api/sessions/${id}/drawings`),
  addDrawing: (id: string, d: { t: number; shapes: Shape[]; note: string; author: string }) => req<Drawing>(`/api/sessions/${id}/drawings`, json('POST', d)),
  delDrawing: (id: string, did: string) => req<void>(`/api/sessions/${id}/drawings/${did}`, { method: 'DELETE' }),
  profiles: () => req<Profile[]>('/api/profiles'),
  addProfile: (p: Partial<Profile>) => req<Profile>('/api/profiles', json('POST', p)),
  delProfile: (id: string) => req<void>(`/api/profiles/${id}`, { method: 'DELETE' }),
  setProfile: (id: string, profile_id: string | null) => req(`/api/sessions/${id}/profile`, json('PUT', { profile_id })),
  progress: (profile?: string) => req<Progress>(`/api/progress${profile ? `?profile_id=${profile}` : ''}`),
  shares: (id: string) => req<Share[]>(`/api/sessions/${id}/shares`),
  addShare: (id: string, days: number) => req<Share>(`/api/sessions/${id}/shares`, json('POST', { days })),
  revokeShare: (id: string, share: string) => req<void>(`/api/sessions/${id}/shares/${share}`, { method: 'DELETE' }),
  labels: (id: string) => req<LabelSet>(`/api/sessions/${id}/labels`),
  setLabels: (id: string, l: LabelSet) => req<LabelSet>(`/api/sessions/${id}/labels`, json('PUT', l)),
  annotations: (id: string) => req<AnnotationSet>(`/api/sessions/${id}/annotations`),
  setAnnotations: (id: string, a: AnnotationSet) => req<AnnotationSet>(`/api/sessions/${id}/annotations`, json('PUT', a)),
  annotationReport: (id: string) => req<Record<string, unknown>>(`/api/sessions/${id}/annotations/report`),
  mocap: (id: string, csv: string, offset: number) => req<Record<string, unknown>>(`/api/sessions/${id}/mocap?offset_s=${offset}`, { method: 'POST', body: csv }),
  series: (id: string, stroke: string) => req<StrokeSeries>(`/api/sessions/${id}/strokes/${stroke}/series`),
  reviewItems: () => req<{ reviewer_mode: boolean; roles: string[]; items: ReviewItem[] }>('/api/review/items'),
  signoff: (b: Record<string, unknown>) => req<{ status: ReviewStatus }>('/api/review/signoff', json('POST', b)),
  withdraw: (id: string, code: string) => req(`/api/review/signoff/${id}/withdraw`, json('POST', { code })),
  reviews: (id: string) => req<RuleReview[]>(`/api/sessions/${id}/reviews`),
  putReview: (id: string, r: Record<string, unknown>) => req<RuleReview>(`/api/sessions/${id}/reviews`, json('PUT', r)),
  decideSegment: (id: string, seg: number, decision: 'confirm' | 'reject') => req<Session>(`/api/sessions/${id}/segments/${seg}`, json('POST', { decision })),
  summary: (id: string) => req<Summary>(`/api/sessions/${id}/summary`),
  makeSummary: (id: string) => req<Summary>(`/api/sessions/${id}/summary`, { method: 'POST' }),
  framing: (jpeg: Blob) => req<Framing>('/api/framing', { method: 'POST', body: jpeg }),
  upload(file: File, intake: Intake, title: string, onProgress: (f: number) => void, profileId = ''): Promise<{ id: string }> {
    return new Promise((resolve, reject) => {
      const fd = new FormData()
      fd.append('file', file)
      fd.append('intake', JSON.stringify(intake))
      fd.append('title', title)
      if (profileId) fd.append('profile_id', profileId)
      const x = new XMLHttpRequest()
      x.open('POST', '/api/sessions')
      x.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total)
      x.onload = () => {
        if (x.status === 201) return resolve(JSON.parse(x.responseText))
        let msg = `Upload failed (${x.status})`
        try { msg = JSON.parse(x.responseText).detail || msg } catch { /* keep default */ }
        reject(new Error(msg))
      }
      x.onerror = () => reject(new Error('The upload was interrupted. Check your connection and try again.'))
      x.send(fd)
    })
  },
}

export const fmtTime = (t: number) => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, '0')}`
export const fmtDate = (ts: number) => new Date(ts * 1000).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
export const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1).replace(/_/g, ' ')
