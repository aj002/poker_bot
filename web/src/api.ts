import { useEffect, useRef, useState } from 'react'
import type {
  Action,
  AdvisorOp,
  AdvisorSetup,
  AdvisorView,
  Aids,
  Perspective,
  Policy,
  Replay,
  ResolvedConfig,
  RunBrief,
  RunView,
  Snapshot,
  StudyFilter,
  StudyHand,
  StudyHandPage,
  StudyPack,
  StyleMode,
  StylesCatalog,
} from './types'

const BASE = '/api'

export class ApiFailure extends Error {
  status: number
  code: string
  details: unknown
  constructor(status: number, code: string, message: string, details?: unknown) {
    super(message)
    this.status = status
    this.code = code
    this.details = details
  }
}

async function request<T>(path: string, init: RequestInit & { token?: string | null } = {}): Promise<T> {
  const headers: Record<string, string> = { 'content-type': 'application/json' }
  if (init.token) headers.authorization = `Bearer ${init.token}`
  const res = await fetch(BASE + path, { ...init, headers })
  const body = await res.json().catch(() => ({}))
  if (!res.ok) {
    const err = body?.error ?? {}
    throw new ApiFailure(res.status, err.code ?? 'http_error', err.message ?? res.statusText, err.details)
  }
  return body as T
}

export interface Credentials {
  seat?: number
  seatToken?: string
  hostToken?: string
}

// Session storage is per tab, so two tabs can hold two different seats.
export function loadCredentials(tableId: string): Credentials {
  try {
    return JSON.parse(sessionStorage.getItem(`arena:${tableId}`) ?? '{}')
  } catch {
    return {}
  }
}

export function saveCredentials(tableId: string, creds: Credentials) {
  sessionStorage.setItem(`arena:${tableId}`, JSON.stringify(creds))
}

const OPERATOR_KEY = 'arena:operator'
const OPERATOR_EVENT = 'arena:operator-changed'

// The operator credential is per browser profile (local storage), unlike seat keys.
export function getOperatorToken(): string | null {
  try {
    return localStorage.getItem(OPERATOR_KEY)
  } catch {
    return null
  }
}

export function setOperatorToken(token: string | null) {
  if (token) localStorage.setItem(OPERATOR_KEY, token)
  else localStorage.removeItem(OPERATOR_KEY)
  window.dispatchEvent(new Event(OPERATOR_EVENT))
}

export function useOperatorToken(): string | null {
  const [token, setToken] = useState(getOperatorToken)
  useEffect(() => {
    const update = () => setToken(getOperatorToken())
    window.addEventListener(OPERATOR_EVENT, update)
    window.addEventListener('storage', update)
    return () => {
      window.removeEventListener(OPERATOR_EVENT, update)
      window.removeEventListener('storage', update)
    }
  }, [])
  return token
}

export interface PendingSubmission {
  seat: number
  label: string
  envelope: ActionEnvelope
}

// A submission whose outcome the tab has not seen yet. It survives reloads so a
// reconnecting tab can resend the same action_id and learn whether it committed.
export function loadPending(tableId: string): PendingSubmission | null {
  try {
    return JSON.parse(sessionStorage.getItem(`arena:${tableId}:pending`) ?? 'null')
  } catch {
    return null
  }
}

export function savePending(tableId: string, pending: PendingSubmission | null) {
  if (pending) sessionStorage.setItem(`arena:${tableId}:pending`, JSON.stringify(pending))
  else sessionStorage.removeItem(`arena:${tableId}:pending`)
}

export interface CreateBody {
  config: Record<string, unknown>
  seats: { kind: 'human' | 'bot'; policy?: string | null; reveal?: boolean }[]
  styles?: { mode: StyleMode; learning_aids: boolean; random_pool?: string[] }
}

export const advisorApi = {
  create: (body: Omit<AdvisorSetup, 'payouts' | 'ante_posting'> & { payouts?: string | number[]; ante_posting?: string | null; button?: number }) =>
    request<AdvisorView>('/advisor/sessions', { method: 'POST', body: JSON.stringify(body) }),
  get: (id: string) => request<AdvisorView>(`/advisor/sessions/${id}`),
  do: (id: string, op: AdvisorOp) =>
    request<AdvisorView>(`/advisor/sessions/${id}/do`, { method: 'POST', body: JSON.stringify(op) }),
}

export const api = {
  catalog: (operator?: string | null) =>
    request<{ rulesets: string[]; policies: Policy[]; operator: boolean }>('/catalog', { token: operator }),
  resolve: (body: CreateBody, operator?: string | null) =>
    request<{ config: ResolvedConfig; config_hash: string }>('/configs/resolve', {
      method: 'POST',
      body: JSON.stringify(body),
      token: operator,
    }),
  create: (body: CreateBody, operator?: string | null) =>
    request<{ table_id: string; host_token: string; snapshot: Snapshot }>('/tables', {
      method: 'POST',
      body: JSON.stringify(body),
      token: operator,
    }),
  join: (tableId: string, seat?: number) =>
    request<{ seat: number; seat_token: string; snapshot: Snapshot }>(`/tables/${tableId}/join`, {
      method: 'POST',
      body: JSON.stringify(seat === undefined ? {} : { seat }),
    }),
  start: (tableId: string, hostToken: string) =>
    request<Snapshot>(`/tables/${tableId}/start`, { method: 'POST', token: hostToken }),
  snapshot: (tableId: string, token?: string | null) => request<Snapshot>(`/tables/${tableId}`, { token }),
  history: (tableId: string, token?: string | null, perspective?: Perspective) =>
    request<unknown>(`/tables/${tableId}/history${perspective === 'reviewer' ? '?perspective=reviewer' : ''}`, { token }),
  replay: (tableId: string, token: string | null, perspective: Perspective, hand?: number) =>
    request<Replay>(
      `/tables/${tableId}/replay?perspective=${perspective}${hand !== undefined ? `&hand=${hand}` : ''}`,
      { token },
    ),
  whoami: (operator: string) => request<{ operator: true }>('/operator', { token: operator }),
  registerBot: (operator: string, path: string) =>
    request<{ bot: Policy }>('/bots', { method: 'POST', token: operator, body: JSON.stringify({ path }) }),
  createRun: (operator: string, body: RunBody) =>
    request<RunView>('/runs', { method: 'POST', token: operator, body: JSON.stringify(body) }),
  runs: (operator: string) => request<{ runs: RunBrief[] }>('/runs', { token: operator }),
  run: (operator: string, runId: string) => request<RunView>(`/runs/${runId}`, { token: operator }),
  act: (tableId: string, token: string, envelope: ActionEnvelope) =>
    request<{ receipt: unknown; replayed: boolean; snapshot: Snapshot }>(`/tables/${tableId}/actions`, {
      method: 'POST',
      token,
      body: JSON.stringify(envelope),
    }),
  styles: () => request<StylesCatalog>('/styles'),
  aids: (tableId: string, seatToken: string) => request<Aids>(`/tables/${tableId}/aids`, { token: seatToken }),
  studyPacks: () => request<{ packs: StudyPack[] }>('/study/packs'),
  studyHands: (pack: string, filter: StudyFilter, offset: number, limit: number) =>
    request<StudyHandPage>(
      `/study/packs/${encodeURIComponent(pack)}/hands?filter=${filter}&offset=${offset}&limit=${limit}`,
    ),
  studyHand: (pack: string, hand: string) =>
    request<StudyHand>(`/study/packs/${encodeURIComponent(pack)}/hands/${encodeURIComponent(hand)}`),
}

/** Creates a table; when it has a human seat, this tab claims the first one. */
export async function createAndJoin(body: CreateBody, operator?: string | null): Promise<string> {
  const created = await api.create(body, operator)
  const creds: Credentials = { hostToken: created.host_token }
  if (body.seats.some((s) => s.kind === 'human')) {
    const joined = await api.join(created.table_id)
    creds.seat = joined.seat
    creds.seatToken = joined.seat_token
  }
  saveCredentials(created.table_id, creds)
  return created.table_id
}

let stylesCache: Promise<StylesCatalog> | null = null

/** Style profiles, opening charts and the starting-hand grid: the server holds the only copy. */
export function useStyles(): StylesCatalog | null {
  const [catalog, setCatalog] = useState<StylesCatalog | null>(null)
  useEffect(() => {
    let live = true
    stylesCache ??= api.styles().catch((err) => {
      stylesCache = null
      throw err
    })
    stylesCache.then(
      (c) => live && setCatalog(c),
      () => undefined,
    )
    return () => {
      live = false
    }
  }, [])
  return catalog
}

/** Learning aids for this seat's current spot, refreshed as the table moves. Stale results are dropped. */
export function useAids(tableId: string, seatToken: string | null, version: number, enabled: boolean) {
  const [aids, setAids] = useState<Aids | null>(null)
  const [failed, setFailed] = useState(false)
  const latest = useRef(-1)
  useEffect(() => {
    if (!enabled || !seatToken) return
    const asked = version
    const t = window.setTimeout(() => {
      api.aids(tableId, seatToken).then(
        (a) => {
          if (asked < latest.current) return
          latest.current = asked
          setAids(a)
          setFailed(false)
        },
        () => setFailed(true),
      )
    }, 60)
    return () => window.clearTimeout(t)
  }, [tableId, seatToken, version, enabled])
  return { aids, failed }
}

export interface RunBody {
  cells: { ruleset_id: string; players: number }[]
  bots: string[]
  blocks: number
  isolation: 'sandbox' | 'in_process'
  jobs: number
  seed?: string
  config: Record<string, unknown>
}

export interface ActionEnvelope {
  action_id: string
  table_id: string
  hand_id: string
  decision_id: string
  expected_state_version: number
  action: Action
}

export function newActionId(): string {
  return `a_${crypto.randomUUID().replace(/-/g, '')}`
}

// Retries reuse the same action_id, so an ambiguous submission is resolved by
// the server's idempotency registry rather than applied twice.
/** Server-side pauses worth retrying with the same action_id (the server deduplicates it). */
const TRANSIENT = new Set(['storage_unavailable'])

export const isTransient = (err: unknown) => err instanceof ApiFailure && TRANSIENT.has(err.code)

export async function submitWithRetry(tableId: string, token: string, envelope: ActionEnvelope, attempts = 3) {
  let lastError: unknown
  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      return await api.act(tableId, token, envelope)
    } catch (err) {
      if (err instanceof ApiFailure && !isTransient(err)) throw err
      lastError = err
      await new Promise((r) => setTimeout(r, 300 * (attempt + 1)))
    }
  }
  throw lastError
}

export type StreamStatus = 'connecting' | 'live' | 'reconnecting' | 'denied'

/** Keeps the newest committed snapshot for this tab's credential, over WebSocket with HTTP fallback. */
export function useTableStream(tableId: string, token: string | null) {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [status, setStatus] = useState<StreamStatus>('connecting')
  const [error, setError] = useState<ApiFailure | null>(null)
  const [connections, setConnections] = useState(0)
  const versionRef = useRef(-1)

  const accept = (snap: Snapshot) => {
    if (snap.version > versionRef.current) {
      versionRef.current = snap.version
      setSnapshot(snap)
    }
  }

  useEffect(() => {
    versionRef.current = -1
    setSnapshot(null)
    let closed = false
    let ws: WebSocket | null = null
    let retry = 0
    let timer: number | undefined
    let fresh = false

    api.snapshot(tableId, token).then(accept, (err) => err instanceof ApiFailure && setError(err))

    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${location.host}${BASE}/tables/${tableId}/stream`)
      ws.onopen = () => {
        fresh = true
        ws?.send(JSON.stringify({ type: 'auth', token }))
      }
      ws.onmessage = (ev) => {
        const frame = JSON.parse(ev.data)
        if (frame.type === 'snapshot') {
          if (fresh) setConnections((c) => c + 1)
          fresh = false
          retry = 0
          setStatus('live')
          setError(null)
          accept(frame as Snapshot)
        } else if (frame.type === 'error') {
          const code = frame.error?.code ?? 'denied'
          const status = TRANSIENT.has(code) || code === 'table_halted' ? 503 : 403
          setError(new ApiFailure(status, code, frame.error?.message ?? 'denied'))
          if (TRANSIENT.has(code)) {
            setStatus('reconnecting')
          } else {
            setStatus('denied')
            closed = true
          }
        }
      }
      ws.onclose = () => {
        if (closed) return
        setStatus('reconnecting')
        retry = Math.min(retry + 1, 5)
        timer = window.setTimeout(connect, 250 * 2 ** retry)
      }
    }
    connect()
    return () => {
      closed = true
      window.clearTimeout(timer)
      ws?.close()
    }
  }, [tableId, token])

  return { snapshot, status, error, accept, connections }
}
