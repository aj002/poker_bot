import { useEffect, useRef, useState } from 'react'
import {
  ApiFailure,
  api,
  loadCredentials,
  saveCredentials,
  setOperatorToken,
  useOperatorToken,
  useTableStream,
  type Credentials,
} from './api'
import { AdvisorScreen } from './screens/Advisor'
import { LearnScreen } from './screens/Learn'
import { Lobby } from './screens/Lobby'
import { ReplayScreen } from './screens/Replay'
import { Results } from './screens/Results'
import { RunReport } from './screens/RunReport'
import { RunsScreen } from './screens/Runs'
import { Setup, formatFailure } from './screens/Setup'
import { StudyScreen } from './screens/Study'
import { TableScreen } from './screens/TableScreen'

function useHashRoute(): [string, (path: string) => void] {
  const read = () => location.hash.replace(/^#/, '') || '/'
  const [route, setRoute] = useState(read)
  useEffect(() => {
    const onChange = () => setRoute(read())
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return [route, (path: string) => (location.hash = path)]
}

function OperatorDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null)
  const input = useRef<HTMLInputElement>(null)
  const current = useOperatorToken()
  const [value, setValue] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    const d = ref.current
    if (!d) return
    if (open && !d.open) {
      setValue('')
      setError(null)
      d.showModal()
      input.current?.focus()
    } else if (!open && d.open) d.close()
  }, [open])

  const save = async (e: React.FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await api.whoami(value.trim())
      setOperatorToken(value.trim())
      onClose()
    } catch (err) {
      setError(err instanceof ApiFailure && err.status !== 0 ? 'That is not this server’s operator token.' : 'Could not reach the server.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <dialog
      ref={ref}
      className="dialog"
      // `close` is dispatched a task after Escape; by then the dialog may have been reopened.
      onCancel={onClose}
      onClose={() => !ref.current?.open && onClose()}
      aria-labelledby="operator-title"
    >
      <form onSubmit={save}>
        <h2 id="operator-title" className="screen-title">
          Operator credential
        </h2>
        <p className="lede">
          Unlocks registered bot files, seating them and batch runs. <code>make dev</code> prints the token; it is also in{' '}
          <code>artifacts/operator-token</code>. Stored in this browser only.
        </p>
        {current ? (
          <p className="notice">This browser holds an operator credential.</p>
        ) : null}
        <label className="field">
          <span className="label">Token</span>
          <input
            className="text mono"
            type="password"
            autoComplete="off"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            aria-label="Operator token"
            ref={input}
          />
        </label>
        {error && <p className="error">{error}</p>}
        <div className="dialog-actions">
          {current && (
            <button
              type="button"
              className="btn"
              onClick={() => {
                setOperatorToken(null)
                onClose()
              }}
            >
              Sign out
            </button>
          )}
          <button type="button" className="btn" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" className="btn btn-primary" disabled={!value.trim() || busy}>
            {busy ? 'Checking…' : 'Save'}
          </button>
        </div>
      </form>
    </dialog>
  )
}

function Header({ route, navigate }: { route: string; navigate: (p: string) => void }) {
  const operator = useOperatorToken()
  const [dialog, setDialog] = useState(false)
  return (
    <header className="site-header">
      <button className="brand" onClick={() => navigate('/')}>
        <span className="brand-mark" aria-hidden="true" />
        <span className="brand-name">Practice Arena</span>
      </button>
      <nav className="site-nav" aria-label="Main">
        <a href="#/" aria-current={route === '/' ? 'page' : undefined}>
          New table
        </a>
        <a href="#/advisor" aria-current={route === '/advisor' ? 'page' : undefined}>
          Advisor
        </a>
        <a href="#/learn" aria-current={route === '/learn' ? 'page' : undefined}>
          Learn
        </a>
        <a href="#/learn/pluribus" aria-current={route.startsWith('/learn/pluribus') ? 'page' : undefined}>
          Study Pluribus
        </a>
        <a href="#/runs" aria-current={route.startsWith('/runs') ? 'page' : undefined}>
          Batch runs
        </a>
        <button className={`btn btn-small${operator ? ' btn-operator-on' : ''}`} onClick={() => setDialog(true)}>
          {operator ? 'Operator ✓' : 'Operator'}
        </button>
      </nav>
      <span className="practice-label" data-testid="practice-label">
        Practice assumptions — official kit not yet validated
      </span>
      <OperatorDialog open={dialog} onClose={() => setDialog(false)} />
    </header>
  )
}

function JoinSeat({ tableId, seat, navigate }: { tableId: string; seat: number; navigate: (p: string) => void }) {
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    const existing = loadCredentials(tableId)
    if (existing.seatToken) {
      navigate(`/t/${tableId}`)
      return
    }
    api.join(tableId, seat).then(
      (joined) => {
        saveCredentials(tableId, { ...existing, seat: joined.seat, seatToken: joined.seat_token })
        navigate(`/t/${tableId}`)
      },
      (err) => setError(err instanceof ApiFailure ? formatFailure(err) : 'Could not reach the server.'),
    )
  }, [tableId, seat])
  return (
    <main className="panel narrow">
      {error ? (
        <>
          <p className="error">{error}</p>
          <a className="btn" href={`#/t/${tableId}`}>
            Watch the table instead
          </a>
        </>
      ) : (
        <p>Claiming seat {seat}…</p>
      )}
    </main>
  )
}

function errorTitle(error: ApiFailure): string {
  if (error.status === 404) return 'Table not found'
  if (error.code === 'table_halted') return 'Table halted'
  if (error.code === 'storage_unavailable') return 'Table paused'
  return 'Access denied'
}

function errorText(error: ApiFailure): string {
  if (error.status === 404) return 'No table with this link exists here. In-memory mode does not keep tables across a server restart.'
  if (error.code === 'table_halted')
    return 'Its stored record could not be reproduced exactly on recovery, so the table was stopped for diagnosis instead of being repaired.'
  if (error.code === 'storage_unavailable')
    return 'Storage is unavailable. Think time is paused, and the table resumes here automatically when storage is back.'
  return `${error.code}: ${error.message}`
}

function TableRoute({ tableId, navigate }: { tableId: string; navigate: (p: string) => void }) {
  const [creds, setCreds] = useState<Credentials>(() => loadCredentials(tableId))
  const token = creds.seatToken ?? null
  const { snapshot, status, error, accept, connections } = useTableStream(tableId, token)

  if (error && !snapshot) {
    return (
      <main className="panel narrow">
        <h1 className="screen-title">{errorTitle(error)}</h1>
        <p className="lede">{errorText(error)}</p>
        <button className="btn btn-primary" onClick={() => navigate('/')}>
          Set up a new table
        </button>
      </main>
    )
  }
  if (!snapshot) {
    return (
      <main className="panel narrow" aria-busy="true">
        <p className="muted">Connecting to table…</p>
      </main>
    )
  }
  if (snapshot.status === 'waiting') return <Lobby snapshot={snapshot} creds={creds} onCreds={setCreds} />
  if (snapshot.status === 'complete') return <Results snapshot={snapshot} creds={creds} navigate={navigate} />
  return <TableScreen snapshot={snapshot} token={token} stream={status} connections={connections} onSnapshot={accept} />
}

export default function App() {
  const [route, navigate] = useHashRoute()
  const join = route.match(/^\/t\/([\w-]+)\/join\/(\d+)$/)
  const table = route.match(/^\/t\/([\w-]+)$/)
  const replay = route.match(/^\/t\/([\w-]+)\/replay$/)
  const run = route.match(/^\/runs\/([\w-]+)$/)
  return (
    <div className="app">
      <Header route={route} navigate={navigate} />
      {join ? (
        <JoinSeat key={route} tableId={join[1]} seat={Number(join[2])} navigate={navigate} />
      ) : table ? (
        <TableRoute key={table[1]} tableId={table[1]} navigate={navigate} />
      ) : replay ? (
        <ReplayScreen key={replay[1]} tableId={replay[1]} navigate={navigate} />
      ) : run ? (
        <RunReport key={run[1]} runId={run[1]} />
      ) : route === '/learn/pluribus' || route.startsWith('/learn/pluribus/') ? (
        <StudyScreen route={route} navigate={navigate} />
      ) : route === '/learn' ? (
        <LearnScreen navigate={navigate} />
      ) : route === '/advisor' ? (
        <AdvisorScreen />
      ) : route === '/runs' ? (
        <RunsScreen navigate={navigate} />
      ) : (
        <Setup navigate={navigate} />
      )}
    </div>
  )
}
