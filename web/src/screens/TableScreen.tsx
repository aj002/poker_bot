import { useEffect, useState } from 'react'
import { useAids, useStyles, type StreamStatus } from '../api'
import { Card } from '../cards'
import { RULESETS, anteLabel, chips, describeEvents } from '../format'
import { recapLines } from '../learn/HandRecap'
import { LearningPanel } from '../learn/LearningPanel'
import { NotesPanel } from '../learn/NotesPanel'
import { aidsKey, useStored } from '../learn/prefs'
import { isHiddenLive, visibleProfile } from '../learn/styles'
import type { Snapshot, StylesCatalog, TableView } from '../types'
import { ActionPanel } from './ActionPanel'
import { Felt } from './Felt'

function Timer({ snapshot }: { snapshot: Snapshot }) {
  const turn = snapshot.turn
  const [deadline, setDeadline] = useState<number | null>(null)
  const [now, setNow] = useState(() => performance.now())

  useEffect(() => {
    setDeadline(turn?.remaining_ms != null ? performance.now() + turn.remaining_ms : null)
  }, [turn?.decision_id, snapshot.version, turn?.remaining_ms])

  useEffect(() => {
    if (deadline === null) return
    const id = window.setInterval(() => setNow(performance.now()), 200)
    return () => window.clearInterval(id)
  }, [deadline])

  if (!turn || turn.kind !== 'human' || deadline === null || !turn.turn_ms) return null
  const left = Math.max(0, deadline - now)
  const frac = left / turn.turn_ms
  return (
    <span className={`timer${frac < 0.25 ? ' timer-low' : ''}`} role="timer" aria-label={`${Math.ceil(left / 1000)} seconds left`}>
      <span className="timer-bar">
        <span className="timer-fill" style={{ width: `${frac * 100}%` }} />
      </span>
      <span className="mono timer-text">{Math.ceil(left / 1000)}s</span>
    </span>
  )
}

function LastHand({ snapshot, catalog, recap }: { snapshot: Snapshot; catalog: StylesCatalog | null; recap: boolean }) {
  const last = snapshot.last_hand
  if (!last) return null
  const seats = snapshot.setup.seats
  const story = recap
    ? recapLines(last.events, last.hand_id, seats, snapshot.view?.hero_seat ?? null, snapshot.stats, (seat) => {
        const info = seats.find((s) => s.seat === seat)
        return isHiddenLive(info, snapshot) ? null : visibleProfile(info, catalog)
      })
    : []
  const board = last.events.filter((e) => e.type === 'BoardDealt').flatMap((e) => e.data.cards as string[])
  const lines = describeEvents(
    last.events.filter((e) => ['CardsRevealed', 'PotAwarded', 'PlayerEliminated', 'ExcessReturned'].includes(e.type)),
    snapshot.setup.seats,
  )
  return (
    <div className="last-hand" data-testid="last-hand" data-hand-number={last.hand_number ?? undefined}>
      <div className="label">Hand {last.hand_number} result</div>
      {board.length > 0 && (
        <div className="last-board">
          {board.map((c) => (
            <Card key={c} card={c} size="sm" />
          ))}
        </div>
      )}
      <ul className="log-lines">
        {lines.map((l) => (
          <li key={l.seq}>{l.text}</li>
        ))}
      </ul>
      {story.length > 0 && (
        <div className="recap" data-testid="hand-recap">
          <div className="label">What happened</div>
          <ul>
            {story.map((line, i) => (
              <li key={i}>{line}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export function TableScreen({
  snapshot,
  token,
  stream,
  connections,
  onSnapshot,
}: {
  snapshot: Snapshot
  token: string | null
  stream: StreamStatus
  connections: number
  onSnapshot: (s: Snapshot) => void
}) {
  const view = snapshot.view as TableView
  const log = describeEvents(view.action_history, snapshot.setup.seats).slice(-14)
  const timer = <Timer snapshot={snapshot} />
  const level = view.blinds ? Math.floor((view.hand_number - 1) / snapshot.setup.config.hands_per_level) + 1 : null
  const [aidsChoice, setAidsChoice] = useStored<boolean | null>(aidsKey(snapshot.table_id), null)
  const holdem = view.ruleset_id === 'practice_nlhe'
  const aidsAllowed = holdem && (snapshot.setup.styles?.learning_aids ?? true)
  const aidsOn = aidsAllowed && (aidsChoice ?? true)
  const seated = view.hero_seat !== null && !!token
  const { aids, failed } = useAids(snapshot.table_id, token, snapshot.version, aidsOn && seated)
  const catalog = useStyles()
  const positions = aidsOn && aids?.hand_id === view.hand_id ? aids?.positions : undefined

  return (
    <main className="play">
      <div className="hand-strip">
        <span>
          <span className="label">Hand</span>{' '}
          <span className="mono" data-testid="hand-number">
            {view.hand_number}
          </span>
          <span className="muted mono">/{view.hand_cap}</span>
        </span>
        {view.blinds && (
          <span>
            <span className="label">Level {level}</span>{' '}
            <span className="mono">
              {chips(view.blinds.small_blind)}/{chips(view.blinds.big_blind)}
              {view.blinds.ante ? ` · ${anteLabel(view.blinds.ante_posting)} ${chips(view.blinds.ante)}` : ''}
            </span>
          </span>
        )}
        {view.next_level && (
          <span className="muted">
            next {chips(view.next_level.small_blind)}/{chips(view.next_level.big_blind)}
            {view.next_level.ante ? ` · ${anteLabel(view.next_level.ante_posting)} ${chips(view.next_level.ante)}` : ''} at
            hand {view.next_level.starts_at_hand}
          </span>
        )}
        <span className="muted" data-testid="survivors">
          {view.surviving_player_count}/{view.initial_player_count} left · {RULESETS[view.ruleset_id]?.short}
        </span>
        <label className="toggle strip-toggle">
          <input type="checkbox" checked={aidsOn} disabled={!aidsAllowed} onChange={(e) => setAidsChoice(e.target.checked)} />
          <span>{aidsAllowed ? 'Learning aids' : holdem ? 'Learning aids off for this table' : "Learning aids: Hold'em only"}</span>
        </label>
        <a className="strip-link" href={`#/t/${snapshot.table_id}/replay`}>
          Replay so far
        </a>
        <span className={`stream stream-${stream}`} data-testid="stream-status" role="status">
          {stream === 'live' ? 'live' : stream}
        </span>
      </div>

      <div className="play-grid">
        <Felt view={view} seats={snapshot.setup.seats} timer={timer} seatLabels={positions} stats={snapshot.stats?.seats} />
        <aside className="log panel" aria-label="Action log">
          <LastHand snapshot={snapshot} catalog={catalog} recap={aidsOn} />
          <div className="label">This hand</div>
          <ol className="log-lines">
            {log.map((l) => (
              <li key={l.seq} className={`log-${l.type}`}>
                {l.text}
              </li>
            ))}
          </ol>
        </aside>
      </div>

      <ActionPanel snapshot={snapshot} token={token} onSnapshot={onSnapshot} timer={timer} connections={connections} />

      <div className="study-grid">
        {aidsOn && seated && <LearningPanel view={view} aids={aids} failed={failed} catalog={catalog} />}
        <NotesPanel snapshot={snapshot} catalog={catalog} />
      </div>
    </main>
  )
}
