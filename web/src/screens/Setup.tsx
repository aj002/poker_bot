import { useEffect, useMemo, useState } from 'react'
import { ApiFailure, api, createAndJoin, useOperatorToken, useStyles, type CreateBody } from '../api'
import { RULESETS, chips } from '../format'
import { useProgress } from '../learn/progress'
import { Term } from '../learn/Term'
import type { Policy, ResolvedConfig, StyleMode } from '../types'

type Assignment = 'human' | string

const HAND_CAPS = [10, 25, 50, 100, 200]
const RANDOM = 'style_random'

const MODES: { id: StyleMode; name: string; detail: string }[] = [
  { id: 'labelled', name: 'Labelled', detail: 'Every bot shows its style from the start.' },
  { id: 'hidden', name: 'Hidden', detail: 'Bots show only “Player 3”. Work out their styles by watching them play.' },
  { id: 'mixed', name: 'Mixed', detail: 'You choose which bots hide their style.' },
  { id: 'random', name: 'Random styles', detail: 'The server secretly picks a style for every bot, so even you do not know.' },
]

function defaultAssignments(n: number, previous: Assignment[]): Assignment[] {
  return Array.from({ length: n }, (_, i) => previous[i] ?? (i < 2 ? 'human' : 'legal_random'))
}

export function Setup({ navigate }: { navigate: (path: string) => void }) {
  const operator = useOperatorToken()
  const [policies, setPolicies] = useState<Policy[]>([])
  const catalog = useStyles()
  const [progress] = useProgress()
  const [mode, setMode] = useState<StyleMode>('labelled')
  const [hide, setHide] = useState<boolean[]>([])
  const [learningAids, setLearningAids] = useState(true)
  const [ruleset, setRuleset] = useState('practice_nlhe')
  const [seatCount, setSeatCount] = useState(6)
  const [assignments, setAssignments] = useState<Assignment[]>(defaultAssignments(6, []))
  const [stack, setStack] = useState(2000)
  const [handCap, setHandCap] = useState(200)
  const [turnSeconds, setTurnSeconds] = useState(20)
  const [resolved, setResolved] = useState<{ config: ResolvedConfig; config_hash: string } | null>(null)
  const [resolveError, setResolveError] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const [createError, setCreateError] = useState<string | null>(null)

  useEffect(() => {
    api.catalog(operator).then(
      (c) => setPolicies(c.policies.filter((p) => !p.alias_of)),
      () => setResolveError('Server unreachable. Is `make dev` running?'),
    )
  }, [operator])

  useEffect(() => {
    if (policies.length === 0) return
    const supported = (id: string) =>
      id === 'human' || (id === RANDOM ? ruleset === 'practice_nlhe' : policies.find((p) => p.id === id)?.rulesets.includes(ruleset))
    setAssignments((prev) => (prev.every(supported) ? prev : prev.map((a) => (supported(a) ? a : 'legal_random'))))
  }, [ruleset, policies])

  const body: CreateBody = useMemo(
    () => ({
      config: {
        ruleset_id: ruleset,
        initial_player_count: seatCount,
        starting_stack: stack,
        hand_cap: handCap,
        human_turn_ms: Math.round(turnSeconds * 1000),
      },
      seats: assignments.map((a, i) =>
        a === 'human'
          ? { kind: 'human' }
          : { kind: 'bot', policy: mode === 'random' ? RANDOM : a, ...(mode === 'mixed' ? { reveal: a !== RANDOM && !hide[i] } : {}) },
      ),
      styles: { mode, learning_aids: learningAids && ruleset === 'practice_nlhe' },
    }),
    [ruleset, seatCount, stack, handCap, turnSeconds, assignments, mode, hide, learningAids],
  )

  useEffect(() => {
    const t = window.setTimeout(() => {
      api.resolve(body, operator).then(
        (r) => {
          setResolved(r)
          setResolveError(null)
        },
        (err) => {
          setResolved(null)
          setResolveError(err instanceof ApiFailure ? formatFailure(err) : 'Server unreachable. Is `make dev` running?')
        },
      )
    }, 150)
    return () => window.clearTimeout(t)
  }, [body, operator])

  const setCount = (n: number) => {
    setSeatCount(n)
    setAssignments((prev) => defaultAssignments(n, prev))
  }

  const humans = assignments.filter((a) => a === 'human').length

  const create = async () => {
    setCreating(true)
    setCreateError(null)
    try {
      navigate(`/t/${await createAndJoin(body, operator)}`)
    } catch (err) {
      setCreateError(err instanceof ApiFailure ? formatFailure(err) : 'Could not reach the server.')
      setCreating(false)
    }
  }

  const policyById = Object.fromEntries(policies.map((p) => [p.id, p]))
  const usedBots = [...new Set(assignments.filter((a) => a !== 'human' && a !== RANDOM && mode !== 'random'))]
  const profileOf = (policyId: string) => catalog?.styles.find((st) => st.policy_id === policyId) ?? null
  const basics = catalog?.styles.filter((st) => !st.advanced).map((st) => st.id) ?? []
  const unlocked = basics.length > 0 && basics.every((id) => progress.beaten.includes(id))
  const chooseMode = (m: StyleMode) => {
    setMode(m)
    if (m === 'random' && ruleset !== 'practice_nlhe') setRuleset('practice_nlhe')
  }

  return (
    <main className="setup">
      <section className="panel">
        <h1 className="screen-title">New practice table</h1>
        <p className="lede">
          A single-table tournament on the authoritative engine. Chips persist across hands; the table ends with one
          survivor or at the hand cap, where survivors are scored by ICM.
        </p>

        <fieldset className="field">
          <legend className="label">Rule scenario</legend>
          <div className="ruleset-options">
            {Object.entries(RULESETS).map(([id, r]) => (
              <label key={id} className={`option${ruleset === id ? ' option-on' : ''}`}>
                <input type="radio" name="ruleset" value={id} checked={ruleset === id} onChange={() => setRuleset(id)} />
                <span className="option-name">{r.name}</span>
                <span className="option-detail">{r.detail}</span>
                <span className="mono option-id">{id}</span>
              </label>
            ))}
          </div>
        </fieldset>

        <fieldset className="field">
          <legend className="label">Seats</legend>
          <div className="segmented" role="radiogroup" aria-label="Seat count">
            {[2, 3, 4, 5, 6, 7, 8].map((n) => (
              <button
                key={n}
                type="button"
                role="radio"
                aria-checked={seatCount === n}
                className={seatCount === n ? 'seg-on' : ''}
                onClick={() => setCount(n)}
              >
                {n}
              </button>
            ))}
          </div>
          <p className="hint">2–6 for practice; 8 matches the published competition table size.</p>
        </fieldset>

        <fieldset className="field">
          <legend className="label">Opponent styles</legend>
          <div className="mode-options" role="radiogroup" aria-label="Table mode">
            {MODES.map((m) => (
              <label key={m.id} className={`option${mode === m.id ? ' option-on' : ''}`}>
                <input type="radio" name="mode" value={m.id} checked={mode === m.id} onChange={() => chooseMode(m.id)} />
                <span className="option-name">{m.name}</span>
                <span className="option-detail">{m.detail}</span>
              </label>
            ))}
          </div>
          <p className="hint">
            Styles are modelled on the players described in <cite>Harrington on Hold'em, Vol. I</cite> (Part Two). No-limit Hold'em only.
          </p>
        </fieldset>

        <fieldset className="field">
          <legend className="label">Who sits where</legend>
          <ol className="seat-assign">
            {assignments.map((a, i) => {
              const randomSeat = a !== 'human' && mode === 'random'
              const value = randomSeat ? RANDOM : a
              const profile = profileOf(value)
              return (
              <li key={i}>
                <span className="mono seat-id">Seat {i}</span>
                <select
                  aria-label={`Seat ${i} player`}
                  value={value}
                  onChange={(e) => setAssignments((prev) => prev.map((p, j) => (j === i ? e.target.value : p)))}
                >
                  <option value="human">Human</option>
                  {catalog && (
                    <optgroup label="Opponent styles">
                      {catalog.styles.map((st) => {
                        const nlhe = st.rulesets.includes(ruleset)
                        const locked = st.advanced && !unlocked
                        return (
                          <option key={st.id} value={st.policy_id} disabled={!nlhe || locked || mode === 'random'}>
                            Style · {st.name}
                            {!nlhe ? ' (NLHE only)' : locked ? ' (beat the other styles to unlock)' : ''}
                          </option>
                        )
                      })}
                      <option value={RANDOM} disabled={ruleset !== 'practice_nlhe'}>
                        Style · Random (secret)
                      </option>
                    </optgroup>
                  )}
                  {(['strategy', 'baseline', 'registered'] as const).map((source) => {
                    const group = policies.filter((p) => p.source === source)
                    if (group.length === 0) return null
                    return (
                      <optgroup key={source} label={
                          source === 'strategy' ? 'Competition bot' : source === 'baseline' ? 'Baseline bots' : 'Registered files (operator)'
                        }>
                        {group.map((p) => {
                          const ok = p.rulesets.includes(ruleset)
                          return (
                            <option key={p.id} value={p.id} disabled={!ok || mode === 'random'}>
                              Bot · {p.label}
                              {ok ? '' : ` (${p.rulesets.map((r) => RULESETS[r]?.short ?? r).join(', ')} only)`}
                            </option>
                          )
                        })}
                      </optgroup>
                    )
                  })}
                </select>
                {a !== 'human' && (
                  <div className="seat-style" data-testid={`seat-style-${i}`}>
                    {mode === 'mixed' && value !== RANDOM && (
                      <label className="toggle">
                        <input
                          type="checkbox"
                          checked={!!hide[i]}
                          onChange={(e) => setHide((prev) => Object.assign([...prev], { [i]: e.target.checked }))}
                          aria-label={`Hide seat ${i} style`}
                        />
                        <span>Hide style</span>
                      </label>
                    )}
                    <span className="hint">
                      {mode === 'random' || value === RANDOM
                        ? 'Style picked in secret, shown at the end.'
                        : mode === 'hidden' || (mode === 'mixed' && hide[i])
                          ? 'Shown as “Player ' + i + '” until the end.'
                          : profile
                            ? profile.description
                            : 'Not a book style: plays its own simple rules.'}
                    </span>
                  </div>
                )}
              </li>
              )
            })}
          </ol>
          <p className="hint">
            Baselines are the milestone 4 reference bots.{' '}
            {operator ? 'Registered files run in sandboxed workers under the 1 ms budget.' : 'Operators can also seat registered bot files.'}
          </p>
          <p className="hint">
            {humans === 0
              ? 'Bot-only table: you will watch as a spectator.'
              : humans === 1
                ? 'You take the human seat.'
                : `You take the first human seat; share the table link so ${humans - 1} more player${humans > 2 ? 's' : ''} can claim the rest.`}
          </p>
        </fieldset>

        <div className="field-row">
          <label className="field">
            <span className="label">Starting stack</span>
            <input
              type="number"
              min={1}
              step={100}
              value={stack}
              onChange={(e) => setStack(Number(e.target.value))}
              inputMode="numeric"
            />
          </label>
          <label className="field">
            <span className="label">Hand cap</span>
            <select value={handCap} onChange={(e) => setHandCap(Number(e.target.value))}>
              {HAND_CAPS.map((c) => (
                <option key={c} value={c}>
                  {c} hands
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span className="label">Human turn</span>
            <select value={turnSeconds} onChange={(e) => setTurnSeconds(Number(e.target.value))}>
              {[10, 20, 30, 60].map((s) => (
                <option key={s} value={s}>
                  {s} s
                </option>
              ))}
            </select>
          </label>
        </div>

        <label className="toggle toggle-field">
          <input
            type="checkbox"
            checked={learningAids && ruleset === 'practice_nlhe'}
            disabled={ruleset !== 'practice_nlhe'}
            onChange={(e) => setLearningAids(e.target.checked)}
          />
          <span>
            <strong>Learning aids{ruleset === 'practice_nlhe' ? '' : " (No-limit Hold'em only)"}</strong>: your hand in words, <Term k="position">position</Term> tips, the opening chart,{' '}
            <Term k="pot_odds">pot odds</Term>, <Term k="equity">equity</Term> and a recap after each hand. Built only from your own
            cards and what everyone can see.
          </span>
        </label>
      </section>

      <aside className="panel resolved" aria-live="polite">
        <h2 className="label">Resolved configuration</h2>
        {resolveError && <p className="error">{resolveError}</p>}
        {resolved && (
          <>
            <dl className="kv">
              <dt>Ruleset</dt>
              <dd className="mono">{resolved.config.ruleset_id}</dd>
              <dt>Config hash</dt>
              <dd className="mono hash" title={resolved.config_hash}>
                {resolved.config_hash.slice(0, 16)}…
              </dd>
              <dt>Payout points</dt>
              <dd className="mono">[{resolved.config.payout_points.join(', ')}]</dd>
              <dt>Levels</dt>
              <dd className="mono">every {resolved.config.hands_per_level} hands</dd>
            </dl>
            <table className="blinds">
              <thead>
                <tr>
                  <th>Level</th>
                  <th>Hands</th>
                  <th>Blinds</th>
                </tr>
              </thead>
              <tbody>
                {resolved.config.blind_levels
                  .slice(0, Math.ceil(resolved.config.hand_cap / resolved.config.hands_per_level))
                  .map(([sb, bb], i) => (
                    <tr key={i}>
                      <td className="mono">{i + 1}</td>
                      <td className="mono">
                        {i * resolved.config.hands_per_level + 1}–
                        {Math.min((i + 1) * resolved.config.hands_per_level, resolved.config.hand_cap)}
                      </td>
                      <td className="mono">
                        {chips(sb)}/{chips(bb)}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
            {usedBots.length > 0 && (
              <dl className="kv">
                {usedBots.map((id) => (
                  <div key={id} className="kv-pair">
                    <dt>
                      {policyById[id]?.label ?? id}
                      {policyById[id]?.source === 'registered' ? ' · file' : ''}
                    </dt>
                    <dd className="mono hash" title={policyById[id]?.artifact_hash}>
                      sha256 {policyById[id]?.artifact_hash.slice(0, 12)}…
                    </dd>
                  </div>
                ))}
              </dl>
            )}
          </>
        )}
        {createError && <p className="error">{createError}</p>}
        <button className="btn btn-primary btn-block" disabled={!resolved || creating} onClick={create}>
          {creating ? 'Creating…' : 'Create table'}
        </button>
      </aside>
    </main>
  )
}

export function formatFailure(err: ApiFailure) {
  const details = Array.isArray(err.details)
    ? (err.details as { loc: string[]; msg: string }[]).map((d) => `${d.loc.filter((l) => l !== 'body').join('.')}: ${d.msg}`)
    : []
  return [err.message, ...details].join(' — ')
}
