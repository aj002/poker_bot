import { useEffect, useState } from 'react'
import { chips } from '../format'
import type { Aids, StylesCatalog, TableView } from '../types'
import { HandRankings } from './HandRankings'
import { GROUP_TEXT, POSITION_NAMES, POSITION_TIPS } from './positions'
import { COACH_KEY, SEEN_POSITIONS_KEY, useStored } from './prefs'
import { StartingGrid } from './StartingGrid'
import { Term } from './Term'

const pct = (x: number) => `${Math.round(x * 100)}%`

function stackZone(ratio: number) {
  if (ratio >= 20) return 'Plenty of room: you can afford to wait for good hands.'
  if (ratio >= 10) return 'Getting shorter: open a few more hands so the blinds do not wear you down.'
  if (ratio >= 6) return 'Short: look for a good moment to move all in before the blinds shrink you further.'
  return 'Very short: most reasonable hands are worth moving all in with now.'
}

function coachText(c: NonNullable<NonNullable<Aids['hero']>['coach']>, view: TableView) {
  if (c.action === 'raise_to') return `${view.current_bet_to === 0 ? 'Bet' : 'Raise to'} ${chips(c.amount ?? 0)}`
  return c.action.charAt(0).toUpperCase() + c.action.slice(1)
}

function PositionTip({ label, hand }: { label: keyof typeof POSITION_TIPS; hand: number }) {
  const [seen, setSeen] = useStored<Record<string, number>>(SEEN_POSITIONS_KEY, {})
  const first = seen[label]
  useEffect(() => {
    if (first === undefined) setSeen({ ...seen, [label]: hand })
  }, [label, first])
  if (first !== undefined && first !== hand) return null
  return (
    <div className="pos-tip" role="note" data-testid="position-tip">
      <p>
        <strong>First time here: {POSITION_NAMES[label]}.</strong> {POSITION_TIPS[label]}
      </p>
      <button type="button" className="btn btn-small" onClick={() => setSeen({ ...seen, [label]: -1 })}>
        Got it
      </button>
    </div>
  )
}

/**
 * Learning aids for this seat. The numbers come from GET /tables/{id}/aids, which the server
 * computes from this seat's own cards, the board and public state only.
 */
export function LearningPanel({
  view,
  aids,
  failed,
  catalog,
}: {
  view: TableView
  aids: Aids | null
  failed: boolean
  catalog: StylesCatalog | null
}) {
  const [coach, setCoach] = useStored<boolean>(COACH_KEY, false)
  const [sheet, setSheet] = useState(false)
  const me = view.players.find((p) => p.seat === view.hero_seat)
  if (view.hero_seat === null || !me || me.status === 'eliminated') return null

  const current = aids && aids.hand_id === view.hand_id ? aids : null
  const hero = current?.hero ?? null
  const live = !!hero && me.status !== 'folded'
  const ante = view.blinds?.ante ?? 0
  const dealt = view.players.filter((p) => p.status !== 'eliminated').length
  const antes = ante ? (view.blinds?.ante_posting === 'classic' ? ante * dealt : ante) : 0
  const blinds = view.blinds ? view.blinds.small_blind + view.blinds.big_blind + antes : 0
  const ratio = blinds > 0 ? me.stack / blinds : null
  const toCome = view.board.length === 3 ? 2 : view.board.length === 4 ? 1 : 0
  const position = hero?.position ?? null
  const eq = hero?.equity_vs_random
  const odds = hero?.pot_odds ?? null

  return (
    <section className="learn-panel" aria-label="Learning aids" data-testid="learning-aids" aria-busy={!current && !failed}>
      <div className="learn-head">
        <h2 className="label">Learning aids</h2>
        <label className="toggle">
          <input type="checkbox" checked={coach} onChange={(e) => setCoach(e.target.checked)} />
          <span>Coach hint</span>
        </label>
      </div>
      {failed && !current && <p className="error">Learning aids are unavailable right now. The table still works.</p>}

      <div className="learn-card" data-testid="hand-words">
        <div className="label">Your hand</div>
        {live && hero ? (
          <>
            <p className="learn-big">{hero.hand_in_words}</p>
            {view.board.length > 0 && <p className="hint">Started with {hero.hand_class.replace('T', '10')}.</p>}
            {hero.draws.map((d) => (
              <p className="learn-draw" key={d.kind} data-testid="hand-draw">
                {d.words}
                {toCome > 0 && (
                  <>
                    {' '}
                    · about {Math.min(100, d.outs * (toCome === 2 ? 4 : 2))}% to hit by the river (<Term k="outs">outs</Term> × {toCome === 2 ? 4 : 2})
                  </>
                )}
              </p>
            ))}
          </>
        ) : (
          <p className="muted">{me.status === 'folded' ? 'You folded this hand.' : current ? 'Not dealt in this hand.' : 'Reading your cards…'}</p>
        )}
        <button type="button" className="btn btn-small" onClick={() => setSheet(true)}>
          Hand rankings
        </button>
      </div>

      {position && (
        <div className="learn-card" data-testid="hero-position">
          <div className="label">Your seat</div>
          <p className="learn-big">
            {POSITION_NAMES[position.label]} <span className="mono muted">{position.label}</span>
          </p>
          <p className="hint">
            <Term k="position">Position</Term>: {GROUP_TEXT[position.group]}.
          </p>
          <PositionTip label={position.label} hand={view.hand_number} />
        </div>
      )}

      {live && eq && (
        <div className="learn-card" data-testid="pot-odds">
          <div className="label">The price</div>
          <p className="learn-big">
            {odds ? `${odds.words}.` : view.legal ? 'Nothing to call: checking is free.' : 'Not your turn yet.'}
          </p>
          <p data-testid="equity">
            Against one random hand you would win about <strong>{pct(eq.vs_one)}</strong>
            {eq.opponents > 1 && (
              <>
                ; against {eq.opponents} random hands, about <strong>{pct(eq.vs_field)}</strong>
              </>
            )}{' '}
            (<Term k="equity">equity</Term>).
          </p>
          {odds && (
            <p className="hint">
              {eq.vs_one >= odds.break_even
                ? 'That beats the break-even against a random hand. Real players do not bet random hands, so think about who is betting.'
                : 'That is below the break-even. Unless you expect to win more on later streets, folding saves chips.'}{' '}
              See <Term k="pot_odds">pot odds</Term>.
            </p>
          )}
        </div>
      )}

      {ratio !== null && (
        <div className="learn-card" data-testid="stack-ratio">
          <div className="label">
            <Term k="stack_ratio">Stack ratio</Term>
          </div>
          <p className="learn-big">{ratio.toFixed(ratio < 10 ? 1 : 0)} rounds of blinds{antes ? ' and antes' : ''}</p>
          <p className="hint">{stackZone(ratio)}</p>
        </div>
      )}

      {coach && hero?.coach && (
        <div className="learn-card coach" data-testid="coach-hint">
          <div className="label">Coach</div>
          <p className="learn-big">{coachText(hero.coach, view)}</p>
          <p>{hero.coach.reason}</p>
          <p className="hint">{hero.coach.disclaimer}</p>
        </div>
      )}
      {coach && live && !hero?.coach && (
        <div className="learn-card coach" data-testid="coach-hint">
          <div className="label">Coach</div>
          <p className="muted">The coach speaks up when it is your turn.</p>
        </div>
      )}

      {catalog && live && hero && (
        <details className="learn-card learn-wide" key={`${view.hand_id}-${view.street === 'preflop'}`} open={view.street === 'preflop'}>
          <summary className="label">Starting-hand chart · {GROUP_TEXT[hero.charts.group]}</summary>
          <StartingGrid catalog={catalog} group={hero.charts.group} hand={hero.hand_class} />
        </details>
      )}

      <HandRankings open={sheet} onClose={() => setSheet(false)} row={live ? (hero?.rank_row ?? null) : null} />
    </section>
  )
}
