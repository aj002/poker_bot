import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiFailure, advisorApi } from '../api'
import { Card, CardSlot, rankLabel } from '../cards'
import { chips } from '../format'
import type { AdvisorAction, AdvisorAdvice, AdvisorLegal, AdvisorSeat, AdvisorSetup, AdvisorView } from '../types'

const STORAGE_KEY = 'arena:advisor-session'
const RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
const SUITS: { id: string; symbol: string; name: string; red: boolean }[] = [
  { id: 's', symbol: '♠', name: 'spades', red: false },
  { id: 'h', symbol: '♥', name: 'hearts', red: true },
  { id: 'd', symbol: '♦', name: 'diamonds', red: true },
  { id: 'c', symbol: '♣', name: 'clubs', red: false },
]
const PAYOUTS: Record<string, string> = { winner: 'Winner takes all', top2: 'Top 2 paid (65/35)', top3: 'Top 3 paid (50/30/20)' }

const seatName = (n: number) => `Seat ${n + 1}`
const message = (e: unknown) => (e instanceof ApiFailure ? (e.status === 0 ? 'Could not reach the server.' : e.message) : 'Something went wrong.')

type Run = (op: Record<string, unknown> & { op: string }) => Promise<boolean>

function useCommitted(initial: number | string) {
  const [value, setValue] = useState(String(initial))
  return [value, setValue, Number.parseInt(value.replace(/[, ]/g, ''), 10)] as const
}

// ---------------------------------------------------------------- setup

function SetupForm({ onCreate, busy, error }: { onCreate: (body: Parameters<typeof advisorApi.create>[0]) => void; busy: boolean; error: string | null }) {
  const [players, setPlayers] = useState(6)
  const [hero, setHero] = useState(0)
  const [button, setButton] = useState(0)
  const [mode, setMode] = useState<'cash' | 'tournament'>('cash')
  const [payouts, setPayouts] = useState('top3')
  const [sb, setSb, sbN] = useCommitted(50)
  const [bb, setBb, bbN] = useCommitted(100)
  const [ante, setAnte, anteN] = useCommitted(0)
  const [anteKind, setAnteKind] = useState<'classic' | 'big_blind'>('classic')
  const [stack, setStack, stackN] = useCommitted(10000)
  const [custom, setCustom] = useState<Record<number, string>>({})

  const seats = Array.from({ length: players }, (_, i) => i)
  const stackOf = (i: number) => {
    const raw = custom[i]
    return raw === undefined ? stackN : Number.parseInt(raw.replace(/[, ]/g, ''), 10)
  }

  const choosePlayers = (n: number) => {
    setPlayers(n)
    setHero((h) => Math.min(h, n - 1))
    setButton((b) => Math.min(b, n - 1))
  }

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    onCreate({
      players,
      hero,
      button,
      sb: sbN,
      bb: bbN,
      ante: Number.isFinite(anteN) ? anteN : 0,
      ante_posting: anteN > 0 ? anteKind : null,
      mode,
      payouts: mode === 'tournament' ? payouts : undefined,
      stacks: seats.map(stackOf),
    })
  }

  return (
    <form className="panel adv-setup" onSubmit={submit} aria-label="Session setup">
      <h1 className="screen-title">Advisor</h1>
      <p className="lede">Set the game up once. Then type each hand in as it happens and the final bot tells you what to play.</p>

      <fieldset className="field">
        <legend>Players at the table</legend>
        <div className="adv-seg" role="group" aria-label="Players">
          {[2, 3, 4, 5, 6, 7, 8].map((n) => (
            <button key={n} type="button" aria-pressed={players === n} onClick={() => choosePlayers(n)}>
              {n}
            </button>
          ))}
        </div>
      </fieldset>

      <fieldset className="field">
        <legend>Your seat</legend>
        <p className="adv-hint">Number the chairs 1, 2, 3… clockwise, starting anywhere you like.</p>
        <div className="adv-seg" role="group" aria-label="Your seat">
          {seats.map((i) => (
            <button key={i} type="button" aria-pressed={hero === i} onClick={() => setHero(i)}>
              {i + 1}
            </button>
          ))}
        </div>
      </fieldset>

      <fieldset className="field">
        <legend>Who has the dealer button in the first hand</legend>
        <div className="adv-seg" role="group" aria-label="Dealer button">
          {seats.map((i) => (
            <button key={i} type="button" aria-pressed={button === i} onClick={() => setButton(i)}>
              {i + 1}
            </button>
          ))}
        </div>
      </fieldset>

      <fieldset className="field">
        <legend>Game type</legend>
        <div className="adv-seg adv-seg-2" role="group" aria-label="Game type">
          <button type="button" aria-pressed={mode === 'cash'} onClick={() => setMode('cash')}>
            Cash game
          </button>
          <button type="button" aria-pressed={mode === 'tournament'} onClick={() => setMode('tournament')}>
            Tournament
          </button>
        </div>
        {mode === 'tournament' ? (
          <label className="adv-label">
            Prizes
            <select value={payouts} onChange={(e) => setPayouts(e.target.value)} aria-label="Prizes">
              {Object.entries(PAYOUTS).map(([k, v]) => (
                <option key={k} value={k}>
                  {v}
                </option>
              ))}
            </select>
          </label>
        ) : (
          <p className="adv-hint">Cash: every chip is worth the same, so the bot just maximises chips.</p>
        )}
      </fieldset>

      <fieldset className="field">
        <legend>Blinds and ante</legend>
        <div className="adv-grid-3">
          <label className="adv-label">
            Small blind
            <input inputMode="numeric" value={sb} onChange={(e) => setSb(e.target.value)} aria-label="Small blind" />
          </label>
          <label className="adv-label">
            Big blind
            <input inputMode="numeric" value={bb} onChange={(e) => setBb(e.target.value)} aria-label="Big blind" />
          </label>
          <label className="adv-label">
            Ante
            <input inputMode="numeric" value={ante} onChange={(e) => setAnte(e.target.value)} aria-label="Ante" />
          </label>
        </div>
        {anteN > 0 && (
          <div className="adv-seg adv-seg-2" role="group" aria-label="Who pays the ante">
            <button type="button" aria-pressed={anteKind === 'classic'} onClick={() => setAnteKind('classic')}>
              Everyone pays
            </button>
            <button type="button" aria-pressed={anteKind === 'big_blind'} onClick={() => setAnteKind('big_blind')}>
              Big blind pays for all
            </button>
          </div>
        )}
      </fieldset>

      <fieldset className="field">
        <legend>Starting stacks</legend>
        <label className="adv-label">
          Everyone starts with
          <input
            inputMode="numeric"
            value={stack}
            onChange={(e) => {
              setStack(e.target.value)
              setCustom({})
            }}
            aria-label="Starting stack"
          />
        </label>
        <details className="adv-details">
          <summary>Different stacks per seat</summary>
          <div className="adv-grid-2">
            {seats.map((i) => (
              <label className="adv-label" key={i}>
                {i === hero ? `You (seat ${i + 1})` : seatName(i)}
                <input
                  inputMode="numeric"
                  value={custom[i] ?? stack}
                  onChange={(e) => setCustom({ ...custom, [i]: e.target.value })}
                  aria-label={`Stack of ${seatName(i)}`}
                />
              </label>
            ))}
          </div>
        </details>
      </fieldset>

      {error && (
        <p className="adv-error" role="alert">
          {error}
        </p>
      )}
      <button className="btn btn-primary btn-block adv-big" type="submit" disabled={busy}>
        Start session
      </button>
    </form>
  )
}

// ---------------------------------------------------------------- pieces

function CardPicker({ count, used, label, onDone }: { count: number; used: string[]; label: string; onDone: (cards: string[]) => void }) {
  const [picked, setPicked] = useState<string[]>([])
  const [rank, setRank] = useState<string | null>(null)
  const taken = new Set([...used, ...picked])

  const pickSuit = (suit: string) => {
    if (!rank) return
    const next = [...picked, rank + suit]
    setRank(null)
    if (next.length >= count) {
      setPicked([])
      onDone(next)
    } else setPicked(next)
  }

  return (
    <section className="adv-card-picker" aria-label={label}>
      <div className="adv-picked" aria-live="polite">
        {Array.from({ length: count }, (_, i) =>
          picked[i] ? <Card key={i} card={picked[i]} size="md" /> : <CardSlot key={i} size="md" />,
        )}
        {(picked.length > 0 || rank) && (
          <button
            type="button"
            className="btn btn-small"
            onClick={() => {
              if (rank) setRank(null)
              else setPicked(picked.slice(0, -1))
            }}
          >
            Clear
          </button>
        )}
      </div>
      <p className="adv-step-label">{rank ? 'Now the suit' : `${label}: pick the rank`}</p>
      {rank ? (
        <div className="adv-suits" role="group" aria-label="Suit">
          {SUITS.map((s) => {
            const gone = taken.has(rank + s.id)
            return (
              <button
                key={s.id}
                type="button"
                className={`adv-suit${s.red ? ' red' : ''}`}
                disabled={gone}
                aria-label={s.name}
                onClick={() => pickSuit(s.id)}
              >
                <span aria-hidden="true">{s.symbol}</span>
                <small>{s.name}</small>
              </button>
            )
          })}
        </div>
      ) : (
        <div className="adv-ranks" role="group" aria-label="Rank">
          {RANKS.map((r) => {
            const gone = SUITS.every((s) => taken.has(r + s.id))
            return (
              <button key={r} type="button" disabled={gone} aria-label={r === 'T' ? '10' : r} onClick={() => setRank(r)}>
                {rankLabel(r)}
              </button>
            )
          })}
        </div>
      )}
    </section>
  )
}

function SeatStrip({ view }: { view: AdvisorView }) {
  return (
    <ul className="adv-seats" aria-label="Seats">
      {view.seats.map((s: AdvisorSeat) => {
        const classes = ['adv-seat']
        if (s.hero) classes.push('hero')
        if (s.status === 'folded') classes.push('folded')
        if (!s.in_hand) classes.push('out')
        if (view.actor === s.seat) classes.push('acting')
        const state = !s.in_hand ? 'Out' : s.status === 'folded' ? 'Folded' : s.status === 'all_in' ? 'All-in' : view.actor === s.seat ? 'To act' : ''
        return (
          <li key={s.seat} className={classes.join(' ')} data-testid={`adv-seat-${s.seat + 1}`}>
            <span className="adv-seat-top">
              <b>{s.hero ? 'You' : seatName(s.seat).replace('Seat ', 'S')}</b>
              {s.button && <span className="adv-dealer" title="Dealer button">D</span>}
            </span>
            <span className="adv-seat-pos">{s.in_hand ? s.position : ''}</span>
            <span className="adv-seat-stack">{chips(s.stack)}</span>
            <span className="adv-seat-state">{s.street_committed > 0 && state !== 'Folded' ? `in ${chips(s.street_committed)}` : state}</span>
          </li>
        )
      })}
    </ul>
  )
}

function Table({ view }: { view: AdvisorView }) {
  return (
    <section className="adv-table" aria-label="Cards and pot">
      <div className="adv-cards-row">
        <div className="adv-hole" aria-label="Your cards">
          {view.hole ? view.hole.map((c) => <Card key={c} card={c} size="md" />) : [0, 1].map((i) => <CardSlot key={i} size="md" />)}
        </div>
        <div className="adv-board" aria-label="Board">
          {Array.from({ length: 5 }, (_, i) => (view.board[i] ? <Card key={view.board[i]} card={view.board[i]} size="md" /> : <CardSlot key={i} size="md" />))}
        </div>
      </div>
      <p className="adv-pot">
        Pot <b>{chips(view.pot)}</b> · {view.street}
      </p>
    </section>
  )
}

function Advice({ advice, run, actor, busy }: { advice: AdvisorAdvice; run: Run; actor: number; busy: boolean }) {
  const record = (a: AdvisorAction) => run({ op: 'action', seat: actor, type: a.type, ...(a.amount ? { amount: a.amount } : {}) })
  return (
    <section className="adv-advice" aria-label="Recommendation" data-testid="advice">
      <p className="adv-eyebrow">Bot says</p>
      <p className="adv-headline" data-testid="advice-headline">
        {advice.headline}
      </p>
      {advice.mixed && (
        <div className="adv-mix" data-testid="advice-mix">
          <p className="adv-hint">Mixed spot. The bot rolled the dice for you; these are its odds:</p>
          <ul>
            {advice.mix.map((m) => (
              <li key={m.label}>
                <button type="button" disabled={busy} onClick={() => record(m.action)} aria-label={`Record ${m.label}, ${m.pct} percent`}>
                  <span className="adv-mix-bar" style={{ width: `${m.pct}%` }} aria-hidden="true" />
                  <span className="adv-mix-text">{m.label}</span>
                  <span className="adv-mix-pct">{m.pct}%</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
      <p className="adv-reason" data-testid="advice-reason">
        {advice.reason}
      </p>
      <p className="adv-meta">
        {advice.position} · {Math.round(advice.effective_bb)} BB deep
      </p>
      <button type="button" className="btn btn-primary btn-block adv-big" disabled={busy} onClick={() => record(advice.pick)}>
        I did this: {advice.headline}
      </button>
    </section>
  )
}

function ActionPad({ view, run, busy }: { view: AdvisorView; run: Run; busy: boolean }) {
  const legal = view.legal as AdvisorLegal
  const actor = view.actor as number
  const [raising, setRaising] = useState(false)
  const [amount, setAmount, amountN] = useCommitted(legal.min_raise_to ?? '')
  const bet = view.seats.every((s) => s.street_committed === 0)
  const noun = bet ? 'Bet' : 'Raise'
  const lo = legal.min_raise_to ?? legal.all_in_raise_to ?? 0
  const hi = legal.max_raise_to ?? legal.all_in_raise_to ?? 0
  const valid = Number.isFinite(amountN) && ((amountN >= lo && amountN <= hi) || amountN === legal.all_in_raise_to)
  const go = (type: string, amt?: number) => run({ op: 'action', seat: actor, type, ...(amt ? { amount: amt } : {}) })

  useEffect(() => {
    setRaising(false)
    setAmount(String(legal.min_raise_to ?? ''))
  }, [actor, view.street, view.log.length]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <section className="adv-pad" aria-label={`${seatName(actor)} action`}>
      <p className="adv-step-label">
        {view.seats[actor].hero ? 'Your action' : `${seatName(actor)} (${view.seats[actor].position}) does what?`}
      </p>
      <div className="adv-pad-row">
        <button type="button" className="btn adv-act" disabled={busy || !legal.fold} onClick={() => go('fold')}>
          Fold
        </button>
        {legal.check ? (
          <button type="button" className="btn adv-act" disabled={busy} onClick={() => go('check')}>
            Check
          </button>
        ) : (
          <button type="button" className="btn adv-act" disabled={busy} onClick={() => go('call')}>
            Call {chips(legal.call_amount)}
          </button>
        )}
        {legal.can_raise ? (
          <button type="button" className="btn adv-act" aria-expanded={raising} disabled={busy} onClick={() => setRaising(!raising)}>
            {noun}…
          </button>
        ) : legal.short_all_in_raise_to ? (
          <button type="button" className="btn adv-act" disabled={busy} onClick={() => go('raise_to', legal.short_all_in_raise_to as number)}>
            All-in {chips(legal.short_all_in_raise_to)}
          </button>
        ) : null}
      </div>
      {raising && legal.can_raise && (
        <div className="adv-raise">
          <div className="adv-chips-row">
            {legal.min_raise_to != null && (
              <button type="button" className="btn btn-small" onClick={() => setAmount(String(legal.min_raise_to))}>
                Min {chips(legal.min_raise_to)}
              </button>
            )}
            {legal.pot_raise_to != null && legal.pot_raise_to !== legal.all_in_raise_to && (
              <button type="button" className="btn btn-small" onClick={() => setAmount(String(legal.pot_raise_to))}>
                Pot {chips(legal.pot_raise_to)}
              </button>
            )}
            {legal.all_in_raise_to != null && (
              <button type="button" className="btn btn-small" onClick={() => setAmount(String(legal.all_in_raise_to))}>
                All-in {chips(legal.all_in_raise_to)}
              </button>
            )}
          </div>
          <label className="adv-label">
            {noun} to (total this round)
            <input inputMode="numeric" value={amount} onChange={(e) => setAmount(e.target.value)} aria-label={`${noun} to`} />
          </label>
          <button type="button" className="btn btn-primary btn-block adv-big" disabled={busy || !valid} onClick={() => go('raise_to', amountN)}>
            {noun} to {Number.isFinite(amountN) ? chips(amountN) : '…'}
          </button>
          {!valid && Number.isFinite(amountN) && (
            <p className="adv-hint">
              Between {chips(lo)} and {chips(hi)}.
            </p>
          )}
        </div>
      )}
    </section>
  )
}

function Winners({ view, run, busy }: { view: AdvisorView; run: Run; busy: boolean }) {
  const result = view.result!
  const [picks, setPicks] = useState<number[][]>(() => result.pots.map((_, i) => result.awards?.[i] ?? []))
  const toggle = (pot: number, seat: number) =>
    setPicks(picks.map((w, i) => (i !== pot ? w : w.includes(seat) ? w.filter((s) => s !== seat) : [...w, seat])))
  const ready = picks.every((w) => w.length > 0)
  return (
    <section className="adv-pad" aria-label="Who won">
      <p className="adv-step-label">Showdown: who won?</p>
      <p className="adv-hint">Tap the winner of each pot. Tap two players to split it.</p>
      {result.pots.map((pot, i) => (
        <div key={i} className="adv-pot-pick">
          <p>
            {result.pots.length > 1 ? (i === 0 ? 'Main pot' : `Side pot ${i}`) : 'Pot'} · <b>{chips(pot.amount)}</b>
          </p>
          <div className="adv-seg" role="group" aria-label={`Winner of pot ${i + 1}`}>
            {pot.eligible.map((s) => (
              <button key={s} type="button" aria-pressed={picks[i].includes(s)} onClick={() => toggle(i, s)}>
                {view.seats[s].hero ? 'You' : s + 1}
              </button>
            ))}
          </div>
        </div>
      ))}
      <button type="button" className="btn btn-primary btn-block adv-big" disabled={busy || !ready} onClick={() => run({ op: 'award', pots: picks })}>
        Confirm winners
      </button>
    </section>
  )
}

function NextHand({ view, run, busy }: { view: AdvisorView; run: Run; busy: boolean }) {
  const result = view.result!
  const resolved = result.resolved
  const [stacks, setStacks] = useState<Record<number, string>>({})
  const stackOf = (i: number) => stacks[i] ?? String(view.next_stacks[String(i)] ?? 0)
  const gained = result.winner != null ? result.pots.reduce((a, p) => a + p.amount, 0) : null
  return (
    <section className="adv-pad" aria-label="Hand finished">
      {result.ended_by_fold && result.winner != null && (
        <p className="adv-result" data-testid="hand-result">
          {view.seats[result.winner].hero ? 'You win' : `${seatName(result.winner)} wins`} {gained != null && gained > 0 ? chips(gained) : ''} (everyone else folded).
        </p>
      )}
      {resolved && !result.ended_by_fold && <p className="adv-result" data-testid="hand-result">Pot awarded.</p>}
      <button type="button" className="btn btn-primary btn-block adv-big" disabled={busy || !resolved} onClick={() => run({ op: 'new_hand' })}>
        Next hand
      </button>
      <details className="adv-details">
        <summary>Chip counts different on the table? Fix them</summary>
        <div className="adv-grid-2">
          {view.seats.map((s) => (
            <label className="adv-label" key={s.seat}>
              {s.hero ? `You (seat ${s.seat + 1})` : seatName(s.seat)}
              <input
                inputMode="numeric"
                value={stackOf(s.seat)}
                onChange={(e) => setStacks({ ...stacks, [s.seat]: e.target.value })}
                aria-label={`Chips for ${seatName(s.seat)}`}
              />
            </label>
          ))}
        </div>
        <button
          type="button"
          className="btn btn-block"
          disabled={busy}
          onClick={() => run({ op: 'new_hand', stacks: view.seats.map((s) => Number.parseInt(stackOf(s.seat).replace(/[, ]/g, ''), 10) || 0) })}
        >
          Next hand with these chips
        </button>
      </details>
    </section>
  )
}

function HandEditor({ view, run, busy }: { view: AdvisorView; run: Run; busy: boolean }) {
  const [stacks, setStacks] = useState<Record<number, string>>({})
  const [sb, setSb] = useState(String(view.blinds.sb))
  const [bb, setBb] = useState(String(view.blinds.bb))
  const [ante, setAnte] = useState(String(view.blinds.ante))
  const [button, setButton] = useState(view.button)
  const num = (v: string) => Number.parseInt(v.replace(/[, ]/g, ''), 10) || 0
  const stackOf = (i: number) => stacks[i] ?? String(view.next_stacks[String(i)] ?? 0)
  return (
    <details className="adv-details">
      <summary>Blinds, button or chips changed?</summary>
      <div className="adv-grid-3">
        <label className="adv-label">
          Small blind
          <input inputMode="numeric" value={sb} onChange={(e) => setSb(e.target.value)} aria-label="This hand small blind" />
        </label>
        <label className="adv-label">
          Big blind
          <input inputMode="numeric" value={bb} onChange={(e) => setBb(e.target.value)} aria-label="This hand big blind" />
        </label>
        <label className="adv-label">
          Ante
          <input inputMode="numeric" value={ante} onChange={(e) => setAnte(e.target.value)} aria-label="This hand ante" />
        </label>
      </div>
      <p className="adv-step-label">Dealer button</p>
      <div className="adv-seg" role="group" aria-label="This hand dealer button">
        {view.seats.map((s) => (
          <button key={s.seat} type="button" aria-pressed={button === s.seat} disabled={!s.in_hand} onClick={() => setButton(s.seat)}>
            {s.seat + 1}
          </button>
        ))}
      </div>
      <p className="adv-step-label">Chips (0 means out of the game)</p>
      <div className="adv-grid-2">
        {view.seats.map((s) => (
          <label className="adv-label" key={s.seat}>
            {s.hero ? `You (seat ${s.seat + 1})` : seatName(s.seat)}
            <input inputMode="numeric" value={stackOf(s.seat)} onChange={(e) => setStacks({ ...stacks, [s.seat]: e.target.value })} aria-label={`Hand chips for ${seatName(s.seat)}`} />
          </label>
        ))}
      </div>
      <button
        type="button"
        className="btn btn-block"
        disabled={busy}
        onClick={() =>
          run({
            op: 'edit_hand',
            sb: num(sb),
            bb: num(bb),
            ante: num(ante),
            ...(num(ante) > 0 ? { ante_posting: view.blinds.ante_posting ?? 'classic' } : {}),
            button,
            stacks: view.seats.map((s) => num(stackOf(s.seat))),
          })
        }
      >
        Apply to this hand
      </button>
    </details>
  )
}

function SessionMenu({ view, run, onNew, busy }: { view: AdvisorView; run: Run; onNew: () => void; busy: boolean }) {
  const setup: AdvisorSetup = view.setup
  const [payouts, setPayouts] = useState(setup.payouts.length === 3 ? 'top3' : setup.payouts.length === 2 ? 'top2' : 'winner')
  return (
    <details className="adv-details adv-menu">
      <summary>Session</summary>
      <div className="adv-seg adv-seg-2" role="group" aria-label="Game type">
        <button type="button" aria-pressed={setup.mode === 'cash'} disabled={busy} onClick={() => run({ op: 'edit_setup', mode: 'cash' })}>
          Cash game
        </button>
        <button type="button" aria-pressed={setup.mode === 'tournament'} disabled={busy} onClick={() => run({ op: 'edit_setup', mode: 'tournament', payouts })}>
          Tournament
        </button>
      </div>
      {setup.mode === 'tournament' && (
        <label className="adv-label">
          Prizes
          <select
            value={payouts}
            aria-label="Prizes"
            onChange={(e) => {
              setPayouts(e.target.value)
              run({ op: 'edit_setup', mode: 'tournament', payouts: e.target.value })
            }}
          >
            {Object.entries(PAYOUTS).map(([k, v]) => (
              <option key={k} value={k}>
                {v}
              </option>
            ))}
          </select>
        </label>
      )}
      <p className="adv-hint">Changing the game type re-asks the bot for this hand and keeps what it has learned about each seat.</p>
      <button type="button" className="btn btn-block" onClick={onNew}>
        End session and start a new one
      </button>
    </details>
  )
}

// ---------------------------------------------------------------- screen

export function AdvisorScreen() {
  const [view, setView] = useState<AdvisorView | null>(null)
  const [loading, setLoading] = useState(() => Boolean(localStorage.getItem(STORAGE_KEY)))
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [abandon, setAbandon] = useState(false)
  const sessionId = useRef<string | null>(localStorage.getItem(STORAGE_KEY))

  useEffect(() => {
    const id = sessionId.current
    if (!id) return
    advisorApi.get(id).then(
      (v) => {
        setView(v)
        setLoading(false)
      },
      (e) => {
        if (e instanceof ApiFailure && e.status === 404) {
          localStorage.removeItem(STORAGE_KEY)
          sessionId.current = null
        } else setError(message(e))
        setLoading(false)
      },
    )
  }, [])

  const create = useCallback(async (body: Parameters<typeof advisorApi.create>[0]) => {
    setBusy(true)
    setError(null)
    try {
      const v = await advisorApi.create(body)
      sessionId.current = v.session
      localStorage.setItem(STORAGE_KEY, v.session)
      setView(v)
    } catch (e) {
      setError(message(e))
    } finally {
      setBusy(false)
    }
  }, [])

  const run: Run = useCallback(async (op) => {
    const id = sessionId.current
    if (!id) return false
    setBusy(true)
    try {
      setView(await advisorApi.do(id, op))
      setError(null)
      setAbandon(false)
      return true
    } catch (e) {
      setError(message(e))
      return false
    } finally {
      setBusy(false)
    }
  }, [])

  const endSession = () => {
    localStorage.removeItem(STORAGE_KEY)
    sessionId.current = null
    setView(null)
    setError(null)
  }

  if (loading)
    return (
      <main className="panel narrow adv" aria-busy="true">
        <p className="lede">Loading your session…</p>
      </main>
    )
  if (!view) return (
    <main className="adv">
      <SetupForm onCreate={create} busy={busy} error={error} />
    </main>
  )

  const started = !view.editable
  const heroOut = view.result?.stacks_after[String(view.setup.hero)] === 0
  const stageKey = `${view.hand_number}:${view.log.length}:${view.stage}`
  const blindsText = `${chips(view.blinds.sb)}/${chips(view.blinds.bb)}${view.blinds.ante ? ` + ${chips(view.blinds.ante)} ante` : ''}`

  return (
    <main className="adv" data-testid="advisor">
      <header className="adv-bar">
        <div>
          <p className="adv-title">
            Hand {view.hand_number} · {view.setup.mode === 'cash' ? 'Cash' : 'Tournament'}
          </p>
          <p className="adv-sub">
            Blinds {blindsText} · button seat {view.button + 1}
          </p>
        </div>
        <div className="adv-bar-actions">
          <button type="button" className="btn btn-small" disabled={busy || !view.can_undo} onClick={() => run({ op: 'undo' })}>
            Undo
          </button>
          {abandon ? (
            <>
              <button type="button" className="btn btn-small btn-primary" disabled={busy} onClick={() => run({ op: 'new_hand' })}>
                Yes, new hand
              </button>
              <button type="button" className="btn btn-small" onClick={() => setAbandon(false)}>
                Keep
              </button>
            </>
          ) : (
            <button
              type="button"
              className="btn btn-small"
              disabled={busy || !started || view.stage === 'complete'}
              onClick={() => setAbandon(true)}
              title="Throw this hand away and deal the next one"
            >
              New hand
            </button>
          )}
        </div>
      </header>

      <SeatStrip view={view} />
      <Table view={view} />

      {error && (
        <p className="adv-error" role="alert">
          {error}
        </p>
      )}

      <div key={stageKey} className="adv-stage">
        {heroOut && view.stage === 'complete' && <p className="adv-error">You are out of chips. End the session to start another.</p>}
        {view.stage === 'hole' && (
          <>
            <CardPicker count={2} used={[]} label="Your two cards" onDone={(cards) => run({ op: 'hole', cards })} />
            <HandEditor view={view} run={run} busy={busy} />
          </>
        )}
        {view.stage === 'action' && view.actor != null && view.legal && (
          <>
            {view.advice && view.seats[view.actor].hero && <Advice advice={view.advice} run={run} actor={view.actor} busy={busy} />}
            <ActionPad view={view} run={run} busy={busy} />
          </>
        )}
        {view.stage === 'board' && view.need_board && (
          <CardPicker
            count={view.need_board}
            used={view.used_cards}
            label={view.street_needed === 'flop' ? 'The flop' : `The ${view.street_needed}`}
            onDone={(cards) => run({ op: 'board', cards })}
          />
        )}
        {view.stage === 'complete' && view.result && (
          <>
            {!view.result.ended_by_fold && <Winners view={view} run={run} busy={busy} />}
            {view.result.resolved && <NextHand view={view} run={run} busy={busy} />}
            {!view.result.ended_by_fold && !view.result.resolved && (
              <details className="adv-details">
                <summary>Skip this and type the chips yourself</summary>
                <NextHand view={view} run={run} busy={busy} />
              </details>
            )}
          </>
        )}
      </div>

      <details className="adv-details adv-log">
        <summary>Hand log ({view.log.length})</summary>
        <ol>
          {view.log.map((line, i) => (
            <li key={i}>{line}</li>
          ))}
        </ol>
      </details>
      <SessionMenu view={view} run={run} onNew={endSession} busy={busy} />
    </main>
  )
}
