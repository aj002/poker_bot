import { cardText } from './cards'
import type { PublicEvent, SeatInfo } from './types'

export const RULESETS: Record<string, { name: string; short: string; detail: string }> = {
  practice_nlhe: {
    name: "No-limit Hold'em",
    short: 'NLHE',
    detail: '52 cards · two hole cards · no-limit',
  },
  uncertainty_shortdeck: {
    name: "Short-deck Hold'em",
    short: 'Short deck',
    detail: '36 cards (6–A) · flush beats full house',
  },
  uncertainty_plo4: {
    name: 'Pot-limit Omaha',
    short: 'PLO4',
    detail: 'Four hole cards · exactly two used · pot-limit',
  },
}

export const chips = (n: number) => n.toLocaleString('en-US')
export const anteLabel = (posting: unknown) => (posting === 'big_blind' ? 'BB ante' : 'ante')

export function points(n: number) {
  return Number.isInteger(n) ? String(n) : n.toFixed(2)
}

export function ordinal(n: number) {
  const s = ['th', 'st', 'nd', 'rd']
  const v = n % 100
  return n + (s[(v - 20) % 10] || s[v] || s[0])
}

export function placeLabel(positions: number[]) {
  if (positions.length === 1) return ordinal(positions[0])
  return `${ordinal(positions[0])}–${ordinal(positions[positions.length - 1])}`
}

/** How a seat reads on the table. Hidden seats carry only the server's "Player N" label. */
export function seatWho(info: SeatInfo | undefined) {
  if (!info) return 'Human'
  if (info.you) return 'You'
  if (info.kind === 'human') return 'Human'
  return info.hidden && !info.style ? info.label : `Bot · ${info.label}`
}

export function seatName(seat: number, seats: SeatInfo[] | undefined) {
  const info = seats?.find((s) => s.seat === seat)
  if (info?.you) return `Seat ${seat} (you)`
  return `Seat ${seat}`
}

/** Readable lines for a hand's public events. The first postflop raise_to on a street reads "bets". */
export function describeEvents(events: PublicEvent[], seats?: SeatInfo[], nameOf?: (seat: number) => string) {
  const out: { seq: number; text: string; type: string }[] = []
  let opened = false
  for (const e of events) {
    if (e.type === 'BoardDealt') opened = false
    const text = describeEvent(e, seats, opened, nameOf)
    if (e.type === 'ActionApplied' && e.data.action?.type === 'raise_to') opened = true
    if (text) out.push({ seq: e.seq, text, type: e.type })
  }
  return out
}

function describeEvent(
  e: PublicEvent,
  seats: SeatInfo[] | undefined,
  streetOpened: boolean,
  nameOf?: (seat: number) => string,
): string | null {
  const d = e.data
  const who = (s: number) => nameOf?.(s) ?? seatName(s, seats)
  switch (e.type) {
    case 'HandStarted':
      return `Hand ${d.hand_number} · blinds ${chips(d.small_blind)}/${chips(d.big_blind)}${
        d.ante ? ` · ${anteLabel(d.ante_posting)} ${chips(d.ante)}` : ''
      }`
    case 'AntePosted':
      return `${who(d.seat)} antes ${chips(d.amount)}`
    case 'BlindPosted':
      return `${who(d.seat)} posts ${chips(d.amount)}`
    case 'BoardDealt':
      return `${String(d.street).toUpperCase()} ${(d.cards as string[]).map(cardText).join(' ')}`
    case 'ActionFallback':
      return `${who(d.seat)} ${d.reason === 'timeout' ? 'timed out' : `fallback (${d.reason})`}`
    case 'ActionApplied': {
      const a = d.action as { type: string; amount?: number }
      if (d.auto) return null
      if (a.type === 'raise_to') {
        const verb = d.street !== 'preflop' && !streetOpened ? 'bets' : 'raises to'
        return `${who(d.seat)} ${verb} ${chips(a.amount ?? 0)}`
      }
      if (a.type === 'call') {
        const to = d.street_committed as number
        return `${who(d.seat)} calls ${chips(to)}${d.added !== to ? ` · adds ${chips(d.added)}` : ''}`
      }
      return `${who(d.seat)} ${a.type === 'fold' ? 'folds' : 'checks'}`
    }
    case 'ExcessReturned':
      return `${chips(d.amount)} returned to ${who(d.seat)}`
    case 'CardsRevealed':
      return `${who(d.seat)} shows ${(d.cards as string[]).map(cardText).join(' ')}`
    case 'PotAwarded': {
      const awards = Object.entries(d.awards as Record<string, number>)
        .filter(([, v]) => v > 0)
        .map(([s, v]) => `${who(Number(s))} wins ${chips(v)}`)
      return awards.join(', ') || null
    }
    case 'PlayerEliminated':
      return `${who(d.seat)} eliminated · ${placeLabel(d.positions)}`
    default:
      return null
  }
}
