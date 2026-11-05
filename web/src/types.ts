// Wire types mirroring arena/domain/observations.py and arena/api/manager.py.
// The browser renders these; it never derives game state on its own.

export type Street = 'preflop' | 'flop' | 'turn' | 'river'
export type PlayerStatus = 'active' | 'folded' | 'all_in' | 'eliminated'

export interface LegalActions {
  fold: boolean
  check: boolean
  call_amount: number | null
  can_raise: boolean
  min_full_raise_to: number | null
  max_raise_to: number | null
  short_all_in_raise_to: number | null
  all_in_raise_to: number | null
  pot_raise_to: number | null
  raise_blocked_reason: string | null
}

export interface PlayerView {
  seat: number
  stack: number
  street_committed: number
  hand_committed: number
  status: PlayerStatus
  shown_cards: string[] | null
}

export interface PotView {
  amount: number
  eligible: number[]
}

export interface PublicEvent {
  seq: number
  hand_id: string | null
  type: string
  data: Record<string, any>
}

export interface TableView {
  schema_version: number
  perspective: 'seat' | 'spectator'
  compatibility_status: 'practice_assumptions'
  table_id: string
  ruleset_id: string
  config_hash: string
  state_version: number
  hero_seat: number | null
  hole_cards: string[] | null
  seats: number[]
  initial_player_count: number
  surviving_player_count: number
  hand_id: string | null
  hand_number: number
  decision_id: string | null
  button_seat: number | null
  small_blind_seat: number | null
  big_blind_seat: number | null
  street: Street | null
  actor_seat: number | null
  board: string[]
  pot_total: number
  settled_pot: number
  current_bet_to: number
  pots: PotView[]
  players: PlayerView[]
  blinds: { small_blind: number; big_blind: number; ante?: number; ante_posting?: AntePosting } | null
  next_level: {
    small_blind: number
    big_blind: number
    starts_at_hand: number
    ante?: number
    ante_posting?: AntePosting
  } | null
  payouts: number[]
  hands_played: number
  hands_remaining: number
  hand_cap: number
  decision_budget: { bot_budget_ns: number; human_turn_ms: number }
  legal: LegalActions | null
  action_history: PublicEvent[]
  table_complete: boolean
  points: Record<string, number> | null
}

export interface SeatInfo {
  seat: number
  kind: 'human' | 'bot'
  policy: string | null
  label: string
  artifact_hash: string | null
  source: 'baseline' | 'registered' | 'style' | 'strategy' | null
  claimed: boolean
  you: boolean
  /** Set up hidden: before the reveal its policy, label and style are withheld by the server. */
  hidden?: boolean
  style?: string | null
}

export type StyleMode = 'labelled' | 'hidden' | 'mixed' | 'random'

export interface BookRef {
  part: string
  section: string
}

export type StatKey = 'vpip' | 'pfr' | 'aggression' | 'fold_to_reraise'

export interface StyleProfile {
  id: string
  policy_id: string
  name: string
  family: string
  description: string
  book_ref: BookRef
  expected: Partial<Record<StatKey, [number, number] | null>>
  counter_strategy: string
  tells: string[]
  advanced: boolean
  rulesets: string[]
}

export type ChartGroup = 'early' | 'middle' | 'late'

export interface StylesCatalog {
  styles: StyleProfile[]
  modes: StyleMode[]
  random_pool_default: string[]
  stat_definitions: Record<string, { label: string; book_ref?: BookRef; help?: string }>
  charts: Record<'conservative' | 'aggressive', Record<ChartGroup, string[]>>
  grid: string[][]
}

export interface StylesSetup {
  mode: StyleMode
  learning_aids: boolean
  random_pool: string[]
  hidden_seats: number[]
  revealed: boolean
}

export interface Ratio {
  count: number
  of: number
  rate: number | null
}

export interface Showdown {
  hand_number: number
  hand_id: string
  cards: string[]
  board: string[]
  hand_in_words: string
  line: Partial<Record<Street, string[]>>
  won: number
}

export interface SeatStats {
  seat: number
  hands: number
  vpip: Ratio
  pfr: Ratio
  rfi: Ratio
  aggression: { bets_raises: number; calls: number; factor: number | null }
  reraise: { faced: number; folded: number; called: number; reraised: number; fold_rate: number | null }
  showdowns: { count: number; items: Showdown[] }
}

export interface TableStats {
  hands_tracked: number
  seats: Record<string, SeatStats>
}

export interface Evidence {
  stat: string
  label: string
  value: number | null
  count?: number
  of?: number
  expected: [number, number] | null
  in_range: boolean | null
  book_ref: BookRef | null
}

export interface RevealSeat {
  seat: number
  kind: 'human' | 'bot'
  hidden: boolean
  random_pick?: boolean
  policy: string | null
  label: string
  style: string | null
  artifact_hash: string | null
  book_ref: BookRef | null
  summary: string | null
  evidence: Evidence[]
  showdowns: Showdown[]
}

export interface Reveal {
  lineup_hash: string
  lineup: { seat: number; kind: string; policy: string | null; artifact_hash: string | null }[]
  seats: RevealSeat[]
}

export type PositionLabel = 'UTG' | 'MP' | 'CO' | 'BTN' | 'SB' | 'BB'
export type PositionGroup = 'early' | 'middle' | 'late' | 'blinds'

export interface Aids {
  table_id: string
  version: number
  hand_id: string | null
  decision_id: string | null
  positions: Record<string, { label: PositionLabel; group: PositionGroup }>
  hero: {
    seat: number
    position: { label: PositionLabel; group: PositionGroup } | null
    hole_cards: string[]
    hand_class: string
    hand_in_words: string
    rank_row: string
    draws: { kind: 'flush_draw' | 'open_ended' | 'gutshot'; outs: number; words: string }[]
    outs: number
    charts: { group: ChartGroup; conservative: boolean; aggressive: boolean }
    equity_vs_random: { vs_one: number; vs_field: number; opponents: number; samples: number; exact: boolean }
    pot_odds: { to_call: number; pot: number; break_even: number; words: string } | null
    coach: { action: 'fold' | 'check' | 'call' | 'raise_to'; amount?: number; reason: string; heuristic: string; disclaimer: string } | null
  } | null
}

export interface ResolvedConfig {
  ruleset_id: string
  initial_player_count: number
  starting_stack: number
  hand_cap: number
  hands_per_level: number
  blind_levels: [number, number][]
  payout_points: number[]
  human_turn_ms: number
  bot_budget_ns: number
  [key: string]: unknown
}

export interface Placement {
  seat: number
  positions: number[]
  points: number
  method: 'elimination' | 'winner' | 'icm'
  hand_number: number
}

export interface Results {
  placements: Placement[]
  points: Record<string, number>
  payouts: number[]
  hands_played: number
  hand_cap: number
  eliminations: { seat: number; hand_number: number; positions: number[]; points: number }[]
  fallbacks: Record<string, Record<string, number>>
}

export interface Snapshot {
  table_id: string
  status: 'waiting' | 'running' | 'complete'
  version: number
  compatibility_status: 'practice_assumptions'
  viewer: { role: 'seat' | 'host' | 'anonymous'; seat: number | null }
  setup: { config: ResolvedConfig; config_hash: string; seats: SeatInfo[]; styles?: StylesSetup }
  view: TableView | null
  turn: {
    seat: number
    kind: 'human' | 'bot'
    decision_id: string
    turn_ms: number | null
    remaining_ms: number | null
  } | null
  last_hand: { hand_id: string; hand_number: number | null; events: PublicEvent[] } | null
  results: Results | null
  stats?: TableStats | null
  reveal?: Reveal | null
}

export interface Policy {
  id: string
  label: string
  description: string
  artifact_hash: string
  source: 'baseline' | 'registered' | 'style' | 'strategy'
  style?: string
  rulesets: string[]
  state_mode: string
  bot_name: string
  alias_of: string | null
  original_path?: string
  registered_at?: string
}

export type Perspective = 'seat' | 'spectator' | 'reviewer'

export interface ReplayFrame {
  index: number
  kind: 'deal' | 'action' | 'hand_end' | 'complete'
  seq: number
  hand_number: number
  view: TableView
  hole_cards: Record<string, string[]> | null
  bot_note?: { seat: number; decision_id: string; text: string } | null
}

export interface ReplayHand {
  hand_number: number
  hand_id: string | null
  frames: ReplayFrame[]
  events: PublicEvent[]
}

export interface Replay {
  table_id: string
  perspective: Perspective
  viewer_seat: number | null
  table_complete: boolean
  available_perspectives: Perspective[]
  hands: { hand_number: number; hand_id: string | null; frames: number }[]
  hand: ReplayHand
}

export interface Quantiles {
  p50: number | null
  p95: number | null
  p99: number | null
  max: number | null
}

export interface EntrantSummary {
  entrant: number
  name: string
  bot_name: string
  state_mode: string | null
  builtin: boolean
  tables: number
  mean_points: number | null
  points_ci95: [number, number] | null
  decisions: number
  fallbacks: Record<string, number>
  fallback_total: number
  timeout_fallbacks: number
  restarts: number
  timing: { samples: number; end_to_end_s: Quantiles; over_budget_samples: number; worker_wall_s: Quantiles; worker_cpu_s: Quantiles }
}

export interface CellManifest {
  run_id: string
  config: ResolvedConfig
  config_hash: string
  content_hash: string
  competitive: boolean
  isolation: 'sandbox' | 'in_process'
  budget_ns: number
  seed_blocks: number
  rotations_per_block: number
  scheduled_tables: number
  seed_list_sha256: string
  report_note: string
  budget_semantics: string
  code: Record<string, unknown>
  entrants: { entrant: number; name: string; artifact_sha256: string; builtin: boolean; rulesets?: string[]; state_mode?: string }[]
  mean_points_by_seat: Record<string, number>
  summary: {
    tables_scheduled: number
    tables_completed: number
    tables_failed: { table_id: string; error: string }[]
    tables_incomplete: string[]
    blocks: number
    entrants: EntrantSummary[]
    table_wall_s_mean: number | null
  }
}

export type RunStatus = 'queued' | 'running' | 'complete' | 'failed'

export interface RunCellView {
  cell_id: string
  ruleset_id: string
  players: number
  config_hash: string
  entrants: { name: string; builtin: boolean }[]
  status: RunStatus
  done: number
  total: number
  wall_s: number | null
  error: string | null
  manifest: CellManifest | null
}

export interface RunView {
  run_id: string
  status: RunStatus
  created_at: string
  request: { cells: { ruleset_id: string; players: number }[]; bots: string[]; blocks: number; isolation: string; jobs: number; config: Record<string, unknown> }
  bots: Policy[]
  progress: { done: number; total: number }
  cells: RunCellView[]
  error: string | null
}

export interface RunBrief {
  run_id: string
  status: RunStatus
  created_at: string
  progress: { done: number; total: number }
  error: string | null
  scenarios: string[]
  bots: string[]
}

export type Action =
  | { type: 'fold' }
  | { type: 'check' }
  | { type: 'call' }
  | { type: 'raise_to'; amount: number }

export type StudyFilter = 'all' | 'postflop' | 'showdown' | 'big_pot'
export type StudyKind = 'fold' | 'check' | 'call' | 'raise'

export interface StudyPack {
  id: string
  title: string
  hero: string
  ruleset_id: string
  hands: number
  decisions: number
  transform: string
  caveat: string
  filters: Record<StudyFilter, number>
  dataset: { citation: string; licence: string; repo: string; commit: string; paper_doi: string; dataset_doi: string }
}

export interface StudyHandSummary {
  id: string
  title: string
  position: PositionLabel
  hand: string
  decisions: number
  last_street: Street
  showdown: boolean
  pot_bb: number
  net_bb: number
}

export interface StudyHandPage {
  pack: string
  filter: StudyFilter
  total: number
  offset: number
  hands: StudyHandSummary[]
}

export interface StudyFrame {
  kind: 'deal' | 'action' | 'hand_end'
  seq: number
  street: Street | null
  actor_seat: number | null
  board: string[]
  pot_total: number
  settled_pot: number
  current_bet_to: number
  pots: PotView[]
  players: (Omit<PlayerView, 'shown_cards'> & { shown_cards?: string[] })[]
}

export interface StudyOption {
  key: string
  kind: StudyKind
  amount: number | null
  size?: string
  fraction?: number
  bet?: boolean
}

export interface StudySpot {
  position: PositionLabel
  in_position: boolean
  facing: string
  pot: number
  to_call: number
  pot_odds: number
  pot_bb: number
  stack_bb: number
  effective_bb: number
  spr: number
  opponents: number
  behind: number
  raises: number
  hand: string
  preflop_percentile: number
  equity: number
  equity_source: 'table' | 'parent' | 'prior'
  made: string | null
  outs: number
  draws: string[]
  board: string[]
}

export interface StudyDecision {
  frame: number
  street: Street
  options: StudyOption[]
  pluribus: { kind: StudyKind; amount: number | null; option: string; exact?: boolean; fraction?: number; size?: string }
  spot: StudySpot
}

export interface StudyHand {
  id: string
  title: string
  ruleset_id: string
  blinds: { small_blind: number; big_blind: number }
  hero_seat: number
  hero_name: string
  button_seat: number
  small_blind_seat: number | null
  big_blind_seat: number | null
  seats: { seat: number; name: string; position: PositionLabel; start_stack: number }[]
  hole_cards: string[]
  frames: StudyFrame[]
  events: PublicEvent[]
  decisions: StudyDecision[]
  result: {
    final_stacks: Record<string, number>
    net: Record<string, number>
    hole_cards: Record<string, string[]>
    board: string[]
    showdown: number[]
    notes: string[]
  }
}

export type AntePosting = 'classic' | 'big_blind'

export interface AdvisorSetup {
  players: number
  hero: number
  sb: number
  bb: number
  ante: number
  ante_posting: AntePosting | null
  mode: 'cash' | 'tournament'
  payouts: number[]
  stacks: number[]
}

export interface AdvisorSeat {
  seat: number
  hero: boolean
  button: boolean
  position: string | null
  in_hand: boolean
  stack_start: number
  stack: number
  street_committed: number
  status: string
}

export interface AdvisorAction {
  type: 'fold' | 'check' | 'call' | 'raise_to'
  amount?: number
}

export interface AdvisorAdvice {
  headline: string
  pick: AdvisorAction
  mix: { label: string; action: AdvisorAction; pct: number }[]
  mixed: boolean
  reason: string
  source: string
  position: string
  effective_bb: number
}

export interface AdvisorLegal {
  fold: boolean
  check: boolean
  call_amount: number
  can_raise: boolean
  min_raise_to: number | null
  max_raise_to: number | null
  short_all_in_raise_to: number | null
  pot_raise_to: number | null
  all_in_raise_to: number | null
  blocked: string | null
}

export interface AdvisorView {
  session: string
  setup: AdvisorSetup
  hand_number: number
  button: number
  blinds: { sb: number; bb: number; ante: number; ante_posting: AntePosting | null }
  stage: 'hole' | 'action' | 'board' | 'complete'
  street: string
  board: string[]
  hole: string[] | null
  used_cards: string[]
  pot: number
  seats: AdvisorSeat[]
  actor: number | null
  legal: AdvisorLegal | null
  need_board: number | null
  street_needed: string | null
  advice: AdvisorAdvice | null
  log: string[]
  result: {
    ended_by_fold: boolean
    resolved: boolean
    winner: number | null
    pots: { amount: number; eligible: number[] }[]
    awards: number[][] | null
    stacks_after: Record<string, number>
  } | null
  can_undo: boolean
  editable: boolean
  next_stacks: Record<string, number>
}

export type AdvisorOp = { op: string } & Record<string, unknown>
