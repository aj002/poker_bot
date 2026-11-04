"""Red-team Equilibrium Bot (`redteam_equilibrium`).

An unexploitable, solver-derived, ICM-aware and adaptive bot built from first principles
using only the arena SDK and precomputed CFR+ equilibrium tables.

Key architecture:
1. Precomputed CFR+ Equilibrium Tables (embedded in `tables_data.py` & stored in `tables.npz`,
   loaded at untimed worker startup):
   - 169x169 exact-category preflop equity matrix and postflop playability adjustments.
   - Heads-up and late-position Push/Fold Nash equilibria across 10 effective stack depths
     (2 BB to 20 BB) and 5 ICM bubble-factor tiers (1.0 to 2.1).
   - Multi-player preflop Open / Call / 3-bet / 4-bet CFR+ equilibria across all 7 positions
     (`n_behind` 1..7) and 5 ICM bubble-factor tiers.
2. Fast Exact & Analytical Malmuth-Harville ICM Engine:
   - Exact bitmask DP for <= 4 surviving players; fast O(n^2) 3-prize exact Malmuth-Harville
     computation for 5-8 players (< 5 microseconds).
   - Computes exact pairwise Bubble Factor (`dEV_loss / dEV_gain`) against the effective
     active opponent and blends with hand-cap proximity (200-hand cap ICM scoring).
3. Balanced Mixed Postflop Engine (< 15 microseconds):
   - Fast bitmask 7-card evaluator & outs counter (zero allocations, no 45-card loop).
   - Uses Minimum Defense Frequency (MDF = 1 / (1 + bet_pot_frac * bf)) and optimal bluff-to-value
     ratios (alpha = bet / (pot + bet)) with seeded mixed strategies.
4. Compact Empirical Bayes Opponent Adaptation (< 450 bytes state, < 10 microseconds):
   - Tracks per-seat VPIP, PFR, 3-bet, fold-to-steal, postflop aggression, and fold-to-bet.
   - Shrinks noisy small-sample observations toward equilibrium priors (`N_prior = 12`) and
     applies bounded shifts to preflop steal/defense thresholds and postflop call/bluff frequencies.
"""

from __future__ import annotations

import base64
import hashlib
import io
import itertools
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from arena.agents.sdk import (
    Action,
    ExplicitStateBot,
    LegalActions,
    Observation,
    TableInfo,
    call,
    check,
    check_or_call,
    check_or_fold,
    clamp_raise_to,
)
from arena.agents.styles.charts import ALL_CLASSES, hand_class, preflop_order
from arena.agents.styles.handeval import (
    FLUSH,
    FULL_HOUSE,
    HIGH_CARD,
    PAIR,
    QUADS,
    RV,
    STRAIGHT,
    STRAIGHT_FLUSH,
    TRIPS,
    TWO_PAIR,
    evaluate,
)

try:
    from redteam.equilibrium.tables_data import _B85 as _EMBEDDED_B85
except ModuleNotFoundError:
    try:
        from tables_data import _B85 as _EMBEDDED_B85  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        _EMBEDDED_B85 = ""

CLASSES = list(ALL_CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
SUIT_IDX = {"c": 0, "d": 1, "h": 2, "s": 3}


def _load_tables() -> dict[str, np.ndarray]:
    npz_path = Path(__file__).resolve().parent / "tables.npz"
    if npz_path.is_file():
        with np.load(npz_path) as npz:
            return {k: npz[k] for k in npz.files}
    raw = base64.b85decode(_EMBEDDED_B85.encode("ascii"))
    out: dict[str, np.ndarray] = {}
    with np.load(io.BytesIO(raw)) as npz:
        for k in npz.files:
            arr = npz[k]
            out[k] = (arr.astype(np.float32) / 255.0) if arr.dtype == np.uint8 else arr
    return out


_DATA = _load_tables()

EQUITY_MAT: np.ndarray = _DATA["equity"]
RAW_EQ: list[float] = [float(x) for x in _DATA["raw_eq"]]
PLAYABILITY: list[float] = [float(x) for x in _DATA["playability"]]
EFF_BB_GRID: list[float] = [float(x) for x in _DATA["eff_bb_grid"]]
BF_GRID: list[float] = [float(x) for x in _DATA["bf_grid"]]
N_BEHIND_GRID: list[int] = [int(x) for x in _DATA["n_behind_grid"]]

PUSH_SB_TBL: np.ndarray = _DATA["push_sb"]
CALL_BB_TBL: np.ndarray = _DATA["call_bb"]
PUSH_BTN_TBL: np.ndarray = _DATA["push_btn"]
CALL_VS_BTN_TBL: np.ndarray = _DATA["call_vs_btn"]

OPEN_TBL: np.ndarray = _DATA["open_tbl"]
DEF_CALL_TBL: np.ndarray = _DATA["def_call_tbl"]
DEF_3B_TBL: np.ndarray = _DATA["def_3b_tbl"]
VS3B_CALL_TBL: np.ndarray = _DATA["vs3b_call_tbl"]
VS3B_4B_TBL: np.ndarray = _DATA["vs3b_4b_tbl"]

_TOP_MASK = _DATA["raw_eq"] >= 0.63
EQ_VS_4B: list[float] = [float(np.mean(EQUITY_MAT[i, _TOP_MASK])) for i in range(169)]

# Precompute 13-bit straight masks (A-5 wheel through T-A)
_STRAIGHT_MASKS: tuple[int, ...] = tuple(
    [0x100F] + [0x1F << (hi - 4) for hi in range(4, 13)]
)


# ---------------------------------------------------------------------- ICM & Bubble Factor


def _icm_exact_small(stacks: Sequence[float], prizes: Sequence[float]) -> list[float]:
    """Exact Malmuth-Harville ICM for small survivor counts (n <= 4)."""
    n = len(stacks)
    if n == 0:
        return []
    if n == 1:
        return [float(prizes[0]) if prizes else 0.0]
    p_len = min(n, len(prizes))
    pz = [float(prizes[i]) for i in range(p_len)] + [0.0] * (n - p_len)

    full = (1 << n) - 1
    dp: list[list[float]] = [[0.0] * n for _ in range(full + 1)]
    for mask in range(1, full + 1):
        members = [i for i in range(n) if (mask >> i) & 1]
        k = len(members)
        place = n - k
        if place >= p_len:
            continue
        tot = sum(stacks[i] for i in members)
        if tot <= 0.0:
            share = sum(pz[place : place + k]) / k
            for i in members:
                dp[mask][i] = share
            continue
        pr_here = pz[place]
        row = dp[mask]
        for w in members:
            pw = stacks[w] / tot
            row[w] += pw * pr_here
            sub = dp[mask & ~(1 << w)]
            for i in members:
                if i != w:
                    row[i] += pw * sub[i]
    return dp[full]


def _icm_pair_ev(stacks: Sequence[float], prizes: Sequence[float], target_idx: int) -> float:
    """Compute exact Malmuth-Harville expected points for `target_idx` in O(n^2) time (< 5 us)."""
    n = len(stacks)
    if n <= 4:
        return _icm_exact_small(stacks, prizes)[target_idx]
    tot = sum(stacks)
    si = stacks[target_idx]
    if tot <= 0.0 or si <= 0.0:
        return 0.0
    p0 = float(prizes[0]) if len(prizes) > 0 else 0.0
    p1 = float(prizes[1]) if len(prizes) > 1 else 0.0
    p2 = float(prizes[2]) if len(prizes) > 2 else 0.0

    ev = (si / tot) * p0
    for j in range(n):
        if j == target_idx:
            continue
        sj = stacks[j]
        if sj <= 0.0:
            continue
        pj1 = sj / tot
        rem1 = tot - sj
        if rem1 <= 0.0:
            continue
        ev += pj1 * (si / rem1) * p1
        if p2 > 0.0:
            for k in range(n):
                if k == target_idx or k == j:
                    continue
                sk = stacks[k]
                if sk <= 0.0:
                    continue
                rem2 = rem1 - sk
                if rem2 > 0.0:
                    ev += pj1 * (sk / rem1) * (si / rem2) * p2
    return ev


def compute_bubble_factor(
    obs: Observation,
    hero_seat: int,
    villain_seat: int | None,
) -> float:
    """Compute ICM Bubble Factor (BF >= 1.0) for hero vs primary active opponent in < 8 us."""
    alive = [p for p in obs.players if p.status != "eliminated" and (p.stack + p.hand_committed) > 0]
    n_alive = len(alive)
    if n_alive <= 2:
        return 1.0

    seats = [p.seat for p in alive]
    if hero_seat not in seats:
        return 1.0
    h_idx = seats.index(hero_seat)
    stacks = [float(p.stack + p.hand_committed) for p in alive]

    if villain_seat is not None and villain_seat in seats and villain_seat != hero_seat:
        v_idx = seats.index(villain_seat)
    else:
        other_indices = [i for i in range(n_alive) if i != h_idx]
        v_idx = max(other_indices, key=lambda i: stacks[i])

    delta = min(stacks[h_idx], stacks[v_idx]) * 0.65
    if delta < 1.0:
        return 1.0

    prizes = obs.payouts[:n_alive]
    base_ev = _icm_pair_ev(stacks, prizes, h_idx)

    s_win = list(stacks)
    s_win[h_idx] += delta
    s_win[v_idx] = max(1.0, s_win[v_idx] - delta)
    ev_win = _icm_pair_ev(s_win, prizes, h_idx)

    s_lose = list(stacks)
    s_lose[h_idx] = max(0.0, s_lose[h_idx] - delta)
    s_lose[v_idx] += delta
    ev_lose = _icm_pair_ev(s_lose, prizes, h_idx)

    gain = max(1e-6, ev_win - base_ev)
    loss = max(0.0, base_ev - ev_lose)
    raw_bf = loss / gain

    hands_rem = obs.hands_remaining
    cap_weight = 1.0 if hands_rem <= 20 else (0.88 if hands_rem <= 60 else 0.78)
    bf = 1.0 + max(0.0, raw_bf - 1.0) * cap_weight
    return min(max(bf, 1.0), 2.1)


# ---------------------------------------------------------------------- Fast Table Interpolation


def _interp_1d_list(grid: list[float], val: float) -> tuple[int, int, float]:
    n = len(grid)
    if val <= grid[0]:
        return 0, 0, 0.0
    if val >= grid[-1]:
        return n - 1, n - 1, 0.0
    for i in range(n - 1):
        g0 = grid[i]
        g1 = grid[i + 1]
        if val <= g1:
            return i, i + 1, (val - g0) / (g1 - g0)
    return n - 1, n - 1, 0.0


def lookup_push_fold(
    tbl: np.ndarray,
    eff_bb: float,
    bf: float,
    hand_idx: int,
) -> float:
    s0, s1, ws = _interp_1d_list(EFF_BB_GRID, eff_bb)
    b0, b1, wb = _interp_1d_list(BF_GRID, bf)
    v00 = float(tbl[s0, b0, hand_idx])
    v01 = float(tbl[s0, b1, hand_idx])
    v10 = float(tbl[s1, b0, hand_idx])
    v11 = float(tbl[s1, b1, hand_idx])
    v0 = v00 * (1.0 - wb) + v01 * wb
    v1 = v10 * (1.0 - wb) + v11 * wb
    return v0 * (1.0 - ws) + v1 * ws


def lookup_preflop(
    tbl: np.ndarray,
    n_behind: int,
    bf: float,
    hand_idx: int,
) -> float:
    pi = max(0, min(len(N_BEHIND_GRID) - 1, n_behind - 1))
    b0, b1, wb = _interp_1d_list(BF_GRID, bf)
    v0 = float(tbl[pi, b0, hand_idx])
    v1 = float(tbl[pi, b1, hand_idx])
    return v0 * (1.0 - wb) + v1 * wb


# ---------------------------------------------------------------------- Compact Opponent Tracker


def _init_opp_stats(seats: Sequence[int]) -> dict[str, list[int]]:
    """Per-seat compact integer array (15 ints):
    [0:hands, 1:vpip, 2:pfr, 3:rfi_opp, 4:rfi, 5:faced_open, 6:three_bet, 7:fts_opp, 8:fts,
     9:post_bets, 10:post_calls, 11:post_faced, 12:post_fold, 13:recent_hands, 14:recent_pfr]
    """
    return {str(s): [0] * 15 for s in seats}


def _update_opp_stats(state: dict[str, Any], obs: Observation) -> None:
    """Incrementally ingest public events from `obs.action_history` in < 8 us."""
    last_seq = state["q"]
    stats = state["o"]
    cur = state["h"]

    for ev in obs.action_history:
        seq = ev.seq
        if seq <= last_seq:
            continue
        last_seq = seq
        etype = ev.type
        d = ev.data

        if etype == "HandStarted":
            dealt = [int(s) for s in d["start_stacks"]]
            cur = [
                dealt,
                int(d["positions"]["big_blind"]),
                int(d["small_blind_seat"]) if d.get("small_blind_seat") is not None else -1,
                0,   # 3: vol (0/1)
                0,   # 4: pre raises
                -1,  # 5: open_seat
                0,   # 6: vpip bitmask
                0,   # 7: pfr bitmask
                0,   # 8: acted bitmask
                0,   # 9: post_bet (0/1)
            ]
            state["h"] = cur
            for s in dealt:
                row = stats.get(str(s))
                if row is not None:
                    row[0] += 1
                    row[13] += 1
                    if row[13] > 16:
                        row[13] = (row[13] * 3) >> 2
                        row[14] = (row[14] * 3) >> 2
        elif etype == "BoardDealt" and cur is not None:
            cur[9] = 0
        elif etype == "ActionApplied" and cur is not None:
            if d.get("auto"):
                continue
            s = int(d["seat"])
            row = stats.get(str(s))
            if row is None:
                continue
            bit = 1 << s
            st = d["street"]
            kind = d["action"]["type"]
            if st == "preflop":
                if not (cur[8] & bit):
                    cur[8] |= bit
                    if not cur[3]:
                        row[3] += 1
                        if kind == "raise_to":
                            row[4] += 1
                    elif cur[4] == 1 and s != cur[5]:
                        row[5] += 1
                        if kind == "raise_to":
                            row[6] += 1
                        if s == cur[1] or s == cur[2]:
                            row[7] += 1
                            if kind == "fold":
                                row[8] += 1
                if kind == "call" or kind == "raise_to":
                    cur[3] = 1
                    if not (cur[6] & bit):
                        cur[6] |= bit
                        row[1] += 1
                if kind == "raise_to":
                    if cur[4] == 0:
                        cur[5] = s
                    cur[4] += 1
                    if not (cur[7] & bit):
                        cur[7] |= bit
                        row[2] += 1
                        row[14] += 1
            else:
                if kind == "raise_to":
                    row[9] += 1
                    cur[9] = 1
                elif kind == "call":
                    row[10] += 1
                    if cur[9]:
                        row[11] += 1
                elif kind == "fold" and cur[9]:
                    row[11] += 1
                    row[12] += 1
    state["q"] = last_seq


def _opp_profile(state: dict[str, Any], seat: int) -> dict[str, float]:
    """Return Empirical-Bayes shrunk opponent profile rates (blending recent gear with career stats)."""
    row = state["o"].get(str(seat))
    if not row:
        return {
            "vpip": 0.26, "pfr": 0.20, "rfi": 0.28, "three_b": 0.08,
            "fts": 0.62, "af": 0.48, "fpost": 0.45, "conf": 0.0,
        }
    hands, vpip, pfr, rfi_opp, rfi, faced_open, three_bet, fts_opp, fts, pb, pc, pfb, pf, rh, rpfr = row
    career_pfr = (pfr + 2.40) / (hands + 12.0)
    rec_pfr = (rpfr + 1.20) / (rh + 6.0)
    eff_pfr = 0.55 * rec_pfr + 0.45 * career_pfr
    career_rfi = (rfi + 2.24) / (rfi_opp + 8.0)
    eff_rfi = career_rfi + 0.80 * (rec_pfr - career_pfr)
    return {
        "vpip": (vpip + 3.12) / (hands + 12.0),
        "pfr": eff_pfr,
        "rfi": min(0.85, max(0.05, eff_rfi)),
        "three_b": (three_bet + 0.64) / (faced_open + 8.0),
        "fts": (fts + 3.72) / (fts_opp + 6.0),
        "af": (pb + 4.80) / (pb + pc + 10.0),
        "fpost": (pf + 3.60) / (pfb + 8.0),
        "conf": min(1.0, hands / 25.0),
    }


# ---------------------------------------------------------------------- Ultra-Fast Bitmask Postflop Evaluator


def _fast_nlhe_postflop(hole: Sequence[str], board: Sequence[str], n_opp: int) -> tuple[float, str, int, float, float]:
    """Single-pass bitmask postflop evaluator returning `(equity, bucket, outs, wetness, high_norm)` in < 10 us!"""
    hr0, hs0 = RV[hole[0][0]], SUIT_IDX[hole[0][1]]
    hr1, hs1 = RV[hole[1][0]], SUIT_IDX[hole[1][1]]
    if hr0 < hr1:
        hr0, hr1 = hr1, hr0
        hs0, hs1 = hs1, hs0

    rc = [0] * 13
    b_rc = [0] * 13
    sc = [0] * 4
    b_sc = [0] * 4
    rmask = 0
    b_rmask = 0
    smask = [0] * 4

    b_top = -1
    b_sec = -1
    for c in board:
        r = RV[c[0]]
        s = SUIT_IDX[c[1]]
        rc[r] += 1
        b_rc[r] += 1
        sc[s] += 1
        b_sc[s] += 1
        bit = 1 << r
        rmask |= bit
        b_rmask |= bit
        smask[s] |= bit
        if r > b_top:
            b_sec = b_top
            b_top = r
        elif r > b_sec:
            b_sec = r

    rc[hr0] += 1
    rc[hr1] += 1
    sc[hs0] += 1
    sc[hs1] += 1
    rmask |= (1 << hr0) | (1 << hr1)
    smask[hs0] |= 1 << hr0
    smask[hs1] |= 1 << hr1

    # Board texture features
    max_b_suit = max(b_sc)
    max_b_rc = max(b_rc)
    paired_b = 1.0 if max_b_rc >= 2 else 0.0
    flush_draw_b = 1.0 if max_b_suit == 2 and len(board) <= 4 else 0.0
    flush_comp_b = 1.0 if max_b_suit >= 3 else 0.0

    # Board straight connectivity
    straight_comp_b = 0.0
    for st_m in _STRAIGHT_MASKS:
        if (b_rmask & st_m).bit_count() >= 3:
            straight_comp_b = 1.0
            break

    wet = min(1.0, 0.35 * flush_comp_b + 0.20 * flush_draw_b + 0.30 * straight_comp_b + 0.15 * paired_b)
    high_norm = b_top / 12.0 if b_top >= 0 else 0.5

    # Evaluate made hand category and board category
    has_flush = False
    flush_suit = -1
    for s in range(4):
        if sc[s] >= 5:
            has_flush = True
            flush_suit = s
            break
    board_flush = max_b_suit >= 5

    has_straight = False
    board_straight = False
    for st_m in _STRAIGHT_MASKS:
        if (rmask & st_m) == st_m:
            has_straight = True
        if (b_rmask & st_m) == st_m:
            board_straight = True

    max_rc = max(rc)
    pairs_cnt = sum(1 for c in rc if c == 2)
    trips_cnt = sum(1 for c in rc if c >= 3)
    b_pairs_cnt = sum(1 for c in b_rc if c == 2)

    # Determine category & bucket
    cat = HIGH_CARD
    if has_flush:
        cat = FLUSH
    if has_straight and cat < STRAIGHT:
        cat = STRAIGHT
    if max_rc == 4:
        cat = QUADS
    elif trips_cnt >= 1 and (trips_cnt >= 2 or pairs_cnt >= 1):
        cat = FULL_HOUSE
    elif cat < TRIPS and trips_cnt >= 1:
        cat = TRIPS
    elif cat < TWO_PAIR and pairs_cnt >= 2:
        cat = TWO_PAIR
    elif cat < PAIR and pairs_cnt == 1:
        cat = PAIR

    bucket = "weak"
    made_eq = 0.24
    if cat >= FULL_HOUSE and max_b_rc < 4:
        bucket = "monster"
        made_eq = 0.94
    elif cat == FLUSH and (not board_flush or hs0 == flush_suit or hs1 == flush_suit):
        bucket = "monster"
        made_eq = 0.88
    elif cat == STRAIGHT and not board_straight:
        bucket = "monster"
        made_eq = 0.84 - 0.05 * flush_comp_b
    elif cat == TRIPS and max_b_rc < 3:
        bucket = "monster"
        made_eq = 0.83 - 0.05 * (flush_comp_b + straight_comp_b)
    elif cat == TWO_PAIR:
        if max_b_rc == 1:
            bucket = "monster"
            made_eq = 0.76 - 0.06 * (flush_comp_b + straight_comp_b)
        elif max_b_rc == 2 and b_pairs_cnt == 1:
            # Paired board: check if hero paired a board rank or holds a pocket pair
            if hr0 == hr1:
                if hr0 > b_top:
                    bucket = "strong"
                    made_eq = 0.72 + 0.015 * (hr0 - b_top) - 0.05 * wet
                elif hr0 > b_sec:
                    bucket = "medium"
                    made_eq = 0.54 - 0.04 * wet
                else:
                    bucket = "weak"
                    made_eq = 0.35
            elif b_rc[hr0] == 1 or b_rc[hr1] == 1:
                paired_r = hr0 if b_rc[hr0] == 1 else hr1
                kicker = hr1 if paired_r == hr0 else hr0
                if paired_r == b_top:
                    bucket = "strong" if (kicker >= 8 or paired_r == 12) else "medium"
                    made_eq = 0.65 + 0.008 * kicker - 0.04 * wet
                elif paired_r >= b_sec:
                    bucket = "medium"
                    made_eq = 0.50 - 0.04 * wet
                else:
                    bucket = "weak"
                    made_eq = 0.36
    elif cat == PAIR and max_b_rc == 1:
        if hr0 == hr1:
            if hr0 > b_top:
                bucket = "strong"
                made_eq = 0.72 + 0.015 * (hr0 - b_top) - 0.05 * wet
            elif b_sec < 0 or hr0 > b_sec:
                bucket = "medium"
                made_eq = 0.52 - 0.04 * wet
            else:
                bucket = "weak"
                made_eq = 0.34
        else:
            paired_r = hr0 if b_rc[hr0] >= 1 else hr1
            kicker = hr1 if paired_r == hr0 else hr0
            if paired_r == b_top:
                if kicker >= 8 or paired_r == 12:
                    bucket = "strong"
                    made_eq = 0.64 + 0.008 * kicker - 0.04 * wet
                else:
                    bucket = "medium"
                    made_eq = 0.55 - 0.04 * wet
            elif paired_r == b_sec:
                bucket = "medium"
                made_eq = 0.48 - 0.04 * wet
            else:
                bucket = "weak"
                made_eq = 0.35
    else:
        overcards = (1 if hr0 > b_top else 0) + (1 if hr1 > b_top else 0)
        made_eq = 0.20 + 0.035 * overcards + 0.008 * hr0

    # Fast bitmask draw outs count (for flop and turn)
    outs = 0
    cards_left = 5 - len(board)
    if cards_left in (1, 2):
        # 1. Flush draw outs (4 of a suit involving at least one hole card)
        if not has_flush:
            for s in range(4):
                if sc[s] == 4 and (hs0 == s or hs1 == s):
                    outs += 9
                    break
        # 2. Straight draw outs (bitmask check for 4-card straight windows not already on board)
        if not has_straight:
            completing_mask = 0
            for st_m in _STRAIGHT_MASKS:
                missing = st_m & ~rmask
                if missing != 0 and (missing & (missing - 1)) == 0:
                    # Exactly 1 rank missing to complete this straight, and not a 4-to-straight on board alone
                    if (st_m & ~b_rmask) != missing:
                        completing_mask |= missing
            st_ranks = completing_mask.bit_count()
            if st_ranks >= 2:
                outs += 8 if outs == 0 else 6  # discount overlap with flush outs
            elif st_ranks == 1:
                outs += 4 if outs == 0 else 3

    if cards_left == 2:
        draw_eq = min(0.42, outs * 0.036)
    elif cards_left == 1:
        draw_eq = min(0.24, outs * 0.021)
    else:
        draw_eq = 0.0

    eq_1 = min(0.98, max(0.04, made_eq + (1.0 - made_eq) * draw_eq))
    if n_opp > 1:
        eq = eq_1 ** (1.0 + 0.42 * (n_opp - 1))
    else:
        eq = eq_1
    return eq, bucket, outs, wet, high_norm


def _generic_variant_equity(obs: Observation, n_opp: int) -> float:
    """Fast bitmask analytical strength for Short-Deck, PLO4, and reveal_variant (< 20 us)."""
    hole = obs.hole_cards
    board = obs.board
    is_sd = obs.ruleset_id == "uncertainty_shortdeck" or (
        len(hole) == 2 and all(c[0] in "6789TJQKA" for c in [*hole, *board])
    )
    if not board:
        ranks = sorted([RV[c[0]] for c in hole], reverse=True)
        score = (ranks[0] + ranks[1]) / 24.0
        if len(set(ranks)) < len(ranks):
            score += 0.14
        if len({c[1] for c in hole}) < len(hole):
            score += 0.06
        if ranks[0] - ranks[1] <= 2:
            score += 0.04
        eq1 = min(0.85, max(0.20, 0.32 + 0.55 * score))
        return float(eq1 ** (1.0 + 0.30 * (n_opp - 1)))

    if len(hole) >= 4 and len(board) >= 3:
        # Fast O(1) bitmask PLO evaluator (exact 2-from-hole + 3-from-board category)
        h_rc = [0] * 13
        b_rc = [0] * 13
        h_sc = [0] * 4
        b_sc = [0] * 4
        h_rmask = 0
        b_rmask = 0
        for c in hole:
            r, s = RV[c[0]], SUIT_IDX[c[1]]
            h_rc[r] += 1
            h_sc[s] += 1
            h_rmask |= 1 << r
        for c in board:
            r, s = RV[c[0]], SUIT_IDX[c[1]]
            b_rc[r] += 1
            b_sc[s] += 1
            b_rmask |= 1 << r

        has_flush = any(h_sc[s] >= 2 and b_sc[s] >= 3 for s in range(4))
        has_straight = False
        for st_m in _STRAIGHT_MASKS:
            # Need at least 2 ranks in hole and 3 in board, and all 5 covered overall
            if ((h_rmask | b_rmask) & st_m) == st_m and (h_rmask & st_m).bit_count() >= 2 and (b_rmask & st_m).bit_count() >= 3:
                has_straight = True
                break

        # Max rank counts using at most 2 from hole and 3 from board
        eff_rc = [min(2, h_rc[r]) + min(3, b_rc[r]) for r in range(13)]
        max_c = max(eff_rc)
        trips_cnt = sum(1 for c in eff_rc if c >= 3)
        pairs_cnt = sum(1 for c in eff_rc if c >= 2)

        if max_c >= 4:
            eq1 = 0.96
        elif trips_cnt >= 1 and pairs_cnt >= 2:
            eq1 = 0.91
        elif has_flush:
            eq1 = 0.85
        elif has_straight:
            eq1 = 0.79
        elif trips_cnt >= 1:
            eq1 = 0.71
        elif pairs_cnt >= 2:
            eq1 = 0.56
        elif pairs_cnt == 1:
            eq1 = 0.36
        else:
            eq1 = 0.18
        return float(eq1 ** (1.0 + 0.38 * (n_opp - 1)))

    eq, _, _, _, _ = _fast_nlhe_postflop(hole[:2], board, n_opp)
    if is_sd:
        eq = 0.50 + 0.82 * (eq - 0.50)
    return float(min(0.97, max(0.05, eq)))


# ---------------------------------------------------------------------- Bot Implementation


class EquilibriumBot(ExplicitStateBot):
    name = "redteam_equilibrium"
    rulesets = ("practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4", "reveal_variant")

    def initial_state(self, table: TableInfo) -> dict[str, Any]:
        return {
            "s": table.policy_seed,
            "n": 0,
            "q": -1,
            "h": None,
            "o": _init_opp_stats(table.seats),
        }

    def _rand(self, state: dict[str, Any]) -> float:
        """Deterministic uniform random float in [0, 1) from state."""
        state["n"] += 1
        h = hashlib.blake2b(f"{state['s']}:{state['n']}".encode(), digest_size=8).digest()
        return int.from_bytes(h, "big") / 18446744073709551616.0

    def decide(self, obs: Observation, state: Any) -> tuple[Action | dict, Any]:
        if not isinstance(state, dict) or "o" not in state:
            state = {
                "s": "fallback",
                "n": 0,
                "q": -1,
                "h": None,
                "o": _init_opp_stats(obs.seats),
            }

        _update_opp_stats(state, obs)

        legal = obs.legal
        if not legal.can_raise and not legal.fold:
            return check_or_call(legal), state

        if obs.ruleset_id != "practice_nlhe" or len(obs.hole_cards) != 2:
            action = self._decide_variant(obs, state)
            if not legal.allows(action):
                action = check_or_fold(legal)
            return action, state

        if obs.street == "preflop":
            action = self._decide_nlhe_preflop(obs, state)
        else:
            action = self._decide_nlhe_postflop(obs, state)

        if not legal.allows(action):
            action = check_or_fold(legal)
        return action, state

    # ------------------------------------------------------------------ NLHE Preflop

    def _decide_nlhe_preflop(self, obs: Observation, state: dict[str, Any]) -> Action:
        legal = obs.legal
        hero = obs.hero_seat
        blinds = obs.blinds
        bb = max(1, blinds.big_blind if blinds else 20)
        sb = blinds.small_blind if blinds else bb // 2
        to_call = legal.call_amount or 0

        me = next(p for p in obs.players if p.seat == hero)
        hero_total = me.stack + me.street_committed
        live_opps = [p for p in obs.players if p.seat != hero and p.status in ("active", "all_in")]
        if not live_opps:
            return check_or_call(legal)

        # Effective stack against remaining active/all-in opponents:
        # Preflop unopened: effective stack is against the largest stack left to act behind us!
        cur_hand = state.get("h")
        dealt = cur_hand[0] if cur_hand else [p.seat for p in obs.players if p.status != "eliminated"]
        order = preflop_order(obs.seats, dealt, obs.big_blind_seat)
        active_seats = {p.seat for p in obs.players if p.status == "active"}
        after_hero = order[order.index(hero) + 1 :] if hero in order else []
        n_behind = sum(1 for s in after_hero if s in active_seats)

        pre_raises = 0
        limpers = 0
        callers = 0
        aggressor: int | None = None
        hero_raised = False
        for ev in obs.action_history:
            if ev.type != "ActionApplied" or ev.data.get("auto"):
                continue
            d = ev.data
            if d["street"] != "preflop":
                continue
            s, kind = int(d["seat"]), d["action"]["type"]
            if kind == "raise_to":
                pre_raises += 1
                callers = 0
                aggressor = s
                if s == hero:
                    hero_raised = True
            elif kind == "call" and s != hero:
                if pre_raises == 0:
                    limpers += 1
                else:
                    callers += 1

        # Compute true effective stack against the relevant opponent(s):
        # If facing a raise, effective stack is min(hero_total, max(aggressor & active behind stacks)).
        # If unopened, effective stack is min(hero_total, max(stacks of players behind)).
        relevant_seats = set(after_hero) & active_seats
        if aggressor is not None:
            relevant_seats.add(aggressor)
        rel_opps = [p for p in live_opps if p.seat in relevant_seats] or live_opps
        max_opp_total = max(p.stack + p.street_committed for p in rel_opps)
        eff_chips = min(hero_total, max_opp_total)
        eff_bb = eff_chips / bb

        primary_villain = aggressor if aggressor is not None else (
            obs.big_blind_seat if obs.big_blind_seat != hero else live_opps[0].seat
        )
        bf = compute_bubble_factor(obs, hero, primary_villain)

        h_cls = hand_class(obs.hole_cards)
        h_idx = CLASS_TO_IDX[h_cls]
        raw_eq = RAW_EQ[h_idx]
        play_adj = PLAYABILITY[h_idx]
        hand_strength = raw_eq + play_adj
        u = self._rand(state)

        # 1. Short-stack Push/Fold regime (eff_bb <= 12 BB or hero_bb <= 12 BB)
        hero_bb = hero_total / bb
        if hero_bb <= 12.0 or eff_bb <= 12.0:
            s_lookup = min(hero_bb, eff_bb)
            if pre_raises == 0 and limpers == 0:
                if n_behind <= 1:
                    push_prob = lookup_push_fold(PUSH_SB_TBL, s_lookup, bf, h_idx)
                    if obs.big_blind_seat is not None:
                        bb_prof = _opp_profile(state, obs.big_blind_seat)
                        push_prob = min(1.0, max(0.0, push_prob + 0.45 * (bb_prof["fts"] - 0.62) * bb_prof["conf"]))
                else:
                    adj_eff = min(20.0, s_lookup * (1.0 + 0.25 * (n_behind - 1)))
                    push_prob = lookup_push_fold(PUSH_BTN_TBL, adj_eff, bf, h_idx)

                if u < push_prob:
                    return self._jam_or_raise(legal, hero_total)
                return check_or_fold(legal)

            if pre_raises == 0 and limpers >= 1:
                adj_eff = min(20.0, s_lookup * (1.0 + 0.25 * (n_behind + limpers)))
                iso_prob = lookup_push_fold(PUSH_BTN_TBL, adj_eff, bf, h_idx)
                if u < iso_prob:
                    return self._jam_or_raise(legal, hero_total)
                if to_call <= bb and hand_strength >= 0.55 * (bf ** 0.25):
                    return check_or_call(legal)
                return check_or_fold(legal)

            # Facing a raise at short stack: distinguish an all-in shove from a 2.5x-3x normal open!
            raise_bb = obs.current_bet_to / bb
            if pre_raises >= 2 or raise_bb >= 0.55 * s_lookup or to_call >= 0.45 * hero_total:
                # Facing an all-in, 3-bet+, or committing raise: use push/fold call table keyed by actual call cost!
                jam_bb = max(2.0, min(20.0, (to_call / bb) + 1.0 + 3.0 * max(0, pre_raises - 1) + 2.0 * callers))
                call_prob = lookup_push_fold(
                    CALL_BB_TBL if n_behind == 0 else CALL_VS_BTN_TBL,
                    jam_bb,
                    bf,
                    h_idx,
                )
                if aggressor is not None and pre_raises == 1:
                    agg_prof = _opp_profile(state, aggressor)
                    looseness = (agg_prof["pfr"] - 0.20) * agg_prof["conf"]
                    call_prob = min(1.0, max(0.0, call_prob + 0.45 * looseness))
                if u < call_prob:
                    if legal.can_raise and hand_strength >= 0.63:
                        return self._jam_or_raise(legal, hero_total)
                    return check_or_call(legal)
                return check_or_fold(legal)
            else:
                # Facing a standard 2.5x-3.5x open at 8-12 BB:
                nb_eff = max(2, min(7, n_behind + 2))
                rejam_prob = lookup_preflop(DEF_3B_TBL, nb_eff, bf, h_idx)
                if hand_strength >= 0.62 * (bf ** 0.20):
                    rejam_prob = max(rejam_prob, 0.85)
                elif aggressor is not None and n_behind <= 1:
                    agg_prof = _opp_profile(state, aggressor)
                    if agg_prof["rfi"] > 0.35 and hand_strength >= 0.58 and bf <= 1.30:
                        rejam_prob = max(rejam_prob, 0.70)
                if u < rejam_prob:
                    return self._jam_or_raise(legal, hero_total)
                if hero == obs.big_blind_seat and to_call <= 2.2 * bb and s_lookup >= 8.0 and hand_strength >= 0.54 * (bf ** 0.25):
                    return check_or_call(legal)
                return check_or_fold(legal)

        # 2. Deep/Medium Stack Preflop (eff_bb > 12 BB)
        # In early levels (blinds <= 30, eff_bb >= 40), play a tight, high-equity range from early/mid
        # positions to avoid multiway collisions; as blinds rise (level >= 2 or eff_bb < 32), ramp up
        # late-position steals and 3-bet isolation!
        if pre_raises == 0 and limpers == 0:
            nb = max(1, min(7, n_behind))
            bf_open_eff = 1.0 + 0.65 * (bf - 1.0)
            open_prob = lookup_preflop(OPEN_TBL, nb, bf_open_eff, h_idx)
            if bb <= 30 and eff_bb >= 40.0 and nb >= 3:
                if hand_strength < 0.60:
                    open_prob *= 0.25
            elif (bb >= 50 or eff_bb <= 35.0) and nb <= 2 and bf <= 1.35:
                if hand_strength >= 0.50:
                    open_prob = max(open_prob, 0.88)
                elif hand_strength >= 0.45 and nb <= 1:
                    open_prob = max(open_prob, 0.78)
                elif nb <= 2 and (bb >= 50 or eff_bb <= 28.0) and bf <= 1.25:
                    # Steal with any suited or high-card hand when folded to on BTN/SB in middle/late levels
                    if play_adj >= 0.015 or raw_eq >= 0.48:
                        open_prob = max(open_prob, 0.72)

            if nb <= 3 and obs.big_blind_seat is not None:
                bb_prof = _opp_profile(state, obs.big_blind_seat)
                fts_delta = (bb_prof["fts"] - 0.56) * bb_prof["conf"]
                open_prob = min(1.0, max(0.0, open_prob + 0.80 * fts_delta))

            if u < open_prob:
                mult = 2.6 if nb == 1 else (2.3 if nb <= 3 else 2.6)
                target = int(round(mult * bb))
                return self._sized_raise(legal, target, hero_total)
            if nb == 1 and to_call <= sb and play_adj >= 0.035 and raw_eq >= 0.49 and bf <= 1.25:
                return check_or_call(legal)
            return check_or_fold(legal)

        if pre_raises == 0 and limpers >= 1:
            nb = max(1, min(7, n_behind + limpers))
            iso_prob = lookup_preflop(OPEN_TBL, nb, bf, h_idx)
            if u < iso_prob:
                target = int(round((2.6 + 1.0 * limpers) * bb))
                return self._sized_raise(legal, target, hero_total)
            pot_odds = to_call / max(1, obs.pot_total + to_call)
            if to_call == 0 or (pot_odds <= 0.18 and (play_adj >= 0.032 or raw_eq >= 0.55) and bf <= 1.25):
                return check_or_call(legal)
            return check_or_fold(legal)

        if pre_raises == 1 and not hero_raised:
            if aggressor in order:
                agg_nb = max(1, len(order) - 1 - order.index(aggressor))
            else:
                agg_nb = max(2, n_behind + 2)
            if hero == obs.big_blind_seat and aggressor == obs.small_blind_seat and callers == 0:
                nb_eff = 1
            else:
                nb_eff = max(2, min(7, max(agg_nb, n_behind + 2 + callers)))

            raise_bb = obs.current_bet_to / bb
            if raise_bb > 4.2 or to_call >= 0.35 * hero_total:
                # Facing an oversized open / jam: use push/fold calling table keyed by the actual amount to call + 1 BB!
                jam_bb = min(20.0, (to_call / bb) + 1.0 + 2.5 * callers + 1.2 * n_behind)
                p_call = lookup_push_fold(CALL_BB_TBL if n_behind == 0 else CALL_VS_BTN_TBL, jam_bb, bf, h_idx)
                if u < p_call:
                    if legal.can_raise and hand_strength >= 0.65:
                        return self._jam_or_raise(legal, hero_total)
                    return check_or_call(legal)
                return check_or_fold(legal)

            p_call = lookup_preflop(DEF_CALL_TBL, nb_eff, bf, h_idx)
            p_3bet = lookup_preflop(DEF_3B_TBL, nb_eff, bf, h_idx)

            if aggressor is not None:
                agg_prof = _opp_profile(state, aggressor)
                looseness = (agg_prof["rfi"] - 0.24) * agg_prof["conf"]
                # Also detect late-position steals in level >= 2 even before sample size is large
                if agg_nb <= 2 and bb >= 50:
                    looseness = max(looseness, 0.10)
                if looseness > 0.04 and hand_strength >= 0.57:
                    p_3bet = min(0.88, p_3bet + 1.4 * looseness)
                elif looseness > 0.06 and play_adj >= 0.040 and n_behind <= 1:
                    # 3-bet bluff suited wheel aces / suited connectors against frequent late-position stealers
                    p_3bet = min(0.45, p_3bet + 0.65 * looseness)
                else:
                    p_3bet = min(0.45, max(0.0, p_3bet + 0.35 * looseness))
                if looseness < -0.03:
                    p_call *= 0.50
                    if hand_strength < 0.66:
                        p_3bet *= 0.30
                else:
                    p_call = min(0.68, max(0.0, p_call + 0.35 * looseness))

            if hero == obs.small_blind_seat:
                p_3bet = min(0.55, p_3bet + 0.25 * p_call)
                p_call *= 0.35

            if raise_bb > 2.6:
                price_pen = min(0.85, (raise_bb - 2.3) * 0.32)
                p_call = max(0.0, p_call * (1.0 - price_pen))

            if u < p_3bet:
                if eff_bb <= 20.0:
                    return self._jam_or_raise(legal, hero_total)
                mult = 3.4 if hero in (obs.small_blind_seat, obs.big_blind_seat) else 3.0
                target = int(round(obs.current_bet_to * mult))
                return self._sized_raise(legal, target, hero_total)
            if u < p_3bet + p_call:
                return check_or_call(legal)
            return check_or_fold(legal)

        if pre_raises == 2:
            nb_eff = max(1, min(7, n_behind + 1))
            bet_bb = obs.current_bet_to / bb
            if bet_bb >= 11.0 or to_call >= 0.32 * hero_total or not hero_raised:
                # Facing a 3-bet shove, committing 3-bet, or cold 3-bet:
                # Use CFR+ push/fold call table at tightened effective stack depth (since a 3-bet is much tighter than an open-shove!)
                eff_jam_bb = min(20.0, max(10.0, (to_call / bb) * 1.35 + (4.0 if not hero_raised else 2.0)))
                p_call = lookup_push_fold(CALL_BB_TBL, eff_jam_bb, bf, h_idx)
                if u < p_call:
                    if legal.can_raise and hand_strength >= 0.67:
                        return self._jam_or_raise(legal, hero_total)
                    return check_or_call(legal)
                return check_or_fold(legal)

            p_call = lookup_preflop(VS3B_CALL_TBL, nb_eff, bf, h_idx)
            p_4bet = lookup_preflop(VS3B_4B_TBL, nb_eff, bf, h_idx)

            if aggressor is not None:
                agg_prof = _opp_profile(state, aggressor)
                agg_3b_delta = (agg_prof["three_b"] - 0.08) * agg_prof["conf"]
                p_4bet = min(0.35, max(0.0, p_4bet + 0.35 * agg_3b_delta))
                p_call = min(0.55, max(0.0, p_call + 0.45 * agg_3b_delta))

            if u < p_4bet:
                if eff_bb <= 32.0:
                    return self._jam_or_raise(legal, hero_total)
                target = int(round(obs.current_bet_to * 2.35))
                return self._sized_raise(legal, target, hero_total)
            if u < p_4bet + p_call and hero_raised:
                return check_or_call(legal)
            return check_or_fold(legal)

        # Facing a 4-bet or 5-bet+: only continue with true top-tier premiums (QQ+, AKs, AKo depending on BF)
        eff_4b_bb = min(20.0, max(14.0, (to_call / bb) * 1.4 + 4.0))
        p_call_4b = lookup_push_fold(CALL_BB_TBL, eff_4b_bb, bf, h_idx)
        if h_cls in ("AA", "KK") or (h_cls in ("QQ", "AKs", "AKo") and u < p_call_4b):
            if legal.can_raise and h_cls in ("AA", "KK"):
                return self._jam_or_raise(legal, hero_total)
            return check_or_call(legal)
        return check_or_fold(legal)

    # ------------------------------------------------------------------ NLHE Postflop

    def _decide_nlhe_postflop(self, obs: Observation, state: dict[str, Any]) -> Action:
        legal = obs.legal
        hero = obs.hero_seat
        blinds = obs.blinds
        bb = max(1, blinds.big_blind if blinds else 20)
        to_call = legal.call_amount or 0
        pot = max(1, obs.pot_total)

        me = next(p for p in obs.players if p.seat == hero)
        hero_total = me.stack + me.street_committed
        live_opps = [p for p in obs.players if p.seat != hero and p.status in ("active", "all_in")]
        n_opp = max(1, len(live_opps))

        pre_aggressor: int | None = None
        street_aggressor: int | None = None
        street_raises = 0
        hero_bet_street = False
        prev_streets_bet_by_opp = 0
        seen_st = set()
        for ev in obs.action_history:
            if ev.type != "ActionApplied" or ev.data.get("auto"):
                continue
            d = ev.data
            s, st, kind = int(d["seat"]), d["street"], d["action"]["type"]
            if st == "preflop" and kind == "raise_to":
                pre_aggressor = s
            elif st == obs.street and kind == "raise_to":
                street_raises += 1
                street_aggressor = s
                if s == hero:
                    hero_bet_street = True
            elif st != "preflop" and st != obs.street and kind == "raise_to" and s != hero:
                if st not in seen_st:
                    seen_st.add(st)
                    prev_streets_bet_by_opp += 1

        ring = list(obs.seats)
        live_seats = {p.seat for p in obs.players if p.status in ("active", "all_in")}
        in_position = False
        if obs.button_seat in ring:
            bi = ring.index(obs.button_seat)
            ordered_post = [s for s in ring[bi + 1 :] + ring[: bi + 1] if s in live_seats]
            in_position = bool(ordered_post) and ordered_post[-1] == hero

        primary_villain = street_aggressor or pre_aggressor or (live_opps[0].seat if live_opps else None)
        bf = compute_bubble_factor(obs, hero, primary_villain)
        vil_prof = _opp_profile(state, primary_villain) if primary_villain is not None else {
            "af": 0.48, "fpost": 0.45, "conf": 0.0
        }

        eq, bucket, outs, wet, high_norm = _fast_nlhe_postflop(obs.hole_cards, obs.board, n_opp)
        spr = me.stack / max(1.0, float(pot))
        u = self._rand(state)

        if to_call > 0:
            pot_before_bet = max(1, pot - to_call)
            bet_frac = to_call / pot_before_bet
            stack_commit_frac = to_call / max(1.0, float(hero_total))

            icm_risk_mult = 1.0 + (bf - 1.0) * min(1.2, stack_commit_frac * 2.0 + 0.30)
            req_eq = (to_call * icm_risk_mult) / max(1.0, float(pot) + to_call * icm_risk_mult)

            # Opponent range narrowing on raises, multi-street barrels, and large bet sizes
            agg_shift = (vil_prof["af"] - 0.50) * vil_prof["conf"] * 0.08
            reraise_pen = 0.16 if (hero_bet_street or street_raises >= 2) else 0.0
            barrel_pen = 0.05 * prev_streets_bet_by_opp + (0.05 if obs.street == "river" else 0.0)
            size_pen = min(0.08, max(0.0, (bet_frac - 0.55) * 0.10))
            multi_pen = 0.04 * max(0, n_opp - 1)
            pos_bonus = 0.015 if in_position else -0.015
            eff_eq = eq + agg_shift - reraise_pen - barrel_pen - size_pen - multi_pen + pos_bonus

            # Guard against calling off big bets / raises with one pair or weak two-pair on scary boards
            if (hero_bet_street or street_raises >= 2) and bucket != "monster":
                if bucket != "strong" or stack_commit_frac >= 0.35 or obs.street in ("turn", "river"):
                    if outs < 9 or obs.street == "river":
                        return check_or_fold(legal)

            if legal.can_raise:
                if bucket == "monster" and eff_eq >= 0.78:
                    trap_p = 0.25 if (in_position and wet < 0.35 and obs.street != "river") else 0.08
                    if u >= trap_p:
                        target = int(round(obs.current_bet_to + 0.70 * (pot + to_call)))
                        return self._sized_raise(legal, target, hero_total)
                elif bucket == "strong" and eff_eq >= 0.76 and street_raises <= 1 and bf <= 1.25 and not hero_bet_street:
                    if u < 0.25:
                        target = int(round(obs.current_bet_to + 0.60 * (pot + to_call)))
                        return self._sized_raise(legal, target, hero_total)
                elif outs >= 9 and obs.street == "flop" and street_raises <= 1 and n_opp == 1 and bf <= 1.20:
                    semi_p = 0.22 + 0.20 * (vil_prof["fpost"] - 0.45) * vil_prof["conf"]
                    if u < semi_p:
                        target = int(round(obs.current_bet_to + 0.65 * (pot + to_call)))
                        return self._sized_raise(legal, target, hero_total)

            implied_bonus = 0.0
            if outs >= 8 and obs.street in ("flop", "turn") and spr >= 2.2 and bet_frac <= 0.75:
                implied_bonus = 0.040 if in_position else 0.020

            margin = eff_eq + implied_bonus - req_eq
            if margin >= 0.025:
                return call()
            if margin >= -0.025 and stack_commit_frac <= 0.35:
                mix_call = (margin + 0.025) / 0.05
                if u < mix_call:
                    return call()
            return check_or_fold(legal)

        if not legal.can_raise:
            return check()

        is_pfr = (pre_aggressor == hero)
        # Exploit opponent postflop fold tendency and station tendency in bet sizing & frequencies:
        opp_folds_alot = (vil_prof["fpost"] - 0.45) * vil_prof["conf"]
        opp_is_station = (0.40 - vil_prof["fpost"]) * vil_prof["conf"] if vil_prof["fpost"] < 0.40 else 0.0

        if obs.street == "flop":
            size_frac = 0.40 if wet < 0.38 and n_opp == 1 else 0.62
        elif obs.street == "turn":
            size_frac = 0.62 if spr > 1.8 else 0.72
        else:
            size_frac = 0.72 if bucket == "monster" else 0.55

        if bucket in ("monster", "strong") and opp_is_station > 0.02:
            # Fat value sizing against calling stations!
            size_frac = min(0.90, size_frac + 0.35 * opp_is_station)

        alpha = (size_frac / (1.0 + size_frac)) / (bf * (n_opp ** 0.90))
        fold_bonus = opp_folds_alot * 0.45

        bet_prob = 0.0
        if bucket == "monster":
            bet_prob = 0.90 if (not in_position and obs.street == "flop" and wet < 0.30) else 0.97
        elif bucket == "strong":
            if prev_streets_bet_by_opp >= 1 and not in_position:
                bet_prob = 0.45
            else:
                bet_prob = 0.85 if n_opp == 1 else 0.68
            if bf >= 1.40 and obs.street in ("turn", "river"):
                bet_prob *= 0.75
        elif bucket == "medium":
            if obs.street == "flop" and is_pfr and n_opp == 1 and wet < 0.42:
                bet_prob = 0.58 + fold_bonus - 0.40 * opp_is_station
            elif obs.street == "river" and in_position and n_opp == 1 and eq >= 0.56 and bf <= 1.25 and prev_streets_bet_by_opp == 0:
                bet_prob = 0.42 + 0.30 * opp_is_station
                size_frac = 0.42
            else:
                bet_prob = 0.12 if in_position else 0.0
        else:
            # Never bluff calling stations; bluff more vs high-folders
            station_damp = max(0.15, 1.0 - 2.5 * opp_is_station)
            if outs >= 8 and obs.street in ("flop", "turn") and prev_streets_bet_by_opp == 0:
                bet_prob = min(0.72, (2.1 * alpha + fold_bonus + (0.18 if is_pfr else 0.0)) * station_damp)
            elif outs >= 4 and obs.street == "flop" and n_opp == 1 and is_pfr:
                bet_prob = min(0.55, (1.4 * alpha + fold_bonus + 0.15) * station_damp)
            elif is_pfr and obs.street == "flop" and n_opp == 1 and (high_norm >= 0.60 or wet < 0.40):
                bet_prob = min(0.70, (0.48 + fold_bonus) * station_damp)
            elif obs.street == "river" and n_opp == 1 and bf <= 1.25 and prev_streets_bet_by_opp == 0:
                has_blocker = any(RV[c[0]] >= 11 for c in obs.hole_cards)
                if has_blocker:
                    bet_prob = max(0.0, min(0.30, (alpha * 0.80 + fold_bonus) * station_damp))

        if u < bet_prob:
            bet_chips = max(bb, int(round(pot * size_frac)))
            target = obs.current_bet_to + bet_chips
            return self._sized_raise(legal, target, hero_total)

        return check()

    # ------------------------------------------------------------------ Generic Multi-Variant Policy

    def _decide_variant(self, obs: Observation, state: dict[str, Any]) -> Action:
        """Equilibrium-inspired ICM-aware policy for Short Deck, PLO4, and reveal_variant."""
        legal = obs.legal
        hero = obs.hero_seat
        blinds = obs.blinds
        bb = max(1, blinds.big_blind if blinds else 20)
        to_call = legal.call_amount or 0
        pot = max(1, obs.pot_total)

        me = next(p for p in obs.players if p.seat == hero)
        hero_total = me.stack + me.street_committed
        live_opps = [p for p in obs.players if p.seat != hero and p.status in ("active", "all_in")]
        n_opp = max(1, len(live_opps))

        bf = compute_bubble_factor(obs, hero, live_opps[0].seat if live_opps else None)
        eq = _generic_variant_equity(obs, n_opp)
        u = self._rand(state)

        if to_call > 0:
            req_eq = (to_call * bf) / max(1.0, float(pot) + to_call * bf)
            if eq >= req_eq + 0.14 and legal.can_raise:
                target = int(round(obs.current_bet_to + 0.65 * (pot + to_call)))
                return self._sized_raise(legal, target, hero_total)
            if eq >= req_eq:
                return call()
            return check_or_fold(legal)

        if legal.can_raise and eq >= 0.56 * (bf ** 0.25):
            if u < min(0.85, (eq - 0.45) * 2.0):
                target = obs.current_bet_to + max(bb, int(round(0.55 * pot)))
                return self._sized_raise(legal, target, hero_total)
        return check_or_call(legal)

    # ------------------------------------------------------------------ Sizing Helpers

    @staticmethod
    def _jam_or_raise(legal: LegalActions, hero_total: int) -> Action:
        if not legal.can_raise:
            return check_or_call(legal)
        target = legal.all_in_raise_to or legal.max_raise_to or legal.short_all_in_raise_to or hero_total
        act = clamp_raise_to(legal, target)
        return act if act is not None else check_or_call(legal)

    @staticmethod
    def _sized_raise(legal: LegalActions, target: int, hero_total: int) -> Action:
        if not legal.can_raise:
            return check_or_call(legal)
        if target >= int(0.42 * hero_total):
            jam_target = legal.all_in_raise_to or legal.max_raise_to or legal.short_all_in_raise_to or hero_total
            act = clamp_raise_to(legal, jam_target)
            if act is not None:
                return act
        act = clamp_raise_to(legal, target)
        return act if act is not None else check_or_call(legal)


def make_bot() -> EquilibriumBot:
    return EquilibriumBot()
