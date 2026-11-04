"""Precompute 169x169 NLHE equity matrix and solve CFR+ equilibrium push/fold and preflop tables.

Uses only stdlib, numpy, and arena handeval/charts.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np

from arena.agents.styles.charts import ALL_CLASSES

# Order of 169 canonical hand classes: ALL_CLASSES is row-major 13x13 grid (AA, AKs, ..., 22)
CLASSES = list(ALL_CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
assert len(CLASSES) == 169

# Number of combos for each of the 169 classes (6 for pairs, 4 for suited, 12 for offsuit)
COMBO_WEIGHTS = np.array([
    6.0 if len(c) == 2 else (4.0 if c[2] == "s" else 12.0)
    for c in CLASSES
], dtype=np.float64)


def fast_eval_7(r0: int, s0: int, r1: int, s1: int,
                br: tuple[int, int, int, int, int],
                bs: tuple[int, int, int, int, int]) -> int:
    """Fast 7-card Hold'em evaluator returning a single integer score."""
    ranks = (r0, r1, br[0], br[1], br[2], br[3], br[4])
    suits = (s0, s1, bs[0], bs[1], bs[2], bs[3], bs[4])
    rc = [0] * 13
    sc = [0] * 4
    smask = [0] * 4
    rmask = 0
    for r, s in zip(ranks, suits):
        rc[r] += 1
        sc[s] += 1
        smask[s] |= (1 << r)
        rmask |= (1 << r)

    flush_suit = -1
    for s in range(4):
        if sc[s] >= 5:
            flush_suit = s
            break

    if flush_suit >= 0:
        fmask = smask[flush_suit]
        for hi in range(12, 3, -1):
            st = 0x1F << (hi - 4)
            if (fmask & st) == st:
                return (8 << 20) | hi
        if (fmask & 0x100F) == 0x100F:
            return (8 << 20) | 3
        top5 = 0
        cnt = 0
        for r in range(12, -1, -1):
            if (fmask >> r) & 1:
                top5 = (top5 << 4) | r
                cnt += 1
                if cnt == 5:
                    break
        flush_val = (5 << 20) | top5
    else:
        flush_val = 0

    quads = -1
    trips1 = -1
    trips2 = -1
    pair1 = -1
    pair2 = -1
    for r in range(12, -1, -1):
        c = rc[r]
        if c == 4:
            quads = r
        elif c == 3:
            if trips1 < 0:
                trips1 = r
            else:
                trips2 = r
        elif c == 2:
            if pair1 < 0:
                pair1 = r
            elif pair2 < 0:
                pair2 = r

    if quads >= 0:
        kicker = max(r for r in range(12, -1, -1) if r != quads and rc[r] > 0)
        return (7 << 20) | (quads << 4) | kicker

    if trips1 >= 0 and (trips2 >= 0 or pair1 >= 0):
        fh_pair = max(trips2, pair1)
        return (6 << 20) | (trips1 << 4) | fh_pair

    if flush_val:
        return flush_val

    for hi in range(12, 3, -1):
        st = 0x1F << (hi - 4)
        if (rmask & st) == st:
            return (4 << 20) | hi
    if (rmask & 0x100F) == 0x100F:
        return (4 << 20) | 3

    if trips1 >= 0:
        k = 0
        cnt = 0
        for r in range(12, -1, -1):
            if r != trips1 and rc[r] > 0:
                k = (k << 4) | r
                cnt += 1
                if cnt == 2:
                    break
        return (3 << 20) | (trips1 << 8) | k

    if pair1 >= 0 and pair2 >= 0:
        kicker = max(r for r in range(12, -1, -1) if r != pair1 and r != pair2 and rc[r] > 0)
        return (2 << 20) | (pair1 << 8) | (pair2 << 4) | kicker

    if pair1 >= 0:
        k = 0
        cnt = 0
        for r in range(12, -1, -1):
            if r != pair1 and rc[r] > 0:
                k = (k << 4) | r
                cnt += 1
                if cnt == 3:
                    break
        return (1 << 20) | (pair1 << 12) | k

    top5 = 0
    cnt = 0
    for r in range(12, -1, -1):
        if rc[r] > 0:
            top5 = (top5 << 4) | r
            cnt += 1
            if cnt == 5:
                break
    return top5


def compute_equity_matrix(n_boards: int = 2000, seed: int = 20261104) -> tuple[np.ndarray, np.ndarray]:
    """Compute 169x169 preflop all-in equity matrix and card-removal weight matrix."""
    t0 = time.perf_counter()
    combos = []
    combo_class = []
    c0_arr = []
    c1_arr = []
    ranks_str = "23456789TJQKA"
    for c0 in range(52):
        for c1 in range(c0 + 1, 52):
            r0, s0 = c0 // 4, c0 % 4
            r1, s1 = c1 // 4, c1 % 4
            hi, lo = (r0, r1) if r0 >= r1 else (r1, r0)
            if hi == lo:
                cls_name = ranks_str[hi] + ranks_str[lo]
            else:
                cls_name = ranks_str[hi] + ranks_str[lo] + ("s" if s0 == s1 else "o")
            combos.append((c0, c1, r0, s0, r1, s1))
            combo_class.append(CLASS_TO_IDX[cls_name])
            c0_arr.append(c0)
            c1_arr.append(c1)

    c0_np = np.array(c0_arr, dtype=np.int32)
    c1_np = np.array(c1_arr, dtype=np.int32)

    disjoint = (
        (c0_np[:, None] != c0_np[None, :])
        & (c0_np[:, None] != c1_np[None, :])
        & (c1_np[:, None] != c0_np[None, :])
        & (c1_np[:, None] != c1_np[None, :])
    )

    M = np.zeros((169, 1326), dtype=np.float64)
    for idx, cls_idx in enumerate(combo_class):
        M[cls_idx, idx] = 1.0

    valid_pairs = M @ disjoint.astype(np.float64) @ M.T  # (169, 169)

    win_sum = np.zeros((169, 169), dtype=np.float64)
    pair_count = np.zeros((169, 169), dtype=np.float64)

    rng = random.Random(seed)
    deck_indices = list(range(52))

    for _ in range(n_boards):
        board = rng.sample(deck_indices, 5)
        b_set = set(board)
        br = tuple(c // 4 for c in board)
        bs = tuple(c % 4 for c in board)

        scores = np.zeros(1326, dtype=np.int32)
        valid_combo = np.ones(1326, dtype=bool)
        for idx, (c0, c1, r0, s0, r1, s1) in enumerate(combos):
            if c0 in b_set or c1 in b_set:
                valid_combo[idx] = False
            else:
                scores[idx] = fast_eval_7(r0, s0, r1, s1, br, bs)

        active = np.where(valid_combo)[0]
        sc_a = scores[active]
        dis_a = disjoint[np.ix_(active, active)]
        M_a = M[:, active]

        gt = (sc_a[:, None] > sc_a[None, :]) & dis_a
        eq = (sc_a[:, None] == sc_a[None, :]) & dis_a

        pts = gt.astype(np.float64) + 0.5 * eq.astype(np.float64)
        win_sum += M_a @ pts @ M_a.T
        pair_count += M_a @ dis_a.astype(np.float64) @ M_a.T

    equity = win_sum / np.maximum(pair_count, 1.0)
    equity = 0.5 * (equity + (1.0 - equity.T))
    np.fill_diagonal(equity, 0.5)

    elapsed = time.perf_counter() - t0
    print(f"Computed 169x169 equity matrix across {n_boards} boards in {elapsed:.2f}s")
    return equity, valid_pairs


def solve_push_fold_cfr(
    equity: np.ndarray,
    valid_pairs: np.ndarray,
    eff_bb_grid: list[float],
    bf_grid: list[float],
    sb_posted: float = 0.5,
    bb_posted: float = 1.0,
    dead_pot: float = 0.0,
    iterations: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve heads-up push/fold game for each (eff_bb, bubble_factor) via CFR+."""
    n_s = len(eff_bb_grid)
    n_bf = len(bf_grid)
    push_table = np.zeros((n_s, n_bf, 169), dtype=np.float32)
    call_table = np.zeros((n_s, n_bf, 169), dtype=np.float32)

    W = valid_pairs / np.sum(valid_pairs)
    P_cond = W / np.sum(W, axis=1, keepdims=True)

    for si, S in enumerate(eff_bb_grid):
        risk_p = max(0.1, S - sb_posted)
        fold_gain_p = bb_posted + dead_pot
        win_gain_p = S - sb_posted + dead_pot

        risk_d = max(0.1, S - bb_posted)
        win_gain_d = S + sb_posted + dead_pot

        for bi, bf in enumerate(bf_grid):
            # Pusher faces full BF, while Defender uses dampened BF in equilibrium solve so Pusher
            # tightens monotonically as BF increases (preventing suicidal ICM shoves against normal callers).
            bf_p = 1.0 + 1.25 * (bf - 1.0)
            bf_d = 1.0 + 0.35 * (bf - 1.0)
            u_p_call = equity * win_gain_p - (1.0 - equity) * risk_p * bf_p
            u_d_call = (1.0 - equity.T) * win_gain_d - equity.T * risk_d * bf * 0.90
            u_d_call_for_p = (1.0 - equity.T) * win_gain_d - equity.T * risk_d * bf_d

            reg_p = np.zeros((169, 2), dtype=np.float64)
            reg_d = np.zeros((169, 2), dtype=np.float64)
            sum_p = np.zeros(169, dtype=np.float64)
            sum_d = np.zeros(169, dtype=np.float64)
            w_sum = 0.0

            strat_p = np.full(169, 0.5, dtype=np.float64)
            strat_d = np.full(169, 0.3, dtype=np.float64)

            for t in range(1, iterations + 1):
                ev_push_ij = (1.0 - strat_d[None, :]) * fold_gain_p + strat_d[None, :] * u_p_call
                ev_push = np.sum(P_cond * ev_push_ij, axis=1)
                ev_p = strat_p * ev_push

                reg_p[:, 0] = np.maximum(0.0, reg_p[:, 0] + (0.0 - ev_p))
                reg_p[:, 1] = np.maximum(0.0, reg_p[:, 1] + (ev_push - ev_p))
                denom_p = np.maximum(reg_p[:, 0] + reg_p[:, 1], 1e-12)
                strat_p = np.where(reg_p[:, 0] + reg_p[:, 1] > 1e-12, reg_p[:, 1] / denom_p, 0.5)

                reach_p = P_cond * strat_p[None, :]
                ev_call = np.sum(reach_p * u_d_call, axis=1)
                ev_d = strat_d * ev_call

                reg_d[:, 0] = np.maximum(0.0, reg_d[:, 0] + (0.0 - ev_d))
                reg_d[:, 1] = np.maximum(0.0, reg_d[:, 1] + (ev_call - ev_d))
                denom_d = np.maximum(reg_d[:, 0] + reg_d[:, 1], 1e-12)
                strat_d = np.where(reg_d[:, 0] + reg_d[:, 1] > 1e-12, reg_d[:, 1] / denom_d, 0.0)

                w = float(t)
                sum_p += w * strat_p
                sum_d += w * strat_d
                w_sum += w

            push_table[si, bi] = (sum_p / w_sum).astype(np.float32)
            call_table[si, bi] = (sum_d / w_sum).astype(np.float32)

    return push_table, call_table


def compute_playability() -> np.ndarray:
    """Compute postflop playability/realization adjustment for each of the 169 hand classes."""
    playability = np.zeros(169, dtype=np.float64)
    ranks_str = "23456789TJQKA"
    for idx, cls_name in enumerate(CLASSES):
        if len(cls_name) == 2:
            r = ranks_str.index(cls_name[0])
            # Pocket pairs have set-mining and overpair realization
            playability[idx] = 0.025 + 0.002 * r
        else:
            r_hi = ranks_str.index(cls_name[0])
            r_lo = ranks_str.index(cls_name[1])
            suited = cls_name[2] == "s"
            gap = r_hi - r_lo
            bonus = 0.032 if suited else -0.012
            if gap == 1:
                bonus += 0.022
            elif gap == 2:
                bonus += 0.014
            elif gap == 3:
                bonus += 0.006
            if r_hi == 12 and suited:
                bonus += 0.015
            elif r_hi >= 10 and r_lo >= 8:
                bonus += 0.010
            playability[idx] = bonus
    return playability


def solve_rfi_3bet_cfr(
    equity: np.ndarray,
    valid_pairs: np.ndarray,
    n_players_behind_list: list[int],
    bf_grid: list[float],
    iterations: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve preflop Open / 3-bet / Call-3bet / 4-bet game across positions and bubble factors.

    With `n_behind` players left to act:
    - Each individual player behind holds a random hand j (with card removal P_cond[i, j])
      and applies single-defender strategy `strat_def[j] = (p_fold, p_call, p_3bet)`.
    - For an opener with hand i, the probability a single random player behind folds is
      `f_1(i) = sum_j P_cond[i, j] * p_fold[j]`, calls is `c_1(i)`, and 3-bets is `b_1(i)`.
    - Across `k = n_behind` independent players behind:
      * Probability ALL k players fold: `P_all_fold(i) = f_1(i) ** k`
      * Probability at least one 3-bets: `P_any_3b(i) = 1.0 - (1.0 - b_1(i)) ** k`
      * Probability no one 3-bets and at least one calls: `P_call_only(i) = (1.0 - b_1(i)) ** k - f_1(i) ** k`
    - When a defender acts, they represent one of the `k` seats behind, accounting for the
      chance that remaining seats behind them might still wake up with a 3-bet or squeeze.
    """
    n_pos = len(n_players_behind_list)
    n_bf = len(bf_grid)

    open_table = np.zeros((n_pos, n_bf, 169), dtype=np.float32)
    def_call_table = np.zeros((n_pos, n_bf, 169), dtype=np.float32)
    def_3bet_table = np.zeros((n_pos, n_bf, 169), dtype=np.float32)
    vs3bet_call_table = np.zeros((n_pos, n_bf, 169), dtype=np.float32)
    vs3bet_4bet_table = np.zeros((n_pos, n_bf, 169), dtype=np.float32)

    W = valid_pairs / np.sum(valid_pairs)
    P_cond = W / np.sum(W, axis=1, keepdims=True)
    playability = compute_playability()
    raw_eq = (equity @ COMBO_WEIGHTS) / 1326.0
    hand_score = raw_eq + playability

    for pi, n_behind in enumerate(n_players_behind_list):
        k = float(n_behind)
        is_sb = (n_behind == 1)
        hero_blind = 0.5 if is_sb else 0.0
        open_sz = 2.5 if is_sb else (2.2 if n_behind <= 3 else 2.3)
        three_bet_sz = 8.0 if is_sb else 7.5
        four_bet_sz = 22.0

        # Positional realization shift:
        # In SB (n_behind=1), opener is OOP vs BB (-0.035).
        # On BTN (n_behind=2), opener is IP vs both blinds (+0.025).
        # Earlier positions are OOP vs field callers and face multiway/squeeze risk.
        pos_shift = -0.035 if is_sb else (0.025 - 0.018 * (n_behind - 2))

        for bi, bf in enumerate(bf_grid):
            # Postflop equity realization in single-raised pot (opener i vs caller j)
            e_srp = np.clip(
                0.5 + 1.30 * (equity - 0.5) + pos_shift + playability[:, None] - playability[None, :],
                0.04, 0.96
            )
            # Realized share in 3-bet pot
            e_3bp = np.clip(
                0.5 + 1.22 * (equity - 0.5) + 0.6 * pos_shift + 0.6 * (playability[:, None] - playability[None, :]),
                0.03, 0.97
            )
            e_4bp = equity

            dead_other = 1.0 if is_sb else 1.5
            pot_srp = 2.0 * open_sz + (0.0 if is_sb else 0.5)
            pot_3bp = 2.0 * three_bet_sz + (0.0 if is_sb else 0.5)
            pot_4bp = 2.0 * four_bet_sz + (0.0 if is_sb else 0.5)

            # Defender's risk aversion is dampened so Opener cannot abuse ICM to open trash:
            # Instead, Opener faces full BF on risk_o_srp, while Defender's defense threshold uses bf_def = 1 + 0.25*(bf - 1).
            bf_def = 1.0 + 0.25 * (bf - 1.0)
            bf_open = 1.0 + 1.35 * (bf - 1.0)

            risk_o_srp = open_sz - hero_blind
            win_o_srp = pot_srp - risk_o_srp
            u_o_dfold = dead_other
            u_o_dcall = e_srp * win_o_srp - (1.0 - e_srp) * risk_o_srp * bf_open - (1.0 - e_srp) * 0.70 * bf_open

            u_o_3bfold = -risk_o_srp * bf_open
            risk_o_3bp = three_bet_sz - hero_blind
            win_o_3bp = pot_3bp - risk_o_3bp
            u_o_3bcall = e_3bp * win_o_3bp - (1.0 - e_3bp) * risk_o_3bp * bf_open - (1.0 - e_3bp) * 0.90 * bf_open

            u_o_4bfold = three_bet_sz + (0.0 if is_sb else 0.5)
            risk_o_4bp = four_bet_sz - hero_blind
            win_o_4bp = pot_4bp - risk_o_4bp
            u_o_4bcall = e_4bp * win_o_4bp - (1.0 - e_4bp) * risk_o_4bp * bf_open

            # Single-defender vs Open:
            # Note: in CFR, Defender acts AFTER seeing an Open, so Defender's fold baseline is 0 (forfeiting posted blind).
            # In BB (is_sb=True), Defender already posted 1.0 BB and closes action (discount=0.80 BB).
            # In non-BB seats (n_behind >= 2), an individual seat behind has on average 0.35 BB posted,
            # and calling OOP or with players behind incurs postflop positional + squeeze cost.
            def_posted = 1.0 if is_sb else (0.45 if n_behind == 2 else 0.25)
            risk_d_srp = open_sz - def_posted
            win_d_srp = pot_srp - risk_d_srp
            implied_flat = 0.70 * np.maximum(0.0, playability)[:, None]
            flat_disadvantage = (0.48 if is_sb else (0.42 + 0.14 * (k - 1.0))) * bf_def
            u_d_call = (1.0 - e_srp.T + implied_flat) * win_d_srp - e_srp.T * risk_d_srp * bf_def - flat_disadvantage

            risk_d_3b = three_bet_sz - def_posted
            cold_behind_tax = 0.14 * max(0.0, k - 1.0) * bf_def
            u_d_3b_ofold = (open_sz + dead_other - def_posted) * (0.90 / (1.0 + 0.04 * (k - 1.0))) - cold_behind_tax
            win_d_3bp = pot_3bp - risk_d_3b
            postflop_3b_loss = np.maximum(0.0, 0.52 - (1.0 - e_3bp.T)) * 2.8 * bf_def
            u_d_3b_ocall = (1.0 - e_3bp.T) * win_d_3bp - e_3bp.T * risk_d_3b * bf_def - cold_behind_tax - postflop_3b_loss

            u_d_4b_dfold = -risk_d_3b * bf_def - cold_behind_tax
            risk_d_4bp = four_bet_sz - def_posted
            win_d_4bp = pot_4bp - risk_d_4bp
            u_d_4b_dcall = (1.0 - e_4bp.T) * win_d_4bp - e_4bp.T * risk_d_4bp * bf_def

            reg_open = np.zeros((169, 2), dtype=np.float64)
            reg_def = np.zeros((169, 3), dtype=np.float64)
            reg_vs3b = np.zeros((169, 3), dtype=np.float64)
            reg_vs4b = np.zeros((169, 2), dtype=np.float64)

            sum_open = np.zeros(169, dtype=np.float64)
            sum_def = np.zeros((169, 3), dtype=np.float64)
            sum_vs3b = np.zeros((169, 3), dtype=np.float64)
            w_sum = 0.0

            strat_open = np.clip((hand_score - 0.50) * 5.0, 0.05, 0.95)
            strat_def = np.zeros((169, 3), dtype=np.float64)
            strat_def[:, 0] = 0.75
            strat_def[:, 1] = 0.17
            strat_def[:, 2] = 0.08
            strat_vs3b = np.zeros((169, 3), dtype=np.float64)
            strat_vs3b[:, 0] = 0.55
            strat_vs3b[:, 1] = 0.30
            strat_vs3b[:, 2] = 0.15
            strat_vs4b = np.clip((hand_score - 0.62) * 6.0, 0.0, 1.0)

            for t in range(1, iterations + 1):
                # 1. Defender vs 4-bet
                reach_o_4b = P_cond * (strat_open * strat_vs3b[:, 2])[None, :]
                denom_o4b = np.maximum(np.sum(reach_o_4b, axis=1), 1e-9)
                ev_vs4b_fold = np.full(169, u_d_4b_dfold, dtype=np.float64)
                ev_vs4b_call = np.sum(reach_o_4b * u_d_4b_dcall, axis=1) / denom_o4b
                ev_vs4b = (1.0 - strat_vs4b) * ev_vs4b_fold + strat_vs4b * ev_vs4b_call
                reg_vs4b[:, 0] = np.maximum(0.0, reg_vs4b[:, 0] + (ev_vs4b_fold - ev_vs4b))
                reg_vs4b[:, 1] = np.maximum(0.0, reg_vs4b[:, 1] + (ev_vs4b_call - ev_vs4b))
                s_v4 = reg_vs4b[:, 0] + reg_vs4b[:, 1]
                strat_vs4b = np.divide(reg_vs4b[:, 1], s_v4, out=np.zeros(169), where=s_v4 > 1e-12)

                # 2. Opener vs 3-bet (given a defender 3-bet)
                reach_d_3b = P_cond * strat_def[None, :, 2]
                denom_d3b = np.maximum(np.sum(reach_d_3b, axis=1), 1e-9)
                u_o_4b_ij = (1.0 - strat_vs4b[None, :]) * u_o_4bfold + strat_vs4b[None, :] * u_o_4bcall
                ev_vs3b_0 = np.full(169, u_o_3bfold, dtype=np.float64)
                ev_vs3b_1 = np.sum(reach_d_3b * u_o_3bcall, axis=1) / denom_d3b
                ev_vs3b_2 = np.sum(reach_d_3b * u_o_4b_ij, axis=1) / denom_d3b
                ev_vs3b = (
                    strat_vs3b[:, 0] * ev_vs3b_0
                    + strat_vs3b[:, 1] * ev_vs3b_1
                    + strat_vs3b[:, 2] * ev_vs3b_2
                )
                reg_vs3b[:, 0] = np.maximum(0.0, reg_vs3b[:, 0] + (ev_vs3b_0 - ev_vs3b))
                reg_vs3b[:, 1] = np.maximum(0.0, reg_vs3b[:, 1] + (ev_vs3b_1 - ev_vs3b))
                reg_vs3b[:, 2] = np.maximum(0.0, reg_vs3b[:, 2] + (ev_vs3b_2 - ev_vs3b))
                s_v3 = np.sum(reg_vs3b, axis=1, keepdims=True)
                default_v3 = np.zeros_like(reg_vs3b)
                default_v3[:, 0] = 1.0
                strat_vs3b = np.divide(reg_vs3b, s_v3, out=default_v3, where=s_v3 > 1e-12)

                # 3. Opener initial open decision against k independent players behind
                f_1 = P_cond @ strat_def[:, 0]  # (169,) single-opponent fold prob
                c_1 = P_cond @ strat_def[:, 1]  # (169,) single-opponent call prob
                b_1 = P_cond @ strat_def[:, 2]  # (169,) single-opponent 3bet prob

                p_all_fold = f_1 ** k
                p_any_3b = 1.0 - ((1.0 - b_1) ** k)
                p_call_only = np.maximum(0.0, ((1.0 - b_1) ** k) - p_all_fold)

                # Conditional utility when called (given defender called)
                reach_d_call = P_cond * strat_def[None, :, 1]
                denom_dcall = np.maximum(np.sum(reach_d_call, axis=1), 1e-9)
                ev_when_called = np.sum(reach_d_call * u_o_dcall, axis=1) / denom_dcall

                # Conditional utility when 3-bet (given defender 3-bet)
                ev_when_3bet = ev_vs3b

                ev_open = (
                    p_all_fold * u_o_dfold
                    + p_call_only * ev_when_called
                    + p_any_3b * ev_when_3bet
                )
                ev_o = strat_open * ev_open
                reg_open[:, 0] = np.maximum(0.0, reg_open[:, 0] + (0.0 - ev_o))
                reg_open[:, 1] = np.maximum(0.0, reg_open[:, 1] + (ev_open - ev_o))
                s_o = reg_open[:, 0] + reg_open[:, 1]
                strat_open = np.divide(reg_open[:, 1], s_o, out=np.zeros(169), where=s_o > 1e-12)

                # 4. Defender response (given Opener open reach)
                reach_o = P_cond * strat_open[None, :]
                denom_o = np.maximum(np.sum(reach_o, axis=1), 1e-9)
                u_d_4b_ji = (1.0 - strat_vs4b[:, None]) * u_d_4b_dfold + strat_vs4b[:, None] * u_d_4b_dcall
                u_d_3b_ji = (
                    strat_vs3b[None, :, 0] * u_d_3b_ofold
                    + strat_vs3b[None, :, 1] * u_d_3b_ocall
                    + strat_vs3b[None, :, 2] * u_d_4b_ji
                )
                ev_d_0 = np.zeros(169, dtype=np.float64)
                ev_d_1 = np.sum(reach_o * u_d_call, axis=1) / denom_o
                ev_d_2 = np.sum(reach_o * u_d_3b_ji, axis=1) / denom_o
                ev_d = (
                    strat_def[:, 0] * ev_d_0
                    + strat_def[:, 1] * ev_d_1
                    + strat_def[:, 2] * ev_d_2
                )
                reg_def[:, 0] = np.maximum(0.0, reg_def[:, 0] + (ev_d_0 - ev_d))
                reg_def[:, 1] = np.maximum(0.0, reg_def[:, 1] + (ev_d_1 - ev_d))
                reg_def[:, 2] = np.maximum(0.0, reg_def[:, 2] + (ev_d_2 - ev_d))
                s_d = np.sum(reg_def, axis=1, keepdims=True)
                default_d = np.zeros_like(reg_def)
                default_d[:, 0] = 1.0
                strat_def = np.divide(reg_def, s_d, out=default_d, where=s_d > 1e-12)

                w = float(t)
                sum_open += w * strat_open
                sum_def += w * strat_def
                sum_vs3b += w * strat_vs3b
                w_sum += w

            open_table[pi, bi] = (sum_open / w_sum).astype(np.float32)
            def_call_table[pi, bi] = (sum_def[:, 1] / w_sum).astype(np.float32)
            def_3bet_table[pi, bi] = (sum_def[:, 2] / w_sum).astype(np.float32)
            vs3bet_call_table[pi, bi] = (sum_vs3b[:, 1] / w_sum).astype(np.float32)
            vs3bet_4bet_table[pi, bi] = (sum_vs3b[:, 2] / w_sum).astype(np.float32)

    return open_table, def_call_table, def_3bet_table, vs3bet_call_table, vs3bet_4bet_table


def main() -> None:
    equity, valid_pairs = compute_equity_matrix(n_boards=2000, seed=20261104)

    eff_bb_grid = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    bf_grid = [1.0, 1.15, 1.35, 1.65, 2.1]
    n_behind_grid = [1, 2, 3, 4, 5, 6, 7]

    t0 = time.perf_counter()
    push_sb, call_bb = solve_push_fold_cfr(
        equity, valid_pairs, eff_bb_grid, bf_grid, sb_posted=0.5, bb_posted=1.0, dead_pot=0.0, iterations=400
    )
    push_btn, call_vs_btn = solve_push_fold_cfr(
        equity, valid_pairs, eff_bb_grid, bf_grid, sb_posted=0.0, bb_posted=1.0, dead_pot=0.5, iterations=400
    )
    open_tbl, def_call_tbl, def_3b_tbl, vs3b_call_tbl, vs3b_4b_tbl = solve_rfi_3bet_cfr(
        equity, valid_pairs, n_behind_grid, bf_grid, iterations=400
    )
    print(f"Solved CFR+ push/fold and preflop tables in {time.perf_counter() - t0:.2f}s")

    w_norm = COMBO_WEIGHTS / 1326.0
    for s in [3.0, 6.0, 10.0, 15.0]:
        idx = eff_bb_grid.index(s)
        p_chip = float(push_sb[idx, 0] @ w_norm)
        c_chip = float(call_bb[idx, 0] @ w_norm)
        p_icm = float(push_sb[idx, 2] @ w_norm)
        c_icm = float(call_bb[idx, 2] @ w_norm)
        print(f"SB Push/Call @ {s:4.1f}bb: chip-EV push={p_chip:.1%} call={c_chip:.1%} | BF=1.35 push={p_icm:.1%} call={c_icm:.1%}")

    for pi, nb in enumerate(n_behind_grid):
        o_chip = float(open_tbl[pi, 0] @ w_norm)
        c_chip = float(def_call_tbl[pi, 0] @ w_norm)
        b3_chip = float(def_3b_tbl[pi, 0] @ w_norm)
        v3c = float(vs3b_call_tbl[pi, 0] @ w_norm)
        v34 = float(vs3b_4b_tbl[pi, 0] @ w_norm)
        o_icm = float(open_tbl[pi, 2] @ w_norm)
        print(f"n_behind={nb}: chip-EV open={o_chip:.1%} def_call={c_chip:.1%} def_3b={b3_chip:.1%} vs3b_call={v3c:.1%} vs3b_4b={v34:.1%} | BF=1.35 open={o_icm:.1%}")

    out_path = Path(__file__).resolve().parent / "tables.npz"
    raw_eq = (equity @ w_norm).astype(np.float32)
    playability = compute_playability().astype(np.float32)
    np.savez_compressed(
        out_path,
        equity=equity.astype(np.float32),
        raw_eq=raw_eq,
        playability=playability,
        eff_bb_grid=np.array(eff_bb_grid, dtype=np.float32),
        bf_grid=np.array(bf_grid, dtype=np.float32),
        n_behind_grid=np.array(n_behind_grid, dtype=np.int32),
        push_sb=push_sb,
        call_bb=call_bb,
        push_btn=push_btn,
        call_vs_btn=call_vs_btn,
        open_tbl=open_tbl,
        def_call_tbl=def_call_tbl,
        def_3b_tbl=def_3b_tbl,
        vs3b_call_tbl=vs3b_call_tbl,
        vs3b_4b_tbl=vs3b_4b_tbl,
    )
    print(f"Saved precomputed tables to {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
