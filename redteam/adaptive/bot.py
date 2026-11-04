"""Adaptive red-team bot: fast opponent modelling, image switching, ICM-aware endgame.

See README.md in this directory. Self-contained: standard library plus
``arena.agents.sdk`` only. All state is explicit JSON (well under 64 KiB).
"""

import hashlib

from arena.agents.sdk import ExplicitStateBot, call, check, clamp_raise_to, fold, with_note

RANKS = "23456789TJQKA"
RV = {c: i for i, c in enumerate(RANKS)}

# per-seat counter slots
HANDS, VPIP, PFR, PF_FACED, PF_FOLD, P_BET, P_CALL, P_FACED, P_FOLD, RFI_OPP, RFI = range(11)
NSTAT = 11
PRIORS = {"vpip": 0.32, "pfr": 0.13, "pf_fold": 0.55, "p_fold": 0.45, "agg": 0.9}
PRIOR_N = 12.0
TUNE = {"open": -1.5, "call": -1.5, "v3": -1.0, "pm": 0.1, "push": -1.0}


def chen(c1, c2):
    r1, r2 = RV[c1[0]], RV[c2[0]]
    hi, lo = max(r1, r2), min(r1, r2)
    pts = {12: 10.0, 11: 8.0, 10: 7.0, 9: 6.0}.get(hi, (hi + 2) / 2)
    if hi == lo:
        return max(pts * 2, 5.0)
    if c1[1] == c2[1]:
        pts += 2
    gap = hi - lo - 1
    pts -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
    if gap <= 1 and hi < 10:
        pts += 1
    return pts


def post_strength(hole, board):
    """Rough equity-like score in [0,1] from made hand plus draws (no sampling)."""
    cards = hole + board
    hr = [RV[c[0]] for c in hole]
    br = [RV[c[0]] for c in board]
    cnt = [0] * 13
    for c in cards:
        cnt[RV[c[0]]] += 1
    suits = {}
    for c in cards:
        suits[c[1]] = suits.get(c[1], 0) + 1
    hsuits = {c[1] for c in hole}
    flush = any(n >= 5 and s in hsuits for s, n in suits.items())
    rs = {RV[c[0]] for c in cards}
    if 12 in rs:
        rs.add(-1)
    straight = False
    oesd = gut = False
    hs = set(hr) | ({-1} if 12 in hr else set())
    for lo in range(-1, 9):
        win = [lo + i for i in range(5)]
        have = [x in rs for x in win]
        n = sum(have)
        if n == 5:
            straight = True
        elif n == 4 and any(x in hs for x in win if x in rs):
            gut = True
    for lo in range(0, 9):
        four = [lo + i for i in range(4)]
        if all(x in rs for x in four) and (lo - 1) in range(-1, 13) and (lo + 4) <= 12 and any(x in hs for x in four):
            oesd = True
    hole_in_straight = straight
    counts = sorted((n for n in cnt if n), reverse=True)
    bmax = max(br)
    street_left = 5 - len(board)
    # board-only strength is not ours
    bcnt = [0] * 13
    for r in br:
        bcnt[r] += 1
    quads = counts[0] == 4
    trips = counts[0] == 3
    pairs = sum(1 for n in cnt if n == 2)
    full = (trips and pairs >= 1) or (counts[0] == 3 and len(counts) > 1 and counts[1] >= 2) or counts[0] == 4
    if quads or full:
        own = any(cnt[r] > bcnt[r] for r in hr)
        return 0.97 if own else 0.55
    if flush:
        return 0.93
    if straight and hole_in_straight:
        brs = {r for r in br}
        # straight made purely by the board is weak
        board_only = any(all((lo + i) in (brs | ({-1} if 12 in brs else set())) for i in range(5)) for lo in range(-1, 9))
        return 0.5 if board_only else 0.9
    if trips:
        own = sum(1 for r in hr if cnt[r] == 3)
        if own == 2:
            return 0.9
        if own == 1:
            return 0.84 if (hr[0] != hr[1]) else 0.9
        return 0.45
    s = 0.0
    own_pairs = [r for r in set(hr) if cnt[r] >= 2]
    if pairs >= 2:
        pr = sorted((r for r in range(13) if cnt[r] == 2), reverse=True)
        mine = [r for r in pr if r in hr]
        if len(mine) >= 2:
            s = 0.8
        elif len(mine) == 1:
            s = 0.62 if mine[0] >= pr[0] or mine[0] >= bmax else 0.55
        else:
            s = 0.3
    elif pairs == 1:
        r = next(r for r in range(13) if cnt[r] == 2)
        if r in hr:
            if hr[0] == hr[1]:
                if r > bmax:
                    s = 0.74 + 0.1 * (r - bmax) / 12
                else:
                    s = 0.46 if sum(1 for b in br if b > r) == 1 else 0.38
            elif r == bmax:
                kick = max(h for h in hr if h != r)
                s = 0.6 + 0.12 * kick / 12
            elif r >= sorted(br)[len(br) // 2]:
                s = 0.45
            else:
                s = 0.33
        else:
            kick = max(hr)
            s = 0.2 + 0.1 * (kick / 12)
    else:
        hi = max(hr)
        s = 0.06 + 0.1 * (hi / 12) + 0.04 * sum(1 for h in hr if h > bmax)
    if street_left > 0:
        fd = any(n == 4 and su in hsuits for su, n in suits.items())
        if fd:
            s += 0.2 if street_left == 2 else 0.12
        if oesd:
            s += 0.17 if street_left == 2 else 0.1
        elif gut:
            s += 0.07 if street_left == 2 else 0.04
        elif street_left == 2 and 0 not in ():
            pass
    # paired / monotone boards cap one-pair holdings
    if s < 0.7:
        if max(bcnt) >= 2:
            s -= 0.04
        if max(suits.get(su, 0) for su in "cdhs") - 0 >= 4 and not flush and not any(
            n == 4 and su in hsuits for su, n in suits.items()
        ):
            s -= 0.05
    return max(0.02, min(0.99, s))


def icm_equity(stacks, hero, payouts):
    """Malmuth-Harville equity (in payout points) of seat index ``hero``."""
    n = len(stacks)
    k = min(len(payouts), n)
    total = float(sum(stacks))
    if total <= 0:
        return 0.0

    def rec(used, depth, tot, prob):
        if depth == k:
            return 0.0
        acc = 0.0
        pay = payouts[depth]
        for i in range(n):
            if used >> i & 1 or stacks[i] <= 0:
                continue
            p = prob * stacks[i] / tot
            if i == hero:
                acc += p * pay
                # hero already placed; remaining places pay hero nothing
            else:
                acc += rec(used | (1 << i), depth + 1, tot - stacks[i], p)
        return acc

    return rec(0, 0, total, 1.0)


class AdaptiveBot(ExplicitStateBot):
    name = "adaptive_redteam"
    rulesets = ("practice_nlhe",)

    def initial_state(self, table):
        return {
            "seed": table.policy_seed,
            "draws": 0,
            "last_seq": -1,
            "st": {},
            "phase0": int(hashlib.sha256(str(table.policy_seed).encode()).hexdigest()[:4], 16) % 2,
            "pf_agg": {},
            "hand": None,
        }

    def _rand(self, state):
        d = hashlib.sha256(f"{state['seed']}|{state['draws']}".encode()).digest()
        state["draws"] += 1
        return int.from_bytes(d[:8], "big") / 2**64

    # ------------------------------------------------------------ modelling
    def _update(self, obs, state):
        st = state["st"]
        last = state["last_seq"]
        hand_id = obs.hand_id
        new_hand = state["hand"] != hand_id
        if new_hand:
            state["hand"] = hand_id
        raises = 0
        voluntary = set()
        raiser = None
        callers = 0
        street = "preflop"
        bets = 0
        bettor = None
        acted_post = set()
        info = {"raises": 0, "raiser": None, "callers": 0, "post_bets": 0, "bettor": None, "pf_raiser": None}
        for ev in obs.action_history:
            t = ev.type
            new = ev.seq > last
            d = ev.data
            if t == "HandStarted":
                if new:
                    for s in (d.get("start_stacks") or {}):
                        if (d["start_stacks"][s] or 0) > 0:
                            self._seat(st, int(s))[HANDS] += 1
                    for s in d.get("positions", {}).get("order", []) if isinstance(d.get("positions"), dict) else []:
                        pass
            elif t == "BoardDealt":
                street = d.get("street", street)
                if street != "preflop":
                    bets = 0
                    bettor = None
                    acted_post = set()
            elif t == "ActionApplied" and not d.get("auto"):
                seat = d["seat"]
                typ = d["action"]["type"]
                row = self._seat(st, seat)
                if d["street"] == "preflop":
                    faced = raises > 0
                    if new:
                        if faced:
                            row[PF_FACED] += 1
                            if typ == "fold":
                                row[PF_FOLD] += 1
                        if raises == 0 and seat not in voluntary and typ in ("fold", "call", "raise_to"):
                            row[RFI_OPP] += 1
                            if typ == "raise_to":
                                row[RFI] += 1
                        if typ in ("call", "raise_to") and seat not in voluntary:
                            row[VPIP] += 1
                        if typ == "raise_to":
                            row[PFR] += 1
                    if typ in ("call", "raise_to"):
                        voluntary.add(seat)
                    if typ == "raise_to":
                        raises += 1
                        raiser = seat
                        callers = 0
                    elif typ == "call":
                        callers += 1
                else:
                    if new:
                        if typ == "raise_to":
                            row[P_BET] += 1
                        elif typ == "call":
                            row[P_CALL] += 1
                        if bets > 0 and seat != bettor:
                            row[P_FACED] += 1
                            if typ == "fold":
                                row[P_FOLD] += 1
                    if typ == "raise_to":
                        bets += 1
                        bettor = seat
        info["raises"] = raises
        info["raiser"] = raiser
        info["callers"] = callers
        info["post_bets"] = bets
        info["bettor"] = bettor
        if obs.action_history:
            state["last_seq"] = max(last, obs.action_history[-1].seq)
        return info

    @staticmethod
    def _seat(st, s):
        key = str(s)
        row = st.get(key)
        if row is None:
            row = st[key] = [0] * NSTAT
        return row

    def _read(self, st, s):
        row = st.get(str(s)) or [0] * NSTAT
        h = row[HANDS]
        pad = lambda a, b, prior: (a + prior * PRIOR_N) / (b + PRIOR_N)
        vpip = pad(row[VPIP], h, PRIORS["vpip"])
        pfr = pad(row[PFR], h, PRIORS["pfr"])
        pf_fold = pad(row[PF_FOLD], row[PF_FACED], PRIORS["pf_fold"])
        p_fold = pad(row[P_FOLD], row[P_FACED], PRIORS["p_fold"])
        agg = (row[P_BET] + PRIORS["agg"] * 6) / (row[P_CALL] + 6)
        return vpip, pfr, pf_fold, p_fold, agg, h

    # ----------------------------------------------------------- main logic
    def decide(self, obs, state):
        info = self._update(obs, state)
        action, note = self._choose(obs, state, info)
        return with_note(action, note[:190]), state

    def _icm_need(self, obs, legal, state, bettor_seat):
        """Equity needed to call under ICM (or None when stack too deep to matter)."""
        pl = [p for p in obs.players if p.status != "eliminated"]
        me = next(p for p in pl if p.seat == obs.hero_seat)
        pay = [p for p in obs.payouts if p > 0]
        if len(pl) <= 2 or len(pl) > 8:
            return None
        call_amt = min(legal.call_amount or 0, me.stack)
        if call_amt <= 0:
            return None
        gain = obs.pot_total - me.hand_committed
        stacks = [p.stack + (p.hand_committed if False else 0) for p in pl]
        h = [p.seat for p in pl].index(obs.hero_seat)
        base = list(stacks)
        win = list(stacks)
        win[h] += gain
        lose = list(stacks)
        lose[h] -= call_amt
        v = None
        for i, p in enumerate(pl):
            if p.seat == bettor_seat:
                v = i
        if v is not None:
            take = min(call_amt, pl[v].stack)
            win[v] = max(0, win[v] - take)
            lose[v] += call_amt
        if lose[h] <= 0:
            lose[h] = 0
        e_f = icm_equity(base, h, pay)
        e_w = icm_equity(win, h, pay)
        e_l = icm_equity(lose, h, pay) if lose[h] > 0 else 0.0
        if e_w - e_l <= 1e-9:
            return None
        return (e_f - e_l) / (e_w - e_l)

    def _position(self, obs):
        act = [s for s in obs.seats if next(p for p in obs.players if p.seat == s).status != "eliminated"]
        act.sort()
        n = len(act)
        btn = obs.button_seat
        after = [s for s in act if s > btn] + [s for s in act if s <= btn]
        # after[0] is SB (or first after dead button), after[-1] is button if alive
        idx = after.index(obs.hero_seat)
        if n == 2:
            return n, (0.0 if obs.hero_seat == (obs.small_blind_seat if obs.small_blind_seat is not None else -1) else 0.0), "hu"
        # preflop order: UTG ... BTN, SB, BB. idx 0 = SB, 1 = BB, 2 = UTG ...
        order = (idx - 2) % n
        frac = order / max(1, n - 1)  # 0 = UTG, ~0.75 = BTN, 1 = BB
        return n, frac, ("SB" if idx == 0 else "BB" if idx == 1 else "o")

    def _choose(self, obs, state, info):
        legal = obs.legal
        hole = obs.hole_cards
        me = next(p for p in obs.players if p.seat == obs.hero_seat)
        stack = me.stack
        bb = obs.blinds.big_blind
        to_call = legal.call_amount or 0
        n_act, pos, kind = self._position(obs)
        r = self._rand(state)
        hands = obs.hand_number
        phase = (state["phase0"] + hands // 18) % 2  # 0 tight image, 1 loose image
        opps = [p for p in obs.players if p.seat != obs.hero_seat and p.status != "eliminated"]
        reads = {p.seat: self._read(state["st"], p.seat) for p in opps}
        mean_vpip = sum(v[0] for v in reads.values()) / max(1, len(reads))
        eff = max([min(stack, p.stack) for p in opps if p.status != "folded"] or [stack])
        eff_bb = eff / bb
        live = [p for p in opps if p.status in ("active", "all_in")]
        n_live = len(live)

        if obs.street == "preflop":
            return self._preflop(obs, state, info, locals())
        return self._postflop(obs, state, info, locals())

    # --------------------------------------------------------------- preflop
    def _preflop(self, obs, state, info, L):
        legal, hole, stack, bb, to_call = L["legal"], L["hole"], L["stack"], L["bb"], L["to_call"]
        pos, kind, r, phase, reads = L["pos"], L["kind"], L["r"], L["phase"], L["reads"]
        eff_bb, n_act = L["eff_bb"], L["n_act"]
        score = chen(hole[0], hole[1])
        raises = info["raises"]
        raiser = info["raiser"]
        loose = 1.0 if phase else -1.0
        # who is left to act behind us (for steal fold equity)
        behind = [p for p in L["opps"] if p.status == "active"]
        fold_eq = 1.0
        for p in behind:
            fold_eq *= reads[p.seat][2] if p.street_committed < obs.current_bet_to else 0.6
        steal_bonus = 0.0
        if raises == 0:
            steal_bonus = max(0.0, (fold_eq - 0.3)) * 4.0
        tbl_tight = 1.0 - L["mean_vpip"]  # higher = tighter table
        icm_tight = 0.0
        pay = obs.payouts
        if obs.surviving_player_count <= 4 and obs.surviving_player_count > 3:
            icm_tight = 0.8
        elif obs.surviving_player_count == 4 or obs.surviving_player_count == 5:
            icm_tight = 0.8
        # short-stacked pressure: need top-3 finish
        if raises == 0:
            if n_act == 2:
                thr = 2.5
            else:
                thr = 8.8 - 4.2 * pos  # UTG 8.8 .. BTN ~5.6
                if kind == "SB":
                    thr = 5.5
                if kind == "BB":
                    thr = 99
            thr -= 0.9 * loose + steal_bonus
            thr += TUNE['open']
            thr += icm_tight * 0.5 * (1 if eff_bb < 20 else 0)
            if kind == "BB":
                # big blind with option
                if legal.check:
                    if score >= 9 and r < 0.7:
                        t = obs.current_bet_to + int(3 * bb)
                        a = clamp_raise_to(legal, t)
                        if a:
                            return a, "bb iso raise"
                    return check(), "bb check"
            if eff_bb <= 11:
                push_thr = 3.2 + 0.32 * eff_bb + 2.2 * (1 - pos) * (1 if n_act > 2 else 0)
                push_thr -= steal_bonus * 0.7
                push_thr += icm_tight * 0.6 + TUNE['push']
                if score >= push_thr:
                    a = clamp_raise_to(legal, legal.all_in_raise_to or stack)
                    if a:
                        return a, f"push {eff_bb:.0f}bb"
                return (check(), "chk") if legal.check else (fold(), "fold short")
            if score >= thr:
                size = (2.2 + 0.4 * (1 if phase else 0) + 0.3 * r) * bb
                if kind == "SB":
                    size += bb
                if score >= 12 and r < 0.15:
                    size = 3.5 * bb
                a = clamp_raise_to(legal, int(obs.current_bet_to + size - bb + bb))
                if a:
                    return a, f"open {score:.1f}>={thr:.1f}"
            if legal.check:
                return check(), "chk"
            if kind == "SB" and score >= thr - 1.5 and to_call <= bb / 2 + 1 and r < 0.6:
                return call(), "complete"
            return fold(), "fold open"

        # facing a raise (or limp-raise)
        if raiser is None:
            raiser = obs.actor_seat
        rd = reads.get(raiser, (0.3, 0.13, 0.55, 0.45, 1, 0))
        r_vpip, r_pfr, r_pf_fold = rd[0], rd[1], rd[2]
        raise_size = obs.current_bet_to
        cost_frac = to_call / max(1, stack)
        # tightness scales with opponent's raising range
        opener_adj = 0.0
        if rd[5] >= 12:
            opener_adj = -min(2.5, max(-2.5, (r_pfr - 0.13) * 22))
        elif r_pfr > 0.2:
            opener_adj = -1.0
        # defend vs a raise
        raise_bbs = raise_size / bb
        # 3-bet / 4-bet value & bluff region
        value3 = 10.5 + (1.0 if raises >= 2 else 0) * 2.5 + 0.8 * (-loose) + (opener_adj * 0.4) + icm_tight_adj(obs)
        call_thr = 8.0 + 0.55 * max(0, raise_bbs - 3) + opener_adj + (0.5 if kind == "o" else -0.3) + icm_tight_adj(obs) + 0.6 * (-loose)
        if kind == "BB":
            call_thr -= 1.0 * (1 if raises == 1 else 0)
        if raises >= 2:
            call_thr += 1.6
        call_thr += TUNE['call']
        value3 += TUNE['v3']
        stack_commit = (to_call + me_street(obs)) / max(1, stack + me_street(obs))
        if eff_bb <= 14:
            # shove or fold zone
            sh = 7.4 + 0.1 * eff_bb - (r_pfr - 0.13) * 8 + icm_tight_adj(obs) * 1.2
            if score >= sh + (1.0 if raises >= 2 else 0):
                a = clamp_raise_to(legal, legal.all_in_raise_to or stack)
                if a:
                    return a, "shove vs raise"
                return call(), "call shove-zone"
            if to_call <= bb and score >= call_thr - 1.5:
                return call(), "cheap call"
            return fold(), "fold short vs raise"
        # light 3-bet bluffs with blockers when raiser folds often and we are in position-ish
        bluff3 = (
            raises == 1
            and r_pf_fold > 0.62
            and rd[5] >= 10
            and score >= 5.5
            and score < call_thr
            and L["r"] < 0.35
            and obs.surviving_player_count > 3
        )
        if score >= value3 or bluff3:
            mult = 3.0 if kind in ("o",) else 3.6
            t = int(raise_size * mult)
            a = clamp_raise_to(legal, t)
            if a and not (a.amount and a.amount >= stack + me_street(obs) and score < 11):
                return a, "3bet value" if score >= value3 else "3bet bluff"
            if to_call < stack:
                return call(), "call 3b"
        if score >= call_thr and stack_commit < 0.4 + 0.02 * (score - call_thr) * 5:
            return call(), f"call {score:.1f}>={call_thr:.1f}"
        if score >= call_thr and to_call <= 0.18 * stack:
            return call(), "call small"
        if legal.check:
            return check(), "chk"
        return fold(), f"fold {score:.1f}"

    # -------------------------------------------------------------- postflop
    def _postflop(self, obs, state, info, L):
        legal, hole, stack, bb, to_call = L["legal"], L["hole"], L["stack"], L["bb"], L["to_call"]
        r, phase, reads = L["r"], L["phase"], L["reads"]
        s = post_strength(hole, obs.board)
        pot = obs.pot_total
        n_live = max(1, L["n_live"])
        street = obs.street
        bettor = info["bettor"]
        me = next(p for p in obs.players if p.seat == obs.hero_seat)
        loose = 1.0 if phase else 0.0
        live = L["live"]
        # multiway discount
        mw = 0.03 * (n_live - 1)
        opp_fold = 1.0
        for p in live:
            opp_fold *= reads[p.seat][3]
        # aggressor info
        was_agg = False
        if to_call == 0:
            # we may bet
            sizes = lambda f: obs.current_bet_to + max(int(bb), int(pot * f))
            station = max((reads[p.seat][4] for p in live), default=1) < 0.25 and opp_fold < 0.35
            bluff_ok = opp_fold > 0.35 and n_live <= 2
            if s >= 0.72 - mw:
                f = 0.7 if r < 0.7 else 0.4
                if station:
                    f = 0.8
                a = clamp_raise_to(legal, sizes(f))
                if a:
                    return a, f"value {s:.2f}"
            if info["raiser"] == obs.hero_seat and street == "flop" and n_live <= 2 and r < 0.35 + 0.8 * (opp_fold - 0.3) + 0.1 * loose and s >= 0.1:
                a = clamp_raise_to(legal, sizes(0.5))
                if a:
                    return a, f"cbet fold={opp_fold:.2f}"
            if s >= 0.5 - mw and not station and r < 0.55 + 0.2 * loose:
                a = clamp_raise_to(legal, sizes(0.5))
                if a:
                    return a, f"thin {s:.2f}"
            if s >= 0.45 and station and r < 0.6:
                a = clamp_raise_to(legal, sizes(0.5))
                if a:
                    return a, "thin vs station"
            if bluff_ok and street != "river" and 0.15 <= s < 0.5 and r < (0.25 + 0.3 * (opp_fold - 0.35) + 0.12 * loose):
                a = clamp_raise_to(legal, sizes(0.6))
                if a:
                    return a, f"semi/bluff fold={opp_fold:.2f}"
            if bluff_ok and street == "river" and s < 0.25 and r < 0.12 + 0.25 * (opp_fold - 0.35) and stack > pot * 0.4:
                a = clamp_raise_to(legal, sizes(0.7))
                if a:
                    return a, "river bluff"
            return check(), "check"
        # facing a bet
        need = to_call / (pot + 0.0)
        bd = reads.get(bettor, (0.3, 0.13, 0.55, 0.45, 1.0, 0)) if bettor is not None else (0.3, 0.13, 0.55, 0.45, 1.0, 0)
        agg = bd[4]
        hands_known = bd[5]
        icm = self._icm_need(obs, legal, state, bettor) if (to_call >= 0.2 * stack or to_call >= stack) else None
        # margin: bettors who bet a lot are bluffing more often
        margin = 0.0
        if hands_known >= 12:
            margin = -0.09 if agg > 1.6 else (0.07 if agg < 0.55 else 0.0)
        margin += mw + TUNE['pm'] + 0.12 * min(1.5, to_call / max(1, pot - to_call))
        pot_odds = to_call / (pot + to_call - 0.0)
        # pot already includes the bet in obs.pot_total
        pot_odds = to_call / pot if pot > 0 else 1.0
        pot_odds = to_call / (pot)  # pot includes opponent bet; hero adds to_call
        need_eq = to_call / (pot + to_call)
        if icm is not None:
            need_eq = max(need_eq, 0.5 * need_eq + 0.5 * icm)
        # raise for value
        strong = s >= 0.82 - mw
        if strong and legal.can_raise and r < 0.8:
            a = clamp_raise_to(legal, int(obs.current_bet_to + pot * 0.75 + to_call))
            if a:
                return a, f"raise value {s:.2f}"
        # bluff raise vs high fold bettors, rarely
        if (
            legal.can_raise
            and 0.2 <= s < 0.5
            and street != "river"
            and bd[3] > 0.58
            and bd[5] >= 15
            and n_live <= 2
            and r < 0.12 + 0.1 * loose
            and to_call < 0.3 * stack
        ):
            a = clamp_raise_to(legal, int(obs.current_bet_to * 2.6 + pot * 0.3))
            if a:
                return a, "bluff raise"
        if s - margin >= need_eq + 0.02 or (s >= 0.3 and street != "river" and s - margin >= need_eq - 0.06 and need_eq < 0.25):
            return call(), f"call s={s:.2f} need={need_eq:.2f}"
        return fold(), f"fold s={s:.2f} need={need_eq:.2f}"


def me_street(obs):
    return next(p.street_committed for p in obs.players if p.seat == obs.hero_seat)


def icm_tight_adj(obs):
    n = obs.surviving_player_count
    if n <= 3:
        return 0.3
    if n <= 5:
        return 0.9
    if n <= 7:
        return 0.4
    return 0.0


def make_bot():
    return AdaptiveBot()
