"""Pause-and-predict study packs: imported PHH hands as preprocessed JSON for the arena UI.

The arena serves these files as data (``arena.api.study``) and never imports this
module. Each hand is replayed through :func:`strategy.data.phh_import.import_hand`;
every frame is the arena's own seat projection for the studied player
(:func:`arena.table.projections.table_view`), so a frame shows exactly what that seat
saw: its own cards, the public action and cards shown at showdown. Every hole card is
kept separately under ``result`` for the end-of-hand reveal.

Each of the studied player's decisions carries the frozen spot: the canonical action
menu (:func:`strategy.rules.adapter.canonical_actions`), what the player did mapped onto
it, and features from :class:`strategy.features.engine.FeatureEngine` run on the frozen
observation only (pot odds, position, stack depth, hand strength, draws, board).

A pack is a directory: ``pack.json`` (provenance and counts), ``index.json`` (one
summary per hand) and ``hands/<id>.json``.

    python -m strategy.data.study                    # the cached corpus -> artifacts/study/pluribus
    python -m strategy.data.study --source fixture   # the committed fixture hands -> artifacts/study/pluribus-sample
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from arena.table.projections import Viewer, project_events, table_view
from arena.table.service import Table
from strategy.data.fetch import PLURIBUS, FetchError
from strategy.data.phh_import import (RULESET_ID, TRANSFORM_VERSION, Decision, ImportedHand, ImportRejected,
                                      import_hand, iter_source, parse_phh, provenance_for, source_dir)
from strategy.features.engine import DecisionFeatures, FeatureEngine

STUDY_FORMAT = "arena.study-hand"
STUDY_SCHEMA_VERSION = 1
HERO = "Pluribus"
REPO = Path(__file__).resolve().parents[2]
STUDY_DIR = REPO / "artifacts" / "study"
SAMPLE_PACK = STUDY_DIR / "pluribus-sample"
FULL_PACK = STUDY_DIR / "pluribus"

MADE_NAMES = ("no made hand", "weak pair", "middle pair", "top pair", "overpair", "two pair", "trips or a set",
              "straight", "flush", "full house", "quads or better")
SIZE_NAMES = {"raise_min": "min", "raise_pot": "pot", "raise_all_in": "all-in"}
FRACTION_NAMES = {0.33: "⅓ pot", 0.5: "½ pot", 0.75: "¾ pot", 1.0: "pot", 1.5: "1.5× pot", 2.5: "2.5× pot"}
FRAME_KEYS = ("street", "actor_seat", "board", "pot_total", "settled_pot", "current_bet_to", "pots", "players")


EVENT_TYPES = frozenset({"HandStarted", "AntePosted", "BlindPosted", "ActionApplied", "BoardDealt", "CardsRevealed", "PotAwarded",
                         "ExcessReturned", "HandCompleted"})
"""Public events the study log narrates; dealing, decision requests and bet collection are left out."""


def _frame(table: Table, hero: int, kind: str) -> dict[str, Any]:
    view = table_view(table, Viewer.for_seat(hero)).model_dump(mode="json")
    out = {"kind": kind, "seq": table.log[-1].seq}
    out.update({k: view[k] for k in FRAME_KEYS})
    out["players"] = [{k: v for k, v in p.items() if v is not None} for p in view["players"]]
    if kind == "hand_end":
        out["actor_seat"] = None
    return out


def size_name(key: str) -> str:
    if key.startswith("raise_fraction:"):
        f = float(key.split(":")[1])
        return FRACTION_NAMES.get(f, f"{f:g}× pot")
    return SIZE_NAMES.get(key, key)


def options(d: Decision) -> list[dict[str, Any]]:
    obs = d.observation
    bet = obs.current_bet_to == 0
    after_call = max(1, obs.pot_total + (obs.legal.call_amount or 0))
    out = []
    for c in d.menu:
        if c.kind in ("fold", "check", "call"):
            amount = obs.legal.call_amount if c.kind == "call" else None
            out.append({"key": c.key, "kind": c.kind, "amount": amount})
            continue
        assert c.amount is not None
        out.append({"key": c.key, "kind": "raise", "amount": c.amount, "size": size_name(c.key),
                    "fraction": round((c.amount - obs.current_bet_to) / after_call, 3), "bet": bet})
    return out


def facing(street: str, raises: int, limpers: int, to_call: int) -> str:
    if street == "preflop":
        if raises == 0:
            return "limped pot" if limpers else "unopened pot"
        return ("facing a raise", "facing a 3-bet", "facing a 4-bet")[min(raises, 3) - 1]
    if to_call == 0:
        return "no bet yet this street"
    return ("facing a bet", "facing a raise", "facing a re-raise")[min(raises, 3) - 1]


def spot(d: Decision, f: DecisionFeatures, position: str) -> dict[str, Any]:
    s, c = f.state, f.cards
    draws = []
    if c.draws.flush_draw:
        draws.append("nut flush draw" if c.draws.nut_flush_draw else "flush draw")
    if c.draws.oesd:
        draws.append("open-ended straight draw")
    elif c.draws.gutshot:
        draws.append("gutshot")
    board = []
    if d.street != "preflop":
        t = c.texture
        board.append("many draws" if t.dynamic >= 0.4 else "few draws")
        if t.paired:
            board.append("paired")
        if t.flush_possible:
            board.append("flush possible")
        elif t.flush_draw_possible:
            board.append("flush draw possible")
        if t.straight_windows:
            board.append("straight possible")
    return {
        "position": position,
        "in_position": bool(s.in_position),
        "facing": facing(d.street, s.raises, s.limpers, s.to_call),
        "pot": s.pot,
        "to_call": s.call_amount,
        "pot_odds": round(s.pot_odds, 4),
        "pot_bb": round(s.pot_bb, 2),
        "stack_bb": round(s.stack / max(1, s.bb), 1),
        "effective_bb": round(s.effective_bb, 1),
        "spr": round(s.spr, 2),
        "opponents": s.opponents,
        "behind": s.behind,
        "raises": s.raises,
        "hand": c.label,
        "preflop_percentile": round(c.percentile, 3),
        "equity": round(c.equity, 3),
        "equity_source": c.equity_source,
        "made": MADE_NAMES[c.made.level] if c.made is not None else None,
        "outs": c.draws.outs,
        "draws": draws,
        "board": board,
    }


def pluribus_choice(d: Decision) -> dict[str, Any]:
    a = d.action
    out: dict[str, Any] = {"kind": d.category, "amount": a.amount if a.type == "raise_to" else None}
    if a.type == "call":
        out["amount"] = d.observation.legal.call_amount
    if d.size is not None:
        out.update(option=d.size.option.key, exact=d.size.log_error == 0.0, fraction=round(d.size.fraction, 3),
                   size=size_name(d.size.option.key))
    else:
        out["option"] = d.category
    return out


def study_hand(hand: ImportedHand, hero: int, frames: list[dict[str, Any]]) -> dict[str, Any]:
    table = hand.table
    seats = sorted(hand.seat_names)
    engine = FeatureEngine(RULESET_ID, hero, seats)
    record = table.hands[0]
    decisions = []
    for d in hand.decisions:
        if d.seat != hero:
            continue
        features = engine.update(d.observation)
        decisions.append({"frame": d.frame, "street": d.street, "options": options(d),
                          "pluribus": pluribus_choice(d), "spot": spot(d, features, hand.positions[hero])})
    log = [e for e in table.log if e.hand_id == record.hand_id and e.type in EVENT_TYPES]
    events = [e.model_dump(mode="json") for e in project_events(log, Viewer.for_seat(hero))]
    start = record.start_stacks
    final = hand.finishing_stacks
    session, _, number = hand.provenance.path.removeprefix(f"{PLURIBUS.subdir}/").removesuffix(".phh").partition("/")
    return {
        "format": STUDY_FORMAT,
        "schema_version": STUDY_SCHEMA_VERSION,
        "id": hand.hand_id,
        "title": f"Session {session}, hand {number}",
        "ruleset_id": RULESET_ID,
        "rules_hash": hand.rules_hash,
        "blinds": {"small_blind": record.blinds[0], "big_blind": record.blinds[1]},
        "hero_seat": hero,
        "hero_name": hand.seat_names[hero],
        "button_seat": record.positions.button,
        "small_blind_seat": record.positions.small_blind,
        "big_blind_seat": record.positions.big_blind,
        "seats": [{"seat": s, "name": hand.seat_names[s], "position": hand.positions[s], "start_stack": start[s]}
                  for s in seats],
        "hole_cards": hand.hole_cards[hero],
        "frames": frames,
        "events": events,
        "decisions": decisions,
        "result": {
            "final_stacks": {str(s): final[s] for s in seats},
            "net": {str(s): final[s] - start[s] for s in seats},
            "hole_cards": {str(s): hand.hole_cards[s] for s in seats},
            "board": hand.board,
            "showdown": sorted(hand.showdown),
            "notes": hand.notes,
        },
        "source": hand.provenance.as_dict(),
    }


STREET_RANK = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}


def summary(doc: dict[str, Any]) -> dict[str, Any]:
    hero = doc["hero_seat"]
    decisions = doc["decisions"]
    last = max((STREET_RANK[d["street"]] for d in decisions), default=0)
    bb = doc["blinds"]["big_blind"]
    pot = max((f["pot_total"] for f in doc["frames"]), default=0)
    return {
        "id": doc["id"],
        "title": doc["title"],
        "position": doc["seats"][hero]["position"],
        "hand": doc["decisions"][0]["spot"]["hand"] if decisions else None,
        "decisions": len(decisions),
        "last_street": ("preflop", "flop", "turn", "river")[last],
        "showdown": hero in doc["result"]["showdown"],
        "pot_bb": round(pot / bb, 1),
        "net_bb": round(doc["result"]["net"][str(hero)] / bb, 2),
    }


def build_hands(texts: Iterable[tuple[str, bytes]], hero_name: str = HERO,
                errors: dict[str, str] | None = None) -> Iterable[dict[str, Any]]:
    for rel, data in texts:
        try:
            phh = parse_phh(data.decode("utf-8"))
            if hero_name not in phh.players:
                raise ImportRejected("no_hero", f"no player named {hero_name}")
            hero = phh.players.index(hero_name)
            hand = import_hand(data.decode("utf-8"), provenance_for(rel, data),
                               on_frame=lambda t, kind, h=hero: _frame(t, h, kind))
        except ImportRejected as exc:
            if errors is not None:
                errors[rel] = str(exc)
            continue
        yield study_hand(hand, hero, hand.frames)


def _dump(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def write_pack(out: Path, texts: Iterable[tuple[str, bytes]], *, pack_id: str, title: str,
               source_label: str) -> dict[str, Any]:
    if out.exists() and any(out.iterdir()) and not (out / "pack.json").exists():
        raise FileExistsError(f"{out} exists and is not a study pack; refusing to replace it")
    final, out = out, out.with_name(f".{out.name}.building")
    if out.exists():
        shutil.rmtree(out)
    (out / "hands").mkdir(parents=True)
    index = []
    errors: dict[str, str] = {}
    decisions = 0
    for doc in build_hands(texts, errors=errors):
        (out / "hands" / f"{doc['id']}.json").write_text(_dump(doc), encoding="utf-8")
        index.append(summary(doc))
        decisions += len(doc["decisions"])
    meta = {
        "format": "arena.study-pack", "schema_version": STUDY_SCHEMA_VERSION, "id": pack_id, "title": title,
        "hero": HERO, "ruleset_id": RULESET_ID, "hands": len(index), "decisions": decisions,
        "rejected": dict(sorted(errors.items())), "source": source_label, "transform": TRANSFORM_VERSION,
        "dataset": {"repo": PLURIBUS.repo, "commit": PLURIBUS.commit, "subdir": PLURIBUS.subdir,
                    "content_sha256": PLURIBUS.content_sha256, "licence": PLURIBUS.licence,
                    "citation": PLURIBUS.citation, "dataset_doi": PLURIBUS.dataset_doi,
                    "paper_doi": PLURIBUS.paper_doi},
        "caveat": "Pluribus played six-max 100 BB cash games. One sampled action from a mixed strategy is not "
                  "the only good play, and none of it is tuned to the eight-player tournament points.",
    }
    (out / "pack.json").write_text(json.dumps(meta, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
    (out / "index.json").write_text(_dump({"hands": index}), encoding="utf-8")
    if final.exists():
        shutil.rmtree(final)
    out.rename(final)
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build a pause-and-predict study pack from PHH hands.")
    ap.add_argument("--source", default="cache", help="cache (default), fixture, or a directory of .phh files")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--pack", default=None, help="pack id (default: pluribus, or pluribus-sample for the fixture)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    try:
        root = source_dir(args.source)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    sample = args.source == "fixture"
    pack_id = args.pack or ("pluribus-sample" if sample else "pluribus")
    out = args.out or (SAMPLE_PACK if sample else FULL_PACK)
    texts = iter_source(root)
    if args.limit is not None:
        texts = (t for k, t in enumerate(texts) if k < args.limit)
    title = "Pluribus sample hands" if sample else "Pluribus: all 10,000 hands"
    meta = write_pack(out, texts, pack_id=pack_id, title=title, source_label=args.source)
    print(f"{meta['hands']} hands, {meta['decisions']} {HERO} decisions -> {out}"
          + (f"; {len(meta['rejected'])} rejected" if meta["rejected"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
