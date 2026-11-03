"""HandHQ proof of concept: three real PokerStars 25NL records (two folded out, one showdown)
replay through the arena engine with cents, unequal stacks and placeholder hole cards."""

from __future__ import annotations

import pytest

from strategy.data import handhq
from strategy.data.phh_import import FIXTURE_DIR, ImportRejected, _replay

SAMPLE = FIXTURE_DIR.parent / "handhq" / "ps-25nl-sample.phhs"


def test_sample_replays_and_payoffs_match_net_of_rake():
    report = handhq.run([SAMPLE])
    assert (report.hands, report.replayed, sum(report.rejected.values())) == (3, 3, 0)
    assert report.payoff["checked"] == 2
    assert report.rake_cents == 50


def test_conversion_uses_cents_shown_cards_and_distinct_placeholders():
    docs = handhq.load_phhs(SAMPLE)
    conv = handhq.convert(docs["5"], "sample/5")
    assert conv.phh.blinds[:2] == (10, 25) and conv.phh.starting_stacks[5] == 9990
    assert conv.known == {2, 3} and conv.showdown_known
    dealt = [a.split()[3] for a in conv.phh.actions if a.startswith("d dh")]
    assert dealt[2:4] == ["QdAs", "TdJc"]
    cards = [c[i:i + 2] for c in dealt for i in (0, 2)] + ["9c", "5d", "Ts", "8c", "2h"]
    assert len(set(cards)) == len(cards)


def test_showdown_is_decided_by_the_engine():
    docs = handhq.load_phhs(SAMPLE)
    conv = handhq.convert(docs["5"], "sample/5")
    prov = handhq.Provenance(dataset=handhq.DATASET, path="data/handhq/sample/5", sha256="", url=handhq.REPO_URL,
                             commit=handhq.COMMIT, licence=handhq.LICENCE, citation=handhq.CITATION,
                             dataset_doi="", paper_doi="")
    hand = _replay(conv.phh, prov, None, mixed_stacks=True)
    net = {s: hand.finishing_stacks[s] - conv.phh.starting_stacks[s] for s in hand.seat_names}
    assert net[3] > 0 > net[2] and sum(net.values()) == 0


def test_only_six_handed_six_max_hands_are_eligible():
    doc = dict(handhq.load_phhs(SAMPLE)["4"], seat_count=9)
    with pytest.raises(ImportRejected) as exc:
        handhq.convert(doc, "sample/4")
    assert exc.value.code == "not_six_max"
