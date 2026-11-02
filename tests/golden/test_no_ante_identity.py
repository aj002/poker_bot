"""Ante support must leave ante-free play byte-identical (``tests/support/no_ante_identity.json``)."""

import json

from tests.support.identity import PIN, digests


def test_ante_free_play_matches_the_pinned_digests():
    assert digests() == json.loads(PIN.read_text())
