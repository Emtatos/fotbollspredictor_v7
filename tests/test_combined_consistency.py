# tests/test_combined_consistency.py
"""
Regressionstest: Odds & Value och Flera Matcher ska ge identiska kombinerade
sannolikheter, gain, halvgarderingsurval och tecken på samma indata.

Fixture: vecka 36 (13 engelska matcher, 7 halvgarderingar).
"""
import numpy as np
import pytest

from combined_probability import (
    DEFAULT_WEIGHTS,
    build_combined_matches,
    combined_from_current_round,
    combined_from_matchday_matches,
    describe_sources_used,
)
from matchday_import import MatchdayMatch, _make_key
from odds_tool import MatchOddsReport, OddsEntry
from ui_utils import get_halfguard_sign_combined, pick_half_guards_combined

NUM_HALFGUARDS = 7

# (home, away, (streck 1/X/2), (odds 1/X/2))
WEEK36 = [
    ("Hull", "Aston Villa", (39, 25, 36), (4.10, 3.65, 1.97)),
    ("Brighton", "Leeds", (59, 23, 18), (1.96, 3.60, 4.25)),
    ("Fulham", "Crystal Palace", (45, 28, 27), (2.35, 3.50, 3.20)),
    ("Manchester City", "Coventry", (74, 13, 13), (1.19, 8.00, 16.00)),
    ("Nottingham", "Tottenham", (42, 27, 31), (2.63, 3.45, 2.85)),
    ("Brentford", "Sunderland", (61, 21, 18), (1.65, 4.00, 5.80)),
    ("Burnley", "Bristol City", (47, 26, 27), (1.88, 3.75, 4.35)),
    ("Millwall", "Bolton", (63, 21, 16), (1.57, 4.00, 6.50)),
    ("Portsmouth", "Cardiff", (42, 28, 30), (2.38, 3.50, 3.05)),
    ("Queens Park Rangers", "Middlesbrough", (36, 27, 37), (3.45, 3.60, 2.12)),
    ("Sheffield U", "Norwich", (46, 29, 25), (2.60, 3.40, 2.80)),
    ("West Bromwich", "Watford", (55, 25, 20), (1.96, 3.55, 4.10)),
    ("Swansea", "Wrexham", (53, 27, 20), (2.25, 3.45, 3.30)),
]

# Deterministiska mockade modellprediktioner (p1, px, p2) per match.
MOCK_MODEL = [
    np.array([0.30, 0.27, 0.43]),
    np.array([0.52, 0.26, 0.22]),
    np.array([0.42, 0.29, 0.29]),
    np.array([0.80, 0.12, 0.08]),
    np.array([0.36, 0.28, 0.36]),
    np.array([0.58, 0.24, 0.18]),
    np.array([0.50, 0.27, 0.23]),
    np.array([0.60, 0.24, 0.16]),
    np.array([0.40, 0.29, 0.31]),
    np.array([0.30, 0.28, 0.42]),
    np.array([0.38, 0.30, 0.32]),
    np.array([0.49, 0.28, 0.23]),
    np.array([0.44, 0.24, 0.32]),
]

MATCHES = [(h, a) for h, a, _, _ in WEEK36]


def _matchday_matches():
    """Odds & Value-formen: MatchdayMatch-objekt från kupongskanning."""
    out = []
    for home, away, streck, odds in WEEK36:
        entry = OddsEntry(bookmaker="test", home=odds[0], draw=odds[1], away=odds[2])
        out.append(MatchdayMatch(
            home_team=home,
            away_team=away,
            match_key=_make_key(home, away),
            odds_entries=[entry],
            streck={"1": float(streck[0]), "X": float(streck[1]), "2": float(streck[2])},
            odds_report=MatchOddsReport(home_team=home, away_team=away, bookmaker_odds=[entry]),
            has_odds=True,
            has_streck=True,
        ))
    return out


def _current_round():
    """Flera Matcher-formen: current_round i session state."""
    odds_by_key = {}
    streck_by_key = {}
    for home, away, streck, odds in WEEK36:
        key = _make_key(home, away)
        odds_by_key[key] = [OddsEntry(bookmaker="test", home=odds[0], draw=odds[1], away=odds[2])]
        streck_by_key[key] = {"1": float(streck[0]), "X": float(streck[1]), "2": float(streck[2])}
    return {"matches": MATCHES, "odds": odds_by_key, "streck": streck_by_key}


def _via_odds_and_value(model_probs):
    return combined_from_matchday_matches(_matchday_matches(), model_probs)


def _via_flera_matcher(model_probs):
    return combined_from_current_round(MATCHES, _current_round(), model_probs, _make_key)


def _gain(cm):
    return sorted(cm.probs, reverse=True)[1]


def _assert_positionwise_identical(a, b):
    assert len(a) == len(b) == len(WEEK36)
    for i in range(len(WEEK36)):
        assert a[i].home_team == b[i].home_team == WEEK36[i][0]
        assert a[i].away_team == b[i].away_team == WEEK36[i][1]
        assert abs(a[i].prob_1 - b[i].prob_1) < 1e-9, i
        assert abs(a[i].prob_x - b[i].prob_x) < 1e-9, i
        assert abs(a[i].prob_2 - b[i].prob_2) < 1e-9, i
        assert abs(_gain(a[i]) - _gain(b[i])) < 1e-9, i
        assert a[i].sources == b[i].sources, i

    guards_a = pick_half_guards_combined(a, NUM_HALFGUARDS)
    guards_b = pick_half_guards_combined(b, NUM_HALFGUARDS)
    assert len(guards_a) == len(guards_b) == NUM_HALFGUARDS
    for j in range(NUM_HALFGUARDS):
        assert guards_a[j] == guards_b[j], j
        assert (
            get_halfguard_sign_combined(a[guards_a[j]])
            == get_halfguard_sign_combined(b[guards_b[j]])
        ), j


class TestWeek36Consistency:
    def test_with_model_positionwise_identical(self):
        a = _via_odds_and_value(MOCK_MODEL)
        b = _via_flera_matcher(MOCK_MODEL)
        _assert_positionwise_identical(a, b)
        assert all(cm.sources["model"] for cm in a)
        assert describe_sources_used(a) == describe_sources_used(b) == [
            "odds (50%)", "modell (35%)", "streck (15%)",
        ]

    def test_without_model_positionwise_identical(self):
        no_model = [None] * len(WEEK36)
        a = _via_odds_and_value(no_model)
        b = _via_flera_matcher(no_model)
        _assert_positionwise_identical(a, b)
        assert not any(cm.sources["model"] for cm in a)
        assert describe_sources_used(a) == describe_sources_used(b) == [
            "odds (50%)", "streck (15%)",
        ]

    def test_without_model_is_odds_streck_renormalized(self):
        """Ingen vy får tyst falla tillbaka till något annat än odds+streck (77/23)."""
        no_model = [None] * len(WEEK36)
        for cm, (_, _, streck, odds) in zip(_via_odds_and_value(no_model), WEEK36):
            raw = np.array([1 / o for o in odds])
            fair = raw / raw.sum()
            s = np.array(streck, dtype=float) / sum(streck)
            w_o = DEFAULT_WEIGHTS["odds"] / (DEFAULT_WEIGHTS["odds"] + DEFAULT_WEIGHTS["streck"])
            expected = w_o * fair + (1 - w_o) * s
            np.testing.assert_allclose(cm.probs, expected, atol=1e-9)

    def test_partial_model_shows_coverage_in_source_text(self):
        partial = list(MOCK_MODEL)
        partial[3] = None
        partial[12] = None
        a = _via_odds_and_value(partial)
        b = _via_flera_matcher(partial)
        _assert_positionwise_identical(a, b)
        assert describe_sources_used(a) == describe_sources_used(b) == [
            "odds (50%)", "modell (35%, 11/13 matcher)", "streck (15%)",
        ]

    def test_swansea_wrexham_same_sign_in_both_views(self):
        a = _via_odds_and_value(MOCK_MODEL)
        b = _via_flera_matcher(MOCK_MODEL)
        idx = 12
        assert a[idx].home_team == "Swansea"
        assert get_halfguard_sign_combined(a[idx]) == get_halfguard_sign_combined(b[idx])
        assert a[idx].probs.tolist() == b[idx].probs.tolist()


class TestBuildCombinedMatches:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            build_combined_matches(MATCHES, [None] * 13, [None] * 12, [None] * 13)

    def test_preserves_order(self):
        cms = build_combined_matches(
            MATCHES,
            [odds for _, _, _, odds in WEEK36],
            [streck for _, _, streck, _ in WEEK36],
            MOCK_MODEL,
        )
        assert [(c.home_team, c.away_team) for c in cms] == MATCHES
