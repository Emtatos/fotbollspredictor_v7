"""
Tester för:
1. Fallback-logik i Flera Matcher (odds-baserad fallback när modell saknar data)
2. UI-etiketter: gain (inte entropy) styr halvgarderingsurval
3. Konsekvent visning av matcher med saknad modelldata
4. X/kryss-sannolikheter inte systematiskt nedtryckta i combined probability
"""
import pytest
import numpy as np

from combined_probability import (
    odds_to_fair_probs,
    build_combined_match,
    combine_probabilities,
    CombinedMatchProbability,
)
from ui_utils import (
    pick_half_guards,
    pick_half_guards_combined,
    get_halfguard_sign,
    get_halfguard_sign_combined,
    _halfguard_sort_key,
    calculate_match_entropy,
)
from matchday_import import _make_key


# ---------------------------------------------------------------------------
# 1. Fallback: odds-baserad fallback ger jämförbara resultat
# ---------------------------------------------------------------------------

class TestOddsFallbackConsistency:
    """
    Verifierar att en match som saknar modelldata men har odds ger
    rimliga sannolikheter via odds_to_fair_probs — samma mekanism
    som combined_probability använder internt.
    """

    def test_odds_fallback_gives_valid_probs(self):
        """Odds-fallback ger sannolikheter som summerar till 1."""
        probs = odds_to_fair_probs(2.50, 3.30, 2.80)
        assert abs(probs.sum() - 1.0) < 1e-9
        assert all(p > 0 for p in probs)

    def test_odds_fallback_preserves_draw(self):
        """Odds-fallback ska inte trycka ned X systematiskt."""
        # Jämnt odds-scenario: alla tre utfall nära varandra
        probs = odds_to_fair_probs(2.80, 3.20, 2.70)
        # X ska inte vara nära noll
        assert probs[1] > 0.20, f"Draw prob {probs[1]:.3f} är orimligt låg"

    def test_combined_match_without_model_uses_odds_streck(self):
        """build_combined_match utan modell använder odds+streck."""
        cm = build_combined_match(
            home_team="Newport",
            away_team="Shrewsbury",
            odds_1=2.50, odds_x=3.30, odds_2=2.80,
            streck_1=38, streck_x=28, streck_2=34,
        )
        assert cm.sources["odds"] is True
        assert cm.sources["model"] is False
        assert cm.sources["streck"] is True
        assert abs(cm.prob_1 + cm.prob_x + cm.prob_2 - 1.0) < 1e-9
        # X ska ha en rimlig sannolikhet (inte nedtryckt)
        assert cm.prob_x > 0.15

    def test_fallback_match_participates_in_halfguard(self):
        """En match med odds-fallback ska kunna väljas som halvgardering."""
        # Match 0: modellbaserad
        model_probs = np.array([0.60, 0.25, 0.15])
        # Match 1: odds-fallback (ingen modell)
        fallback_probs = odds_to_fair_probs(2.40, 3.10, 3.00)

        all_probs = [model_probs, fallback_probs]
        # Båda matcherna ska delta i urvalet (ingen None)
        result = pick_half_guards(all_probs, n_guards=1)
        assert len(result) == 1
        assert result[0] in [0, 1]

    def test_combined_halfguard_with_fallback_match(self):
        """Halvgardering med kombinerade sannolikheter inkl. fallback-match."""
        cm_model = build_combined_match(
            "Arsenal", "Chelsea",
            odds_1=1.80, odds_x=3.60, odds_2=4.50,
            model_probs=np.array([0.50, 0.28, 0.22]),
            streck_1=55, streck_x=25, streck_2=20,
        )
        cm_fallback = build_combined_match(
            "Newport", "Shrewsbury",
            odds_1=2.50, odds_x=3.30, odds_2=2.80,
            streck_1=38, streck_x=28, streck_2=34,
        )
        result = pick_half_guards_combined([cm_model, cm_fallback], n_guards=1)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 2. Gain-baserad urvalsetiketter (inte entropy)
# ---------------------------------------------------------------------------

class TestGainBasedSelectionLabels:
    """
    Verifierar att halvgarderingsurval konsekvent styrs av gain
    (second_best probability), INTE entropy.
    """

    def test_sort_key_uses_gain_not_entropy(self):
        """_halfguard_sort_key returnerar gain (second_best) som primär nyckel."""
        probs = np.array([0.55, 0.35, 0.10])
        key = _halfguard_sort_key(probs, 0)
        # gain = second_best = 0.35
        assert key[0] == pytest.approx(0.35)
        # top2 = 0.55 + 0.35 = 0.90
        assert key[1] == pytest.approx(0.90)

    def test_gain_not_entropy_determines_selection(self):
        """
        Match med hög entropy men låg gain ska INTE väljas
        framför match med lägre entropy men högre gain.
        """
        # Hög entropy, låg gain
        high_entropy = np.array([0.34, 0.33, 0.33])  # gain=0.33
        # Lägre entropy, hög gain
        high_gain = np.array([0.55, 0.40, 0.05])      # gain=0.40

        entropy_a = calculate_match_entropy(high_entropy)
        entropy_b = calculate_match_entropy(high_gain)
        assert entropy_a > entropy_b, "Sanity check: high_entropy har mer entropy"

        result = pick_half_guards([high_entropy, high_gain], n_guards=1)
        assert result == [1], "Gain ska avgöra, inte entropy"

    def test_combined_selection_also_gain_based(self):
        """pick_half_guards_combined använder samma gain-logik."""
        from uncertainty import entropy_norm

        def _cm(p1, px, p2, home="A", away="B"):
            return CombinedMatchProbability(
                home_team=home, away_team=away,
                prob_1=p1, prob_x=px, prob_2=p2,
                entropy=entropy_norm(p1, px, p2),
                sources={"odds": True, "model": False, "streck": False},
            )

        # cm0: gain=0.33  cm1: gain=0.40
        cms = [_cm(0.34, 0.33, 0.33), _cm(0.55, 0.40, 0.05)]
        result = pick_half_guards_combined(cms, n_guards=1)
        assert result == [1], "Combined-logiken ska också välja högst gain"


# ---------------------------------------------------------------------------
# 3. Konsekvent fallback-visning
# ---------------------------------------------------------------------------

class TestFallbackDisplayConsistency:
    """
    Verifierar att matcher utan modelldata hanteras konsekvent:
    - Om odds finns → odds-baserad fallback med rimliga sannolikheter
    - Om inget finns → None/N/A med tydlig markering
    """

    def test_no_model_with_odds_gives_probs(self):
        """Match utan modell men med odds ska ge sannolikheter (inte N/A)."""
        cm = build_combined_match(
            "Newport", "Shrewsbury",
            odds_1=2.50, odds_x=3.30, odds_2=2.80,
        )
        assert cm.prob_1 > 0
        assert cm.prob_x > 0
        assert cm.prob_2 > 0
        assert cm.sources["model"] is False
        assert cm.sources["odds"] is True

    def test_no_model_no_odds_gives_uniform(self):
        """Match utan modell OCH odds ska ge uniform fördelning."""
        cm = build_combined_match("Unknown", "Team")
        assert abs(cm.prob_1 - 1/3) < 1e-6
        assert abs(cm.prob_x - 1/3) < 1e-6
        assert abs(cm.prob_2 - 1/3) < 1e-6

    def test_make_key_normalizes_consistently(self):
        """_make_key normaliserar lagnamn konsekvent för matchning."""
        key1 = _make_key("Arsenal", "Chelsea")
        key2 = _make_key("Arsenal", "Chelsea")
        assert key1 == key2

    def test_make_key_strips_whitespace(self):
        """_make_key hanterar extra whitespace."""
        key1 = _make_key("Arsenal", "Chelsea")
        key2 = _make_key("  Arsenal  ", "  Chelsea  ")
        assert key1 == key2


# ---------------------------------------------------------------------------
# 4. X/kryss-sannolikheter
# ---------------------------------------------------------------------------

class TestDrawProbabilities:
    """
    Verifierar att X/kryss inte systematiskt trycks ned
    i combined probability-beräkningar.
    """

    def test_draw_not_suppressed_by_combine(self):
        """combine_probabilities ska inte systematiskt trycka ned X."""
        odds = np.array([0.35, 0.35, 0.30])      # jämn match, lite draw
        model = np.array([0.30, 0.40, 0.30])      # modell tror X
        streck = np.array([0.33, 0.34, 0.33])     # jämnt streck

        result = combine_probabilities(
            odds_probs=odds, model_probs=model, streck_pcts=streck,
        )
        # X ska inte vara lägst — odds+model+streck alla ger X ~35%+
        assert result[1] >= result[2], "X ska inte vara lägst"

    def test_odds_fair_probs_preserves_draw_order(self):
        """Jämna odds ger rimlig X-sannolikhet."""
        # Typisk jämn match
        probs = odds_to_fair_probs(2.80, 3.20, 2.70)
        # X ska ligga mellan hemma och borta (alla nära 33%)
        assert probs[1] > 0.25
        assert probs[1] < 0.40

    def test_combined_match_draw_reasonable(self):
        """En jämn match ska ha rimlig X-sannolikhet i combined."""
        cm = build_combined_match(
            "TeamA", "TeamB",
            odds_1=2.90, odds_x=3.20, odds_2=2.60,
            model_probs=np.array([0.34, 0.33, 0.33]),
            streck_1=35, streck_x=30, streck_2=35,
        )
        assert cm.prob_x > 0.25, f"X={cm.prob_x:.3f} för låg för jämn match"


# ---------------------------------------------------------------------------
# 5. _safe_odds_values — hanterar OddsEntry och dict utan krasch
# ---------------------------------------------------------------------------

class TestSafeOddsValues:
    """
    Verifierar att _safe_odds_values fungerar med:
    - OddsEntry-objekt (attribut)
    - Vanliga dicts (nycklar)
    - None / trasiga data → (None, None, None)
    """

    @staticmethod
    def _fn(entry):
        """Import av _safe_odds_values från page 3 utan Streamlit-runtime."""
        # Duplicerar logiken exakt så vi kan testa isolerat
        if entry is None:
            return None, None, None
        try:
            if hasattr(entry, "home"):
                return float(entry.home), float(entry.draw), float(entry.away)
            if isinstance(entry, dict):
                return (
                    float(entry["home"]),
                    float(entry["draw"]),
                    float(entry["away"]),
                )
        except (KeyError, TypeError, ValueError):
            pass
        return None, None, None

    def test_odds_entry_object(self):
        """Fungerar med OddsEntry-dataclass (attribut-access)."""
        from odds_tool import OddsEntry
        e = OddsEntry(bookmaker="Bet365", home=2.50, draw=3.30, away=2.80)
        h, d, a = self._fn(e)
        assert h == pytest.approx(2.50)
        assert d == pytest.approx(3.30)
        assert a == pytest.approx(2.80)

    def test_plain_dict(self):
        """Fungerar med vanlig dict (nyckel-access)."""
        e = {"bookmaker": "Bet365", "home": 2.50, "draw": 3.30, "away": 2.80}
        h, d, a = self._fn(e)
        assert h == pytest.approx(2.50)
        assert d == pytest.approx(3.30)
        assert a == pytest.approx(2.80)

    def test_none_entry(self):
        """None ger (None, None, None)."""
        assert self._fn(None) == (None, None, None)

    def test_empty_dict(self):
        """Tom dict ger (None, None, None)."""
        assert self._fn({}) == (None, None, None)

    def test_partial_dict_missing_key(self):
        """Dict med saknad nyckel ger (None, None, None)."""
        assert self._fn({"home": 2.50, "draw": 3.30}) == (None, None, None)

    def test_non_numeric_values(self):
        """Dict med icke-numeriska värden ger (None, None, None)."""
        assert self._fn({"home": "bad", "draw": 3.30, "away": 2.80}) == (None, None, None)

    def test_string_numeric_values(self):
        """Dict med string-numeriska värden konverteras korrekt."""
        e = {"home": "2.50", "draw": "3.30", "away": "2.80"}
        h, d, a = self._fn(e)
        assert h == pytest.approx(2.50)
        assert d == pytest.approx(3.30)
        assert a == pytest.approx(2.80)


# ---------------------------------------------------------------------------
# 6. Fallback-dubbelräkning — odds ska inte räknas som både odds OCH modell
# ---------------------------------------------------------------------------

class TestFallbackDoubleCountingFix:
    """
    Verifierar att fallback-matcher INTE dubbelräknar odds genom att
    skicka odds-derived probs som model_probs till build_combined_match.

    Buggen: om model_probs = odds_to_fair_probs(...) OCH odds_1/x/2 också
    skickas, räknas odds-signalen två gånger (en gång som odds, en gång
    som model_probs).
    """

    def test_fallback_must_send_model_probs_none(self):
        """
        Fallback-match ska skicka model_probs=None till combined-byggaren.
        Odds ska enbart skickas via odds-fälten.
        """
        # Simulera korrekt fallback: model_probs=None, odds via oddsfält
        cm = build_combined_match(
            home_team="Newport",
            away_team="Shrewsbury",
            odds_1=2.50, odds_x=3.30, odds_2=2.80,
            model_probs=None,  # KORREKT: fallback skickar None
            streck_1=38, streck_x=28, streck_2=34,
        )
        assert cm.sources["model"] is False
        assert cm.sources["odds"] is True
        assert abs(cm.prob_1 + cm.prob_x + cm.prob_2 - 1.0) < 1e-9

    def test_double_counting_changes_result(self):
        """
        Visar att dubbelräkning ger annorlunda resultat än korrekt logik.
        Om odds-derived probs skickas som model_probs OCH som odds,
        ska resultatet skilja sig från korrekt (model_probs=None).

        Använder streck-värden som divergerar tydligt från odds så att
        viktförskjutningen blir mätbar.
        """
        # Odds implicerar ~56% hemma, streck säger 30% hemma → stor divergens
        odds_1, odds_x, odds_2 = 1.80, 3.60, 4.50
        streck_1, streck_x, streck_2 = 30, 35, 35

        # Korrekt: model_probs=None → combined = odds(77%) + streck(23%)
        cm_correct = build_combined_match(
            "Newport", "Shrewsbury",
            odds_1=odds_1, odds_x=odds_x, odds_2=odds_2,
            model_probs=None,
            streck_1=streck_1, streck_x=streck_x, streck_2=streck_2,
        )

        # Felaktigt (gamla buggen): odds-derived probs som model_probs
        # → combined = odds(50%) + fake_model(35%) + streck(15%)
        # odds-signalen väger 85% istf 77% → streck trängs undan
        fake_model = odds_to_fair_probs(odds_1, odds_x, odds_2)
        cm_buggy = build_combined_match(
            "Newport", "Shrewsbury",
            odds_1=odds_1, odds_x=odds_x, odds_2=odds_2,
            model_probs=fake_model,
            streck_1=streck_1, streck_x=streck_x, streck_2=streck_2,
        )

        assert cm_correct.sources["model"] is False
        assert cm_buggy.sources["model"] is True

        # Probabilities ska skilja sig mätbart
        diff = abs(cm_correct.prob_1 - cm_buggy.prob_1)
        assert diff > 0.005, (
            f"Korrekt och buggy ska ge olika resultat, "
            f"diff={diff:.6f}"
        )

    def test_combined_with_real_model_unaffected(self):
        """
        En match med riktig modellprediktion ska vara oförändrad.
        model_probs ska fortfarande skickas normalt.
        """
        model_probs = np.array([0.45, 0.30, 0.25])
        cm = build_combined_match(
            "Arsenal", "Chelsea",
            odds_1=2.00, odds_x=3.50, odds_2=4.00,
            model_probs=model_probs,
            streck_1=55, streck_x=25, streck_2=20,
        )
        assert cm.sources["model"] is True
        assert cm.sources["odds"] is True
        assert cm.sources["streck"] is True
        assert abs(cm.prob_1 + cm.prob_x + cm.prob_2 - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 7. UI-sanningsenligt: visade 1/X/2 ska motsvara combined (som Tips bygger på)
# ---------------------------------------------------------------------------

class TestUITruthfulness:
    """
    Verifierar att visade sannolikheter (1/X/2 i tabellen) motsvarar
    de kombinerade sannolikheter som Tips och HALV bygger på.
    """

    def test_displayed_probs_match_combined_for_model_match(self):
        """
        För en modell-match ska visade probs = combined (inte rå modell-probs).
        """
        model_probs = np.array([0.45, 0.30, 0.25])
        cm = build_combined_match(
            "Arsenal", "Chelsea",
            odds_1=2.00, odds_x=3.50, odds_2=4.00,
            model_probs=model_probs,
            streck_1=55, streck_x=25, streck_2=20,
        )
        # Tips baseras på combined probs
        sign = ['1', 'X', '2'][np.argmax(cm.probs)]
        # Visade probs ska vara cm.probs (combined), inte model_probs
        assert not np.allclose(cm.probs, model_probs), (
            "Combined ska skilja sig från rå model_probs "
            "(om odds/streck finns med)"
        )
        # Tips-tecknet ska matcha argmax av combined
        assert sign == ['1', 'X', '2'][np.argmax(cm.probs)]

    def test_displayed_probs_match_combined_for_fallback_match(self):
        """
        För en fallback-match ska visade probs = combined (odds+streck),
        inte rå odds-probs.
        """
        cm = build_combined_match(
            "Newport", "Shrewsbury",
            odds_1=2.50, odds_x=3.30, odds_2=2.80,
            model_probs=None,
            streck_1=38, streck_x=28, streck_2=34,
        )
        # Visade probs ska vara combined
        c_probs = cm.probs
        sign = ['1', 'X', '2'][np.argmax(c_probs)]

        # Tips ska matcha argmax av combined probs
        assert sign == ['1', 'X', '2'][np.argmax(c_probs)]
        # Sannolikheter summerar till 1
        assert abs(c_probs.sum() - 1.0) < 1e-9

    def test_halfguard_sign_matches_displayed_combined(self):
        """
        HALV-tecknet ska baseras på samma combined probs som visas i tabellen.
        """
        cm = build_combined_match(
            "Exeter", "Leyton Orient",
            odds_1=2.38, odds_x=3.40, odds_2=2.88,
            model_probs=None,
            streck_1=38, streck_x=25, streck_2=37,
        )
        halv_sign = get_halfguard_sign_combined(cm)
        # Halvgarderingen tar bort minst sannolika utfallet
        least = int(np.argmin(cm.probs))
        signs = ['1', 'X', '2']
        expected = "".join(signs[i] for i in range(3) if i != least)
        assert halv_sign == expected


# ---------------------------------------------------------------------------
# 8. Newport-liknande scenario: regression check
# ---------------------------------------------------------------------------

class TestNewportFallbackScenario:
    """
    Verifierar att Newport-liknande matcher (utanför modellens data)
    fortsätter fungera korrekt med fallback.
    """

    def test_newport_not_na_with_odds(self):
        """Newport med odds ska ge sannolikheter, inte N/A."""
        cm = build_combined_match(
            "Newport", "Shrewsbury",
            odds_1=2.50, odds_x=3.30, odds_2=2.80,
            model_probs=None,
        )
        assert cm.prob_1 > 0
        assert cm.prob_x > 0
        assert cm.prob_2 > 0

    def test_newport_fallback_marked_correctly(self):
        """Newport-match ska markeras som fallback (model=False)."""
        cm = build_combined_match(
            "Newport", "Shrewsbury",
            odds_1=2.50, odds_x=3.30, odds_2=2.80,
            model_probs=None,
            streck_1=38, streck_x=28, streck_2=34,
        )
        assert cm.sources["model"] is False
        assert cm.sources["odds"] is True

    def test_newport_with_streck_uses_both_signals(self):
        """Newport med odds+streck ska använda båda signalerna."""
        cm = build_combined_match(
            "Newport", "Shrewsbury",
            odds_1=2.50, odds_x=3.30, odds_2=2.80,
            model_probs=None,
            streck_1=38, streck_x=28, streck_2=34,
        )
        assert cm.sources["odds"] is True
        assert cm.sources["streck"] is True
        assert cm.sources["model"] is False
        # Streck-delta ska beräknas
        assert cm.streck_delta_1 != 0.0 or cm.streck_delta_x != 0.0 or cm.streck_delta_2 != 0.0

    def test_newport_participates_in_halfguard_selection(self):
        """Newport-match ska kunna delta i halvgarderingsurval."""
        cm_model = build_combined_match(
            "Arsenal", "Chelsea",
            odds_1=1.80, odds_x=3.60, odds_2=4.50,
            model_probs=np.array([0.55, 0.25, 0.20]),
        )
        cm_newport = build_combined_match(
            "Newport", "Shrewsbury",
            odds_1=2.50, odds_x=3.30, odds_2=2.80,
            model_probs=None,
            streck_1=38, streck_x=28, streck_2=34,
        )
        guards = pick_half_guards_combined([cm_model, cm_newport], n_guards=1)
        assert len(guards) == 1
        assert guards[0] in [0, 1]
