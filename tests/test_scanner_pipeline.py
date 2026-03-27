"""
Tester for scanner_pipeline.py -- den forbattrade kupongbild-scannern.

Testar:
1. Canonical team mapping / fuzzy matching
2. Validering av rad med rimliga odds/procent
3. Rad som ska markeras osaker vid dalig parsing
4. Scanner-output far ratt struktur aven nar falt saknas
5. Bildforbehandling (grundlaggande)
6. Faltvis parsning
7. Streck-odds-forvirring
8. Konvertering till bakatkompabla CouponRow/CouponExtractionResult
"""

import pytest
from scanner_pipeline import (
    ScannedRow,
    ScannerResult,
    preprocess_image,
    parse_row_fields,
    validate_rows,
    apply_team_mapping,
    map_team_canonical,
    _safe_float,
    _safe_odds,
    _safe_streck,
    _validate_odds_triplet,
    _check_streck_odds_confusion,
    _compute_row_confidence,
    _determine_row_status,
    _normalize_text,
    _fuzzy_match_team,
    _build_scanner_result,
    _parse_ai_response,
    scanned_rows_to_coupon_rows,
    scanner_result_to_extraction_result,
    SCANNER_TEAM_ALIASES,
)


# ---------------------------------------------------------------------------
# 1. Canonical team mapping / fuzzy matching
# ---------------------------------------------------------------------------

class TestCanonicalTeamMapping:
    """Tester for canonical team mapping och fuzzy-matchning."""

    def test_exact_alias_match(self):
        """Kanda alias ska matcha exakt."""
        name, conf, mapped = map_team_canonical("Man Utd")
        assert name == "Manchester United"
        assert conf == 1.0
        assert mapped is True

    def test_case_insensitive_alias(self):
        """Alias ska matchas case-insensitive."""
        name, conf, mapped = map_team_canonical("man utd")
        assert name == "Manchester United"
        assert mapped is True

    def test_wolves_alias(self):
        """Wolves ska matcha Wolverhampton Wanderers."""
        name, conf, mapped = map_team_canonical("Wolves")
        assert name == "Wolverhampton Wanderers"
        assert mapped is True

    def test_brighton_alias(self):
        """Brighton ska matcha Brighton & Hove Albion."""
        name, conf, mapped = map_team_canonical("Brighton")
        assert name == "Brighton & Hove Albion"
        assert mapped is True

    def test_qpr_alias(self):
        """QPR ska matcha Queens Park Rangers."""
        name, conf, mapped = map_team_canonical("QPR")
        assert name == "Queens Park Rangers"
        assert mapped is True

    def test_unknown_team_returns_original(self):
        """Okant lagnamn ska returnera originalet med lagre confidence."""
        name, conf, mapped = map_team_canonical("Nonexistent FC")
        assert conf < 1.0
        assert mapped is False

    def test_empty_name(self):
        """Tomt namn ska ge 0 confidence."""
        name, conf, mapped = map_team_canonical("")
        assert conf == 0.0
        assert mapped is False

    def test_fuzzy_match_close_name(self):
        """Nara namn ska fuzzy-matchas."""
        name, conf, mapped = map_team_canonical(
            "Manchster United",  # stavfel
            canonical_set={"Manchester United", "Manchester City"},
        )
        # Bor ha hittat nagon match
        assert mapped is True or conf > 0.5

    def test_fuzzy_match_with_canonical_set(self):
        """Fuzzy match mot canonical set."""
        canonical = {"Arsenal", "Liverpool", "Chelsea", "Manchester United"}
        name, conf, mapped = map_team_canonical("Arsenall", canonical)
        # Bor vara nara Arsenal
        assert conf > 0.5

    def test_multiple_aliases_for_same_team(self):
        """Flera alias for samma lag ska alla fungera."""
        for alias in ["Man United", "Man Utd", "ManUtd"]:
            name, conf, mapped = map_team_canonical(alias)
            assert name == "Manchester United", f"Failed for alias: {alias}"

    def test_nottingham_forest_variants(self):
        """Alla varianter av Nottingham Forest ska matchas."""
        for alias in ["Nott'm Forest", "Nottm Forest", "Nott Forest", "N Forest"]:
            name, conf, mapped = map_team_canonical(alias)
            assert name == "Nottingham Forest", f"Failed for alias: {alias}"

    def test_sheffield_teams_distinguished(self):
        """Sheffield United och Wednesday ska sarskiljas."""
        name1, _, _ = map_team_canonical("Sheff Utd")
        name2, _, _ = map_team_canonical("Sheff Wed")
        assert name1 == "Sheffield United"
        assert name2 == "Sheffield Wednesday"


class TestApplyTeamMapping:
    """Tester for apply_team_mapping pa rader."""

    def test_maps_known_aliases(self):
        """Kanda alias ska mappas i rader."""
        rows = [
            ScannedRow(home_team="Man Utd", away_team="Wolves",
                       home_team_raw="Man Utd", away_team_raw="Wolves"),
        ]
        result = apply_team_mapping(rows)
        assert result[0].home_team == "Manchester United"
        assert result[0].away_team == "Wolverhampton Wanderers"
        assert result[0].home_team_mapped is True
        assert result[0].away_team_mapped is True

    def test_preserves_raw_names(self):
        """Ra namn ska bevaras efter mapping."""
        rows = [
            ScannedRow(home_team="Man City", away_team="Chelsea",
                       home_team_raw="Man City", away_team_raw="Chelsea"),
        ]
        result = apply_team_mapping(rows)
        assert result[0].home_team_raw == "Man City"


# ---------------------------------------------------------------------------
# 2. Validering av rad med rimliga odds/procent
# ---------------------------------------------------------------------------

class TestRowValidation:
    """Tester for radvalidering med rimliga odds/streck."""

    def test_valid_complete_row(self):
        """Komplett giltig rad ska ha hog confidence."""
        row = ScannedRow(
            home_team="Arsenal", away_team="Liverpool",
            streck_1=45.0, streck_x=28.0, streck_2=27.0,
            odds_1=2.10, odds_x=3.40, odds_2=3.60,
            home_team_confidence=1.0, away_team_confidence=1.0,
            streck_confidence=1.0, odds_confidence=1.0,
            confidence_score=0.9,
        )
        validated = validate_rows([row])
        assert validated[0].row_status == "ok"
        assert validated[0].confidence_score > 0.5

    def test_unreasonable_streck_sum_flagged(self):
        """Strecksumma som avviker kraftigt ska flaggas."""
        row = ScannedRow(
            home_team="A", away_team="B",
            streck_1=30.0, streck_x=20.0, streck_2=10.0,  # sum = 60
            odds_1=2.0, odds_x=3.0, odds_2=4.0,
            confidence_score=0.9,
        )
        validated = validate_rows([row])
        assert any("strecksumma" in i.lower() or "60.0" in i for i in validated[0].issues)

    def test_unreasonable_odds_flagged(self):
        """Orimligt hoga odds ska flaggas."""
        row = ScannedRow(
            home_team="A", away_team="B",
            odds_1=200.0, odds_x=3.0, odds_2=4.0,
            confidence_score=0.9,
        )
        validated = validate_rows([row])
        assert any("orimligt" in i.lower() or "200" in str(i) for i in validated[0].issues)

    def test_identical_teams_flagged(self):
        """Identiska hemma- och bortalag ska flaggas."""
        row = ScannedRow(
            home_team="Arsenal", away_team="Arsenal",
            confidence_score=0.9,
        )
        validated = validate_rows([row])
        assert any("identiska" in i.lower() for i in validated[0].issues)

    def test_overround_validation(self):
        """Overround utanfor normalt intervall ska flaggas."""
        # Normalt overround ~1.05-1.15
        row = ScannedRow(
            home_team="A", away_team="B",
            odds_1=1.2, odds_x=1.2, odds_2=1.2,  # overround = 2.5
            confidence_score=0.9,
        )
        validated = validate_rows([row])
        assert any("overround" in i.lower() for i in validated[0].issues)

    def test_valid_streck_sum_ok(self):
        """Strecksumma nara 100% ska inte ge problem."""
        row = ScannedRow(
            home_team="A", away_team="B",
            streck_1=45.0, streck_x=28.0, streck_2=27.0,  # sum = 100
            confidence_score=0.9,
        )
        validated = validate_rows([row])
        # Inga streck-relaterade issues
        streck_issues = [i for i in validated[0].issues if "streck" in i.lower()]
        assert len(streck_issues) == 0

    def test_zero_streck_flagged(self):
        """Alla streckvarden 0 ska flaggas."""
        row = ScannedRow(
            home_team="A", away_team="B",
            streck_1=0, streck_x=0, streck_2=0,
            confidence_score=0.9,
        )
        validated = validate_rows([row])
        assert any("0" in i for i in validated[0].issues)


# ---------------------------------------------------------------------------
# 3. Rad som ska markeras osaker vid dalig parsing
# ---------------------------------------------------------------------------

class TestUncertainRowDetection:
    """Tester for att osakra rader markeras korrekt."""

    def test_missing_teams_gives_failed(self):
        """Rad utan lagnamn ska vara failed."""
        item = {"HomeTeam": "", "AwayTeam": "", "Streck1": 45}
        row = parse_row_fields(item)
        # Without teams, should have very low confidence
        assert row.home_team_confidence == 0.0
        assert row.away_team_confidence == 0.0

    def test_partial_streck_gives_low_confidence(self):
        """Ofullstandiga streckvarden ska ge lag confidence."""
        item = {
            "HomeTeam": "A", "AwayTeam": "B",
            "Streck1": 45, "StreckX": None, "Streck2": 27,
        }
        row = parse_row_fields(item)
        assert row.streck_confidence < 1.0
        assert any("ofullstandiga" in i.lower() for i in row.issues)

    def test_partial_odds_gives_low_confidence(self):
        """Ofullstandiga odds ska ge lag confidence."""
        item = {
            "HomeTeam": "A", "AwayTeam": "B",
            "Odds1": 2.10, "OddsX": None, "Odds2": 3.60,
        }
        row = parse_row_fields(item)
        assert row.odds_confidence < 1.0

    def test_bad_streck_sum_gives_uncertain(self):
        """Streck som inte summerar till ~100 ska ge uncertain."""
        item = {
            "HomeTeam": "A", "AwayTeam": "B",
            "Streck1": 30, "StreckX": 20, "Streck2": 10,
            "Odds1": 2.0, "OddsX": 3.0, "Odds2": 4.0,
        }
        row = parse_row_fields(item)
        assert row.streck_confidence < 1.0

    def test_no_data_gives_failed(self):
        """Rad utan nagon data ska vara failed."""
        item = {"HomeTeam": "", "AwayTeam": ""}
        row = parse_row_fields(item)
        assert row.confidence_score < 0.4

    def test_api_low_confidence_reduces_score(self):
        """API-rapporterad lag confidence ska sanka scoren."""
        item_high = {
            "HomeTeam": "A", "AwayTeam": "B",
            "Streck1": 45, "StreckX": 28, "Streck2": 27,
            "Odds1": 2.10, "OddsX": 3.40, "Odds2": 3.60,
            "row_confidence": "high",
        }
        item_low = {
            "HomeTeam": "A", "AwayTeam": "B",
            "Streck1": 45, "StreckX": 28, "Streck2": 27,
            "Odds1": 2.10, "OddsX": 3.40, "Odds2": 3.60,
            "row_confidence": "low",
        }
        row_high = parse_row_fields(item_high)
        row_low = parse_row_fields(item_low)
        assert row_high.confidence_score > row_low.confidence_score


# ---------------------------------------------------------------------------
# 4. Scanner-output far ratt struktur aven nar falt saknas
# ---------------------------------------------------------------------------

class TestOutputStructure:
    """Tester for att output har korrekt struktur."""

    def test_complete_row_structure(self):
        """Komplett rad ska ha alla falt."""
        item = {
            "HomeTeam": "Arsenal", "AwayTeam": "Liverpool",
            "Streck1": 45, "StreckX": 28, "Streck2": 27,
            "Odds1": 2.10, "OddsX": 3.40, "Odds2": 3.60,
        }
        row = parse_row_fields(item)
        assert row.home_team == "Arsenal"
        assert row.away_team == "Liverpool"
        assert row.streck_1 == 45.0
        assert row.streck_x == 28.0
        assert row.streck_2 == 27.0
        assert row.odds_1 == 2.10
        assert row.odds_x == 3.40
        assert row.odds_2 == 3.60
        assert row.row_status in ("ok", "uncertain", "failed")
        assert isinstance(row.issues, list)

    def test_partial_row_structure(self):
        """Rad med saknade falt ska anda ha korrekt struktur."""
        item = {
            "HomeTeam": "Arsenal", "AwayTeam": "Liverpool",
            "Streck1": None, "StreckX": None, "Streck2": None,
            "Odds1": None, "OddsX": None, "Odds2": None,
        }
        row = parse_row_fields(item)
        assert row.home_team == "Arsenal"
        assert row.streck_1 is None
        assert row.odds_1 is None
        assert row.row_status in ("ok", "uncertain", "failed")

    def test_scanner_result_structure(self):
        """ScannerResult ska ha korrekt struktur."""
        rows = [
            ScannedRow(home_team="A", away_team="B", row_status="ok"),
            ScannedRow(home_team="C", away_team="D", row_status="uncertain"),
            ScannedRow(home_team="E", away_team="F", row_status="failed"),
        ]
        result = _build_scanner_result(rows, "raw", ["step1"])
        assert result.total_rows == 3
        assert result.ok_rows == 1
        assert result.uncertain_rows == 1
        assert result.failed_rows == 1
        assert result.preprocessing_applied == ["step1"]

    def test_empty_result_structure(self):
        """Tomt resultat ska ha korrekt struktur."""
        result = _build_scanner_result([], "", [])
        assert result.total_rows == 0
        assert result.ok_rows == 0
        assert result.error is None

    def test_row_with_only_teams(self):
        """Rad med bara lag ska ha confidence < 1."""
        item = {"HomeTeam": "A", "AwayTeam": "B"}
        row = parse_row_fields(item)
        assert row.home_team == "A"
        assert row.away_team == "B"
        assert row.streck_1 is None
        assert row.odds_1 is None
        assert row.row_status in ("uncertain", "failed")


# ---------------------------------------------------------------------------
# 5. Bildforbehandling
# ---------------------------------------------------------------------------

class TestImagePreprocessing:
    """Tester for bildforbehandling."""

    def test_preprocess_valid_image(self):
        """Giltig bild ska forbehandlas utan fel."""
        # Skapa en enkel testbild
        try:
            from PIL import Image
            import io
            img = Image.new("RGB", (200, 100), color="white")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            processed, steps = preprocess_image(image_bytes)
            assert isinstance(processed, bytes)
            assert len(processed) > 0
            assert len(steps) > 0
            assert "upscale" in " ".join(steps).lower() or "autocontrast" in " ".join(steps).lower()
        except ImportError:
            pytest.skip("Pillow not available")

    def test_preprocess_small_image_upscaled(self):
        """Liten bild ska skalas upp."""
        try:
            from PIL import Image
            import io
            img = Image.new("RGB", (100, 50), color="white")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            processed, steps = preprocess_image(image_bytes)
            assert any("upscale" in s for s in steps)
        except ImportError:
            pytest.skip("Pillow not available")

    def test_preprocess_large_image_not_upscaled(self):
        """Stor bild ska inte skalas upp."""
        try:
            from PIL import Image
            import io
            img = Image.new("RGB", (2000, 1000), color="white")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            processed, steps = preprocess_image(image_bytes)
            assert not any("upscale" in s for s in steps)
        except ImportError:
            pytest.skip("Pillow not available")

    def test_preprocess_invalid_data(self):
        """Ogiltig bilddata ska hanteras defensivt."""
        processed, steps = preprocess_image(b"not an image")
        assert isinstance(processed, bytes)
        assert any("skip" in s for s in steps)


# ---------------------------------------------------------------------------
# 6. Safe-funktioner och faltparsning
# ---------------------------------------------------------------------------

class TestSafeFunctions:
    """Tester for _safe_float, _safe_odds, _safe_streck."""

    def test_safe_float_valid(self):
        assert _safe_float(3.14) == 3.14
        assert _safe_float("2.5") == 2.5
        assert _safe_float(0) == 0.0

    def test_safe_float_invalid(self):
        assert _safe_float(None) is None
        assert _safe_float("abc") is None
        assert _safe_float(-5) is None

    def test_safe_odds_valid(self):
        assert _safe_odds(2.10) == 2.10
        assert _safe_odds("3.50") == 3.50

    def test_safe_odds_too_low(self):
        assert _safe_odds(1.0) is None
        assert _safe_odds(0.5) is None

    def test_safe_streck_valid(self):
        assert _safe_streck(45.0) == 45.0
        assert _safe_streck(0) == 0.0
        assert _safe_streck(100) == 100.0

    def test_safe_streck_out_of_range(self):
        assert _safe_streck(-5) is None
        assert _safe_streck(150) is None
        assert _safe_streck(None) is None


# ---------------------------------------------------------------------------
# 7. Odds-validering
# ---------------------------------------------------------------------------

class TestOddsValidation:
    """Tester for odds-triplettvalidering."""

    def test_valid_odds_triplet(self):
        """Rimliga odds ska ge hog confidence."""
        issues = []
        conf = _validate_odds_triplet(2.10, 3.40, 3.60, issues)
        assert conf == 1.0
        assert len(issues) == 0

    def test_unreasonable_overround(self):
        """Orimlig overround ska ge lag confidence."""
        issues = []
        conf = _validate_odds_triplet(1.01, 1.01, 1.01, issues)
        assert conf < 1.0

    def test_moderate_overround(self):
        """Moderat overround ska ge ok confidence."""
        issues = []
        # 1/1.5 + 1/3.5 + 1/4.0 = 0.667 + 0.286 + 0.25 = 1.203
        conf = _validate_odds_triplet(1.50, 3.50, 4.00, issues)
        assert conf == 1.0


# ---------------------------------------------------------------------------
# 8. Streck-odds-forvirring
# ---------------------------------------------------------------------------

class TestStreckOddsConfusion:
    """Tester for detektion av forvirring mellan streck och odds."""

    def test_streck_looks_like_odds(self):
        """Streckvarden som ser ut som odds ska flaggas."""
        row = ScannedRow(
            home_team="A", away_team="B",
            streck_1=2.10, streck_x=3.40, streck_2=3.60,  # Ser ut som odds
            streck_confidence=1.0,
        )
        _check_streck_odds_confusion(row)
        assert row.streck_confidence < 1.0
        assert any("forvirr" in i.lower() for i in row.issues)

    def test_odds_looks_like_streck(self):
        """Oddsvarden som ser ut som procent ska flaggas."""
        row = ScannedRow(
            home_team="A", away_team="B",
            odds_1=45.0, odds_x=28.0, odds_2=27.0,  # Ser ut som streck
            odds_confidence=1.0,
        )
        _check_streck_odds_confusion(row)
        assert row.odds_confidence < 1.0
        assert any("forvirr" in i.lower() for i in row.issues)

    def test_normal_values_not_flagged(self):
        """Normala varden ska inte flaggas."""
        row = ScannedRow(
            home_team="A", away_team="B",
            streck_1=45, streck_x=28, streck_2=27,
            odds_1=2.10, odds_x=3.40, odds_2=3.60,
            streck_confidence=1.0, odds_confidence=1.0,
        )
        _check_streck_odds_confusion(row)
        confusion_issues = [i for i in row.issues if "forvirr" in i.lower()]
        assert len(confusion_issues) == 0


# ---------------------------------------------------------------------------
# 9. Confidence och status
# ---------------------------------------------------------------------------

class TestConfidenceAndStatus:
    """Tester for confidence-berakning och statusbestamning."""

    def test_high_confidence_gives_ok(self):
        """Hog confidence ska ge status 'ok'."""
        row = ScannedRow(confidence_score=0.85)
        assert _determine_row_status(row) == "ok"

    def test_medium_confidence_gives_uncertain(self):
        """Medel confidence ska ge status 'uncertain'."""
        row = ScannedRow(confidence_score=0.55)
        assert _determine_row_status(row) == "uncertain"

    def test_low_confidence_gives_failed(self):
        """Lag confidence ska ge status 'failed'."""
        row = ScannedRow(confidence_score=0.2)
        assert _determine_row_status(row) == "failed"

    def test_compute_confidence_with_all_fields(self):
        """Komplett rad ska ge hog confidence."""
        row = ScannedRow(
            home_team_confidence=1.0,
            away_team_confidence=1.0,
            streck_confidence=1.0,
            odds_confidence=1.0,
        )
        conf = _compute_row_confidence(row, 1.0)
        assert conf > 0.8

    def test_compute_confidence_with_missing_fields(self):
        """Rad med saknade falt ska ge lagre confidence."""
        row = ScannedRow(
            home_team_confidence=1.0,
            away_team_confidence=1.0,
            streck_confidence=0.0,
            odds_confidence=0.0,
        )
        conf = _compute_row_confidence(row, 1.0)
        assert conf < 0.8


# ---------------------------------------------------------------------------
# 10. AI-svar-parsning
# ---------------------------------------------------------------------------

class TestParseAiResponse:
    """Tester for parsning av AI-svar."""

    def test_valid_json_response(self):
        """Giltigt JSON ska parseas korrekt."""
        raw = '''[
            {"HomeTeam": "Arsenal", "AwayTeam": "Liverpool",
             "Streck1": 45, "StreckX": 28, "Streck2": 27,
             "Odds1": 2.10, "OddsX": 3.40, "Odds2": 3.60,
             "row_confidence": "high", "field_issues": []}
        ]'''
        rows, error = _parse_ai_response(raw)
        assert error is None
        assert len(rows) == 1
        assert rows[0].home_team == "Arsenal"
        assert rows[0].odds_1 == 2.10

    def test_json_in_markdown(self):
        """JSON i markdown code block ska parseas."""
        raw = '```json\n[{"HomeTeam":"A","AwayTeam":"B"}]\n```'
        rows, error = _parse_ai_response(raw)
        assert error is None
        assert len(rows) == 1

    def test_invalid_json_gives_error(self):
        """Ogiltig JSON ska ge felmeddelande."""
        rows, error = _parse_ai_response("not json at all")
        assert error is not None
        assert len(rows) == 0

    def test_empty_teams_filtered(self):
        """Rader med tomma lag ska filtreras bort."""
        raw = '[{"HomeTeam":"","AwayTeam":""}]'
        rows, error = _parse_ai_response(raw)
        assert error is None
        assert len(rows) == 0

    def test_multiple_rows_parsed(self):
        """Flera rader ska parseas korrekt."""
        raw = '''[
            {"HomeTeam": "A", "AwayTeam": "B", "Streck1": 45, "StreckX": 28, "Streck2": 27},
            {"HomeTeam": "C", "AwayTeam": "D", "Odds1": 2.0, "OddsX": 3.5, "Odds2": 3.5}
        ]'''
        rows, error = _parse_ai_response(raw)
        assert error is None
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# 11. Konvertering till bakatkompabla typer
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Tester for konvertering till CouponRow/CouponExtractionResult."""

    def test_scanned_to_coupon_rows(self):
        """ScannedRow ska konverteras till CouponRow."""
        rows = [
            ScannedRow(
                home_team="Arsenal", away_team="Liverpool",
                streck_1=45, streck_x=28, streck_2=27,
                odds_1=2.10, odds_x=3.40, odds_2=3.60,
                row_status="ok", issues=["test issue"],
            ),
        ]
        coupon_rows = scanned_rows_to_coupon_rows(rows)
        assert len(coupon_rows) == 1
        assert coupon_rows[0].home_team == "Arsenal"
        assert coupon_rows[0].confidence == "ok"
        assert "test issue" in coupon_rows[0].notes

    def test_uncertain_maps_to_uncertain(self):
        """uncertain status ska mappas till confidence='uncertain'."""
        rows = [ScannedRow(home_team="A", away_team="B", row_status="uncertain")]
        coupon_rows = scanned_rows_to_coupon_rows(rows)
        assert coupon_rows[0].confidence == "uncertain"

    def test_failed_maps_to_incomplete(self):
        """failed status ska mappas till confidence='incomplete'."""
        rows = [ScannedRow(home_team="A", away_team="B", row_status="failed")]
        coupon_rows = scanned_rows_to_coupon_rows(rows)
        assert coupon_rows[0].confidence == "incomplete"

    def test_scanner_result_to_extraction_result(self):
        """ScannerResult ska konverteras till CouponExtractionResult."""
        scanner_result = ScannerResult(
            rows=[
                ScannedRow(home_team="A", away_team="B", row_status="ok"),
                ScannedRow(home_team="C", away_team="D", row_status="uncertain"),
            ],
            total_rows=2,
            ok_rows=1,
            uncertain_rows=1,
            failed_rows=0,
            raw_response="test",
        )
        extraction = scanner_result_to_extraction_result(scanner_result)
        assert extraction.total_rows == 2
        assert extraction.complete_rows == 1
        assert extraction.uncertain_rows == 1
        assert len(extraction.rows) == 2


# ---------------------------------------------------------------------------
# 12. Normalize text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    """Tester for textnormalisering."""

    def test_normalize_dashes(self):
        """Olika typer av bindestreck ska normaliseras."""
        assert _normalize_text("Sheffield\u2013United") == "Sheffield-United"
        assert _normalize_text("Sheffield\u2014United") == "Sheffield-United"

    def test_normalize_dots(self):
        """Punkter ska tas bort."""
        assert _normalize_text("Nott.m Forest") == "Nottm Forest"

    def test_normalize_whitespace(self):
        """Extra mellanslag ska normaliseras."""
        assert _normalize_text("  Man   Utd  ") == "Man Utd"


# ---------------------------------------------------------------------------
# 13. Fuzzy match
# ---------------------------------------------------------------------------

class TestFuzzyMatch:
    """Tester for fuzzy-matchning."""

    def test_exact_match_in_canonical(self):
        """Exakt match i canonical set."""
        match, score = _fuzzy_match_team("Arsenal", {"Arsenal", "Chelsea"})
        assert match == "Arsenal"
        assert score > 0.9

    def test_close_match(self):
        """Nara match ska ge hog score."""
        match, score = _fuzzy_match_team("Arsenall", {"Arsenal", "Chelsea"})
        assert score > 0.7

    def test_no_good_match(self):
        """Ingen bra match ska ge lag score."""
        match, score = _fuzzy_match_team("XXXXXXXXX", {"Arsenal", "Chelsea"})
        assert score < 0.5
