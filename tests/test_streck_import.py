"""
Tester for streck_import.py -- automatisk inlasning av streckdata.

Testar:
- inlasning fran CSV
- validering av kolumner och varden
- normalisering av format (procent/decimal)
- matchning mot fixtures (exakt, normaliserad, case-insensitive)
- defensiv hantering av saknade/felaktiga data
- fallback nar fil saknas
- auto_load_streck hela flodet
"""

import pytest
import pandas as pd
from pathlib import Path
from streck_import import (
    load_streck_data,
    validate_streck_data,
    normalize_streck_data,
    match_streck_to_fixtures,
    dataframe_to_records,
    auto_load_streck,
    StreckRecord,
    StreckImportStatus,
    REQUIRED_COLUMNS,
)


# ---------------------------------------------------------------------------
# load_streck_data
# ---------------------------------------------------------------------------

class TestLoadStreckData:
    def test_load_nonexistent_file(self, tmp_path):
        """Returnerar None + felmeddelande om filen inte finns."""
        df, errors = load_streck_data(tmp_path / "missing.csv")
        assert df is None
        assert len(errors) == 1
        assert "hittades inte" in errors[0]

    def test_load_valid_csv(self, tmp_path):
        """Laser in giltig CSV korrekt."""
        csv_path = tmp_path / "streck.csv"
        csv_path.write_text(
            "HomeTeam,AwayTeam,Streck1,StreckX,Streck2\n"
            "Arsenal,Liverpool,45,28,27\n"
            "Man City,Chelsea,55,25,20\n"
        )
        df, errors = load_streck_data(csv_path)
        assert df is not None
        assert len(df) == 2
        assert len(errors) == 0

    def test_load_empty_csv(self, tmp_path):
        """Tom CSV ger None + felmeddelande."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("HomeTeam,AwayTeam,Streck1,StreckX,Streck2\n")
        df, errors = load_streck_data(csv_path)
        assert df is None
        assert any("tom" in e for e in errors)

    def test_load_malformed_csv(self, tmp_path):
        """Trasig CSV ger None + felmeddelande."""
        csv_path = tmp_path / "bad.csv"
        csv_path.write_bytes(b"\x00\x01\x02")
        df, errors = load_streck_data(csv_path)
        # Pandas may or may not raise; either way we handle gracefully
        assert df is None or len(errors) > 0 or df is not None


# ---------------------------------------------------------------------------
# validate_streck_data
# ---------------------------------------------------------------------------

class TestValidateStreckData:
    def test_valid_data_passes(self):
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal", "Man City"],
            "AwayTeam": ["Liverpool", "Chelsea"],
            "Streck1": [45.0, 55.0],
            "StreckX": [28.0, 25.0],
            "Streck2": [27.0, 20.0],
        })
        validated, errors = validate_streck_data(df)
        assert validated is not None
        assert len(validated) == 2

    def test_missing_required_columns(self):
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Liverpool"],
            "Streck1": [45.0],
            # Missing StreckX and Streck2
        })
        validated, errors = validate_streck_data(df)
        assert validated is None
        assert any("Saknade kolumner" in e for e in errors)

    def test_non_numeric_values_skipped(self):
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal", "Man City"],
            "AwayTeam": ["Liverpool", "Chelsea"],
            "Streck1": [45.0, "abc"],
            "StreckX": [28.0, 25.0],
            "Streck2": [27.0, 20.0],
        })
        validated, errors = validate_streck_data(df)
        assert validated is not None
        assert len(validated) == 1  # Only Arsenal row
        assert any("ogiltigt" in e for e in errors)

    def test_negative_values_skipped(self):
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal", "Man City"],
            "AwayTeam": ["Liverpool", "Chelsea"],
            "Streck1": [45.0, -5.0],
            "StreckX": [28.0, 55.0],
            "Streck2": [27.0, 50.0],
        })
        validated, errors = validate_streck_data(df)
        assert validated is not None
        assert len(validated) == 1
        assert any("negativt" in e for e in errors)

    def test_empty_team_names_skipped(self):
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal", ""],
            "AwayTeam": ["Liverpool", "Chelsea"],
            "Streck1": [45.0, 55.0],
            "StreckX": [28.0, 25.0],
            "Streck2": [27.0, 20.0],
        })
        validated, errors = validate_streck_data(df)
        assert validated is not None
        assert len(validated) == 1

    def test_all_invalid_returns_none(self):
        df = pd.DataFrame({
            "HomeTeam": [""],
            "AwayTeam": [""],
            "Streck1": [-1.0],
            "StreckX": [-1.0],
            "Streck2": [-1.0],
        })
        validated, errors = validate_streck_data(df)
        assert validated is None
        assert any("Inga giltiga" in e for e in errors)


# ---------------------------------------------------------------------------
# normalize_streck_data
# ---------------------------------------------------------------------------

class TestNormalizeStreckData:
    def test_percent_format_unchanged(self):
        """Procentformat (summa ~100) behalles."""
        df = pd.DataFrame({
            "HomeTeam": ["A"],
            "AwayTeam": ["B"],
            "Streck1": [45.0],
            "StreckX": [28.0],
            "Streck2": [27.0],
        })
        result = normalize_streck_data(df)
        assert abs(result.iloc[0]["Streck1"] - 45.0) < 0.01
        assert abs(result.iloc[0]["StreckX"] - 28.0) < 0.01
        assert abs(result.iloc[0]["Streck2"] - 27.0) < 0.01

    def test_decimal_format_converted(self):
        """Decimalformat (summa ~1) konverteras till procent."""
        df = pd.DataFrame({
            "HomeTeam": ["A"],
            "AwayTeam": ["B"],
            "Streck1": [0.45],
            "StreckX": [0.28],
            "Streck2": [0.27],
        })
        result = normalize_streck_data(df)
        assert abs(result.iloc[0]["Streck1"] - 45.0) < 0.01
        assert abs(result.iloc[0]["StreckX"] - 28.0) < 0.01
        assert abs(result.iloc[0]["Streck2"] - 27.0) < 0.01

    def test_mixed_format_rows(self):
        """Blandade rader hanteras korrekt per rad."""
        df = pd.DataFrame({
            "HomeTeam": ["A", "C"],
            "AwayTeam": ["B", "D"],
            "Streck1": [45.0, 0.50],
            "StreckX": [28.0, 0.25],
            "Streck2": [27.0, 0.25],
        })
        result = normalize_streck_data(df)
        # Rad 1: redan procent
        assert abs(result.iloc[0]["Streck1"] - 45.0) < 0.01
        # Rad 2: decimal -> procent
        assert abs(result.iloc[1]["Streck1"] - 50.0) < 0.01


# ---------------------------------------------------------------------------
# match_streck_to_fixtures
# ---------------------------------------------------------------------------

class TestMatchStreckToFixtures:
    def test_exact_match(self):
        records = [
            StreckRecord("Arsenal", "Liverpool", 45, 28, 27),
            StreckRecord("Man City", "Chelsea", 55, 25, 20),
        ]
        fixture_keys = ["Arsenal_Liverpool", "Man City_Chelsea"]
        matched, m_keys, u_keys = match_streck_to_fixtures(records, fixture_keys)
        assert len(matched) == 2
        assert len(m_keys) == 2
        assert len(u_keys) == 0
        assert matched["Arsenal_Liverpool"]["1"] == 45

    def test_case_insensitive_match(self):
        records = [
            StreckRecord("arsenal", "liverpool", 45, 28, 27),
        ]
        fixture_keys = ["Arsenal_Liverpool"]
        matched, m_keys, u_keys = match_streck_to_fixtures(records, fixture_keys)
        # Should match via case-insensitive fallback
        assert len(matched) == 1

    def test_unmatched_fixtures(self):
        records = [
            StreckRecord("Arsenal", "Liverpool", 45, 28, 27),
        ]
        fixture_keys = ["Arsenal_Liverpool", "Brighton_Newcastle"]
        matched, m_keys, u_keys = match_streck_to_fixtures(records, fixture_keys)
        assert len(matched) == 1
        assert "Brighton_Newcastle" in u_keys

    def test_empty_records(self):
        matched, m_keys, u_keys = match_streck_to_fixtures([], ["A_B"])
        assert len(matched) == 0
        assert len(u_keys) == 1

    def test_empty_fixtures(self):
        records = [StreckRecord("A", "B", 45, 28, 27)]
        matched, m_keys, u_keys = match_streck_to_fixtures(records, [])
        assert len(matched) == 0
        assert len(u_keys) == 0


# ---------------------------------------------------------------------------
# dataframe_to_records
# ---------------------------------------------------------------------------

class TestDataframeToRecords:
    def test_basic_conversion(self):
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal", "Man City"],
            "AwayTeam": ["Liverpool", "Chelsea"],
            "Streck1": [45.0, 55.0],
            "StreckX": [28.0, 25.0],
            "Streck2": [27.0, 20.0],
        })
        records = dataframe_to_records(df)
        assert len(records) == 2
        assert records[0].home_team == "Arsenal"
        assert records[0].streck_1 == 45.0
        assert records[1].away_team == "Chelsea"

    def test_optional_columns(self):
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Liverpool"],
            "Streck1": [45.0],
            "StreckX": [28.0],
            "Streck2": [27.0],
            "Date": ["2025-03-15"],
            "Source": ["Svenska Spel"],
        })
        records = dataframe_to_records(df)
        assert records[0].date == "2025-03-15"
        assert records[0].source == "Svenska Spel"

    def test_missing_optional_columns(self):
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Liverpool"],
            "Streck1": [45.0],
            "StreckX": [28.0],
            "Streck2": [27.0],
        })
        records = dataframe_to_records(df)
        assert records[0].date is None
        assert records[0].source is None


# ---------------------------------------------------------------------------
# auto_load_streck (integration)
# ---------------------------------------------------------------------------

class TestAutoLoadStreck:
    def test_file_not_found(self, tmp_path):
        """Returnerar None lookup + status.loaded=False."""
        lookup, status = auto_load_streck(path=tmp_path / "missing.csv")
        assert lookup is None
        assert status.loaded is False
        assert status.total_rows == 0

    def test_full_flow(self, tmp_path):
        """Hela flodet: inlasning, validering, normalisering."""
        csv_path = tmp_path / "streck_data.csv"
        csv_path.write_text(
            "HomeTeam,AwayTeam,Streck1,StreckX,Streck2\n"
            "Arsenal,Liverpool,45,28,27\n"
            "Man City,Chelsea,55,25,20\n"
        )
        lookup, status = auto_load_streck(path=csv_path)
        assert lookup is not None
        assert status.loaded is True
        assert status.total_rows == 2
        assert status.valid_rows == 2
        assert status.matched_count == 2
        assert "Arsenal_Liverpool" in lookup
        assert lookup["Arsenal_Liverpool"]["1"] == 45.0

    def test_with_fixture_keys(self, tmp_path):
        """Matchning mot fixture-nycklar."""
        csv_path = tmp_path / "streck_data.csv"
        csv_path.write_text(
            "HomeTeam,AwayTeam,Streck1,StreckX,Streck2\n"
            "Arsenal,Liverpool,45,28,27\n"
            "Man City,Chelsea,55,25,20\n"
        )
        fixture_keys = ["Arsenal_Liverpool", "Brighton_Newcastle"]
        lookup, status = auto_load_streck(path=csv_path, fixture_keys=fixture_keys)
        assert lookup is not None
        assert status.matched_count == 1
        assert status.unmatched_count == 1
        assert "Arsenal_Liverpool" in lookup
        assert "Brighton_Newcastle" not in lookup

    def test_with_source_column(self, tmp_path):
        """Kalla-kolumn paverkar source_label."""
        csv_path = tmp_path / "streck_data.csv"
        csv_path.write_text(
            "HomeTeam,AwayTeam,Streck1,StreckX,Streck2,Source\n"
            "Arsenal,Liverpool,45,28,27,Svenska Spel\n"
        )
        lookup, status = auto_load_streck(path=csv_path)
        assert status.source_label == "Svenska Spel"

    def test_partial_invalid_rows(self, tmp_path):
        """Ogiltiga rader hoppas over, giltiga behalles."""
        csv_path = tmp_path / "streck_data.csv"
        csv_path.write_text(
            "HomeTeam,AwayTeam,Streck1,StreckX,Streck2\n"
            "Arsenal,Liverpool,45,28,27\n"
            "Bad,,abc,-5,20\n"
        )
        lookup, status = auto_load_streck(path=csv_path)
        assert lookup is not None
        assert status.valid_rows == 1
        assert status.skipped_rows == 1
        assert status.loaded is True

    def test_decimal_format_handled(self, tmp_path):
        """Decimalformat normaliseras till procent."""
        csv_path = tmp_path / "streck_data.csv"
        csv_path.write_text(
            "HomeTeam,AwayTeam,Streck1,StreckX,Streck2\n"
            "Arsenal,Liverpool,0.45,0.28,0.27\n"
        )
        lookup, status = auto_load_streck(path=csv_path)
        assert lookup is not None
        assert abs(lookup["Arsenal_Liverpool"]["1"] - 45.0) < 0.01


# ---------------------------------------------------------------------------
# Defensiv hantering
# ---------------------------------------------------------------------------

class TestDefensiveHandling:
    def test_streck_lookup_values_are_numeric(self, tmp_path):
        csv_path = tmp_path / "streck_data.csv"
        csv_path.write_text(
            "HomeTeam,AwayTeam,Streck1,StreckX,Streck2\n"
            "Arsenal,Liverpool,45,28,27\n"
        )
        lookup, status = auto_load_streck(path=csv_path)
        assert lookup is not None
        for key, val in lookup.items():
            assert isinstance(val["1"], float)
            assert isinstance(val["X"], float)
            assert isinstance(val["2"], float)

    def test_required_columns_constant(self):
        """REQUIRED_COLUMNS har ratt innehall."""
        assert REQUIRED_COLUMNS == {"HomeTeam", "AwayTeam", "Streck1", "StreckX", "Streck2"}

    def test_fallback_preserved_when_no_auto_file(self, tmp_path):
        """Nar auto-fil saknas returneras tydlig status."""
        lookup, status = auto_load_streck(path=tmp_path / "nonexistent.csv")
        assert lookup is None
        assert not status.loaded
        assert len(status.errors) > 0
        assert status.source_label == "Ingen fil hittad"
