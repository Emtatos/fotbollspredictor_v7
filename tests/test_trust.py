"""
Tests for trust.py - Trust score computation based on data coverage.
"""
import pytest
from trust import compute_trust_features, trust_score


class TestComputeTrustFeatures:
    """Tests for compute_trust_features function."""

    def test_full_coverage(self):
        """Test with full data coverage."""
        home_state = {"MatchesPlayed": 25}
        away_state = {"MatchesPlayed": 30}
        
        features = compute_trust_features(
            home_state=home_state,
            away_state=away_state,
            h2h_home_wins=3,
            h2h_draws=2,
            h2h_away_wins=5,
            league_code=0,
        )
        
        assert features["form_n_home"] == 5
        assert features["form_n_away"] == 5
        assert features["history_n_home"] == 25
        assert features["history_n_away"] == 30
        assert features["h2h_n"] == 10
        assert features["league_ok"] == 1

    def test_no_coverage(self):
        """Test with no data coverage."""
        home_state = {"MatchesPlayed": 0}
        away_state = {"MatchesPlayed": 0}
        
        features = compute_trust_features(
            home_state=home_state,
            away_state=away_state,
            h2h_home_wins=0,
            h2h_draws=0,
            h2h_away_wins=0,
            league_code=-1,
        )
        
        assert features["form_n_home"] == 0
        assert features["form_n_away"] == 0
        assert features["history_n_home"] == 0
        assert features["history_n_away"] == 0
        assert features["h2h_n"] == 0
        assert features["league_ok"] == 0

    def test_partial_coverage(self):
        """Test with partial data coverage."""
        home_state = {"MatchesPlayed": 3}
        away_state = {"MatchesPlayed": 10}
        
        features = compute_trust_features(
            home_state=home_state,
            away_state=away_state,
            h2h_home_wins=1,
            h2h_draws=0,
            h2h_away_wins=1,
            league_code=1,
        )
        
        assert features["form_n_home"] == 3
        assert features["form_n_away"] == 5
        assert features["history_n_home"] == 3
        assert features["history_n_away"] == 10
        assert features["h2h_n"] == 2
        assert features["league_ok"] == 1

    def test_none_states(self):
        """Test with None states."""
        features = compute_trust_features(
            home_state=None,
            away_state=None,
            h2h_home_wins=0,
            h2h_draws=0,
            h2h_away_wins=0,
            league_code=-1,
        )
        
        assert features["form_n_home"] == 0
        assert features["form_n_away"] == 0
        assert features["history_n_home"] == 0
        assert features["history_n_away"] == 0


class TestTrustScore:
    """Tests for trust_score function."""

    def test_full_coverage_high_score(self):
        """Full coverage should give score near 100 and label HIGH."""
        features = {
            "form_n_home": 5,
            "form_n_away": 5,
            "history_n_home": 20,
            "history_n_away": 20,
            "h2h_n": 10,
            "league_ok": 1,
        }
        
        score, label = trust_score(features)
        
        assert score == 100
        assert label == "HIGH"

    def test_no_coverage_low_score(self):
        """No coverage should give score 0 and label LOW."""
        features = {
            "form_n_home": 0,
            "form_n_away": 0,
            "history_n_home": 0,
            "history_n_away": 0,
            "h2h_n": 0,
            "league_ok": 0,
        }
        
        score, label = trust_score(features)
        
        assert score == 0
        assert label == "LOW"

    def test_medium_coverage_med_score(self):
        """Medium coverage should give MED label."""
        features = {
            "form_n_home": 3,
            "form_n_away": 3,
            "history_n_home": 10,
            "history_n_away": 10,
            "h2h_n": 2,
            "league_ok": 1,
        }
        
        score, label = trust_score(features)
        
        assert 40 <= score < 70
        assert label == "MED"

    def test_high_threshold_boundary(self):
        """Test boundary at HIGH threshold (70)."""
        features = {
            "form_n_home": 5,
            "form_n_away": 5,
            "history_n_home": 14,
            "history_n_away": 14,
            "h2h_n": 5,
            "league_ok": 1,
        }
        
        score, label = trust_score(features)
        
        assert score >= 70
        assert label == "HIGH"

    def test_med_threshold_boundary(self):
        """Test boundary at MED threshold (40)."""
        features = {
            "form_n_home": 3,
            "form_n_away": 3,
            "history_n_home": 8,
            "history_n_away": 8,
            "h2h_n": 2,
            "league_ok": 1,
        }
        
        score, label = trust_score(features)
        
        assert score >= 40
        assert label == "MED"

    def test_low_threshold_boundary(self):
        """Test boundary at LOW threshold (< 40)."""
        features = {
            "form_n_home": 1,
            "form_n_away": 1,
            "history_n_home": 2,
            "history_n_away": 2,
            "h2h_n": 0,
            "league_ok": 0,
        }
        
        score, label = trust_score(features)
        
        assert score < 40
        assert label == "LOW"

    def test_score_components(self):
        """Test that score components add up correctly."""
        features = {
            "form_n_home": 5,
            "form_n_away": 5,
            "history_n_home": 20,
            "history_n_away": 20,
            "h2h_n": 10,
            "league_ok": 1,
        }
        
        score, _ = trust_score(features)
        
        expected_form = 15 + 15
        expected_history = 20 + 20
        expected_h2h = 20
        expected_league = 10
        expected_total = expected_form + expected_history + expected_h2h + expected_league
        
        assert score == expected_total

    def test_capped_values(self):
        """Test that values above caps don't increase score."""
        features_capped = {
            "form_n_home": 10,
            "form_n_away": 10,
            "history_n_home": 50,
            "history_n_away": 50,
            "h2h_n": 20,
            "league_ok": 1,
        }
        
        features_max = {
            "form_n_home": 5,
            "form_n_away": 5,
            "history_n_home": 20,
            "history_n_away": 20,
            "h2h_n": 10,
            "league_ok": 1,
        }
        
        score_capped, _ = trust_score(features_capped)
        score_max, _ = trust_score(features_max)
        
        assert score_capped == score_max


class TestTrustScoreIntegration:
    """Integration tests for trust score computation."""

    def test_compute_and_score_full_coverage(self):
        """Test full pipeline with full coverage."""
        home_state = {"MatchesPlayed": 25}
        away_state = {"MatchesPlayed": 30}
        
        features = compute_trust_features(
            home_state=home_state,
            away_state=away_state,
            h2h_home_wins=4,
            h2h_draws=3,
            h2h_away_wins=3,
            league_code=0,
        )
        
        score, label = trust_score(features)
        
        assert score == 100
        assert label == "HIGH"

    def test_compute_and_score_no_coverage(self):
        """Test full pipeline with no coverage."""
        home_state = {"MatchesPlayed": 0}
        away_state = {"MatchesPlayed": 0}
        
        features = compute_trust_features(
            home_state=home_state,
            away_state=away_state,
            h2h_home_wins=0,
            h2h_draws=0,
            h2h_away_wins=0,
            league_code=-1,
        )
        
        score, label = trust_score(features)
        
        assert score == 0
        assert label == "LOW"

    def test_compute_and_score_medium_coverage(self):
        """Test full pipeline with medium coverage."""
        home_state = {"MatchesPlayed": 8}
        away_state = {"MatchesPlayed": 12}
        
        features = compute_trust_features(
            home_state=home_state,
            away_state=away_state,
            h2h_home_wins=1,
            h2h_draws=1,
            h2h_away_wins=0,
            league_code=2,
        )
        
        score, label = trust_score(features)
        
        assert 40 <= score < 70
        assert label == "MED"
