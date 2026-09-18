"""
Regressionstester: Svenska Spels lagnamn ska normaliseras till lag som
finns i modellens träningsdata så att varje match får modellbidrag.

Träningsdatan simuleras med de råa lagnamnen från football-data.co.uk
(E0–E3, säsong 2324–2627) – ingen nätverks- eller Postgres-åtkomst krävs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils import (
    normalize_team_name,
    set_canonical_teams,
    get_canonical_teams,
    audit_team_names,
    unmatched_team_names,
    TEAM_ALIASES,
)
from feature_engineering import create_features
from model_handler import train_and_save_model
from app_helpers import predict_model_probs


FIXTURES = Path(__file__).parent / "fixtures"

# Råa lagnamn som football-data.co.uk använder i E0–E3 (2324–2627).
FOOTBALL_DATA_RAW_TEAMS = [
    "AFC Wimbledon", "Accrington", "Arsenal", "Aston Villa", "Barnet", "Barnsley", "Barrow",
    "Birmingham", "Blackburn", "Blackpool", "Bolton", "Bournemouth", "Bradford", "Brentford",
    "Brighton", "Bristol City", "Bristol Rvs", "Bromley", "Burnley", "Burton", "Cambridge",
    "Cardiff", "Carlisle", "Charlton", "Chelsea", "Cheltenham", "Chesterfield", "Colchester",
    "Coventry", "Crawley Town", "Crewe", "Crystal Palace", "Derby", "Doncaster", "Everton",
    "Exeter", "Fleetwood Town", "Fulham", "Gillingham", "Grimsby", "Harrogate", "Huddersfield",
    "Hull", "Ipswich", "Leeds", "Leicester", "Leyton Orient", "Lincoln", "Liverpool", "Luton",
    "Man City", "Man United", "Mansfield", "Middlesbrough", "Millwall", "Milton Keynes Dons",
    "Morecambe", "Newcastle", "Newport County", "Northampton", "Norwich", "Nott'm Forest",
    "Notts County", "Oldham", "Oxford", "Peterboro", "Plymouth", "Port Vale", "Portsmouth",
    "Preston", "QPR", "Reading", "Rochdale", "Rotherham", "Salford", "Sheffield United",
    "Sheffield Weds", "Shrewsbury", "Southampton", "Stevenage", "Stockport", "Stoke",
    "Sunderland", "Swansea", "Swindon", "Tottenham", "Tranmere", "Walsall", "Watford",
    "West Brom", "West Ham", "Wigan", "Wolves", "Wrexham", "Wycombe", "York",
]

# Svenska Spel v38 (draw 4971) – fältet `name` per deltagare.
V38_MATCHES = [
    ("Nottingham", "Coventry"),
    ("Brighton", "Arsenal"),
    ("Everton", "Ipswich"),
    ("Newcastle", "Hull"),
    ("Birmingham", "Middlesbrough"),
    ("Burnley", "Derby"),
    ("Lincoln", "Swansea"),
    ("Portsmouth", "Blackburn"),
    ("Queens Park Rangers", "Preston"),
    ("Wrexham", "Southampton"),
    ("Luton", "Bradford"),
    ("Oxford", "Cambridge"),
    ("Sheffield W", "Stockport"),
]

# Övriga kända Svenska Spel-namn (`name` och 9-teckens `mediumName`).
OTHER_SVENSKA_SPEL_NAMES = [
    "Nottingha", "Sheff W", "Sheff U", "Sheffield U", "Man U", "Manchester U", "Manchester C",
    "Birmingha", "Middlesbr", "Portsmout", "Southampt", "Huddersfi", "Bournemou", "Leyton Or",
    "QPR", "Bristol C", "Bristol R", "MK Dons", "Peterborough", "Peterborough United",
    "Accrington Stanley", "Burton Albion", "Crewe Alexandra", "Wycombe Wanderers",
    "Crawley", "Newport", "Tottenham", "West Ham", "West Bromwich", "Wolverhampton",
    "Manchester United", "Manchester City", "Sheffield United", "Sheffield Wednesday",
    "Nottingham Forest", "Leeds", "Leicester", "Norwich", "Cardiff", "Huddersfield",
    "Charlton", "Plymouth", "Rotherham", "Wigan", "Stoke", "Salford", "Swindon",
]


def _training_canonical_teams() -> set:
    """Kanoniska namn som FeatureBuilder.fit() ger träningsdatan."""
    set_canonical_teams(FOOTBALL_DATA_RAW_TEAMS)
    return {normalize_team_name(t) for t in FOOTBALL_DATA_RAW_TEAMS}


@pytest.fixture
def training_teams():
    canon = _training_canonical_teams()
    set_canonical_teams(canon)
    yield canon
    set_canonical_teams([])


class TestSvenskaSpelAliases:
    def test_required_aliases(self, training_teams):
        assert normalize_team_name("Nottingham") == "Nottingham Forest"
        assert normalize_team_name("Sheffield W") == "Sheffield Wednesday"
        assert normalize_team_name("Stockport") == "Stockport County"

    def test_medium_name_variants(self, training_teams):
        assert normalize_team_name("Nottingha") == "Nottingham Forest"
        assert normalize_team_name("Sheff W") == "Sheffield Wednesday"
        assert normalize_team_name("Birmingha") == "Birmingham City"
        assert normalize_team_name("Middlesbr") == "Middlesbrough"
        assert normalize_team_name("Portsmout") == "Portsmouth"
        assert normalize_team_name("Southampt") == "Southampton"

    def test_alias_targets_exist_in_training_data(self, training_teams):
        """Alias får bara peka på namn som faktiskt finns i träningsdatan."""
        # Spurs/Tottenham Hotspur är ett äldre alias utan Svenska Spel-koppling.
        skip = {"Spurs"}
        bad = {k: v for k, v in TEAM_ALIASES.items() if k not in skip and v not in training_teams}
        assert bad == {}

    def test_alias_lookup_without_canonical_set(self):
        set_canonical_teams([])
        assert normalize_team_name("Nottingham") == "Nottingham Forest"
        assert normalize_team_name("Sheffield W") == "Sheffield Wednesday"

    def test_existing_aliases_unchanged(self, training_teams):
        assert normalize_team_name("Man United") == "Manchester United"
        assert normalize_team_name("Nott'm Forest") == "Nottingham Forest"
        assert normalize_team_name("Sheffield Weds") == "Sheffield Wednesday"
        assert normalize_team_name("Sheff Utd") == "Sheffield United"
        assert normalize_team_name("QPR") == "Queens Park Rangers"
        assert normalize_team_name("Brighton") == "Brighton & Hove Albion"

    def test_fuzzy_and_prefix_do_not_guess_ambiguous(self, training_teams):
        assert normalize_team_name("Completely Unknown FC") == "Completely Unknown FC"
        # "Sheffield" är prefix till två lag och får inte gissas via prefix.
        assert normalize_team_name("Sheffield") not in ("Sheffield United", "Sheffield Wednesday")


class TestV38Coverage:
    def test_all_v38_names_in_training_data(self, training_teams):
        names = [n for m in V38_MATCHES for n in m]
        missing = unmatched_team_names(names, training_teams)
        assert missing == [], f"Saknar träningsdata: {missing}"

    def test_fixture_participant_names_in_training_data(self, training_teams):
        draw = json.loads((FIXTURES / "svenskaspel_draw_4971.json").read_text())
        names = []
        for event in draw["draw"]["drawEvents"]:
            for p in event["match"]["participants"]:
                names.append(p["name"])
                names.append(p["mediumName"])
        missing = unmatched_team_names(names, training_teams)
        assert missing == [], f"Saknar träningsdata: {missing}"

    def test_other_known_svenska_spel_names(self, training_teams):
        missing = unmatched_team_names(OTHER_SVENSKA_SPEL_NAMES, training_teams)
        assert missing == [], f"Saknar träningsdata: {missing}"

    def test_audit_helper_reports_fields(self, training_teams):
        rows = audit_team_names(["Nottingham", "Forest Green"], training_teams)
        assert rows[0].raw == "Nottingham"
        assert rows[0].normalized == "Nottingham Forest"
        assert rows[0].in_training is True
        assert rows[1].normalized == "Forest Green"
        assert rows[1].in_training is False

    def test_audit_defaults_to_canonical_teams(self, training_teams):
        rows = audit_team_names(["Sheffield W"])
        assert rows[0].in_training is True
        assert get_canonical_teams() == training_teams


def _synthetic_history(raw_teams, n_rounds: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    date = pd.Timestamp("2024-08-01")
    teams = list(raw_teams)
    for _ in range(n_rounds):
        rng.shuffle(teams)
        for i in range(0, len(teams) - 1, 2):
            fthg, ftag = int(rng.integers(0, 4)), int(rng.integers(0, 4))
            ftr = "H" if fthg > ftag else ("A" if ftag > fthg else "D")
            rows.append({
                "Date": date, "HomeTeam": teams[i], "AwayTeam": teams[i + 1],
                "FTHG": fthg, "FTAG": ftag, "FTR": ftr, "League": "E1", "Season": "2425",
            })
        date += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


class TestV38ModelPredictions:
    def test_predict_model_probs_covers_all_13_matches(self, tmp_path):
        # Råa football-data-namn för de 26 v38-lagen
        raw = [
            "Nott'm Forest", "Coventry", "Brighton", "Arsenal", "Everton", "Ipswich",
            "Newcastle", "Hull", "Birmingham", "Middlesbrough", "Burnley", "Derby",
            "Lincoln", "Swansea", "Portsmouth", "Blackburn", "QPR", "Preston",
            "Wrexham", "Southampton", "Luton", "Bradford", "Oxford", "Cambridge",
            "Sheffield Weds", "Stockport",
        ]
        set_canonical_teams(raw)
        df_features = create_features(_synthetic_history(raw))
        canon = set(df_features["HomeTeam"].astype(str)) | set(df_features["AwayTeam"].astype(str))
        set_canonical_teams(canon)
        try:
            model = train_and_save_model(
                df_features, tmp_path / "model.joblib", run_hyperparam_search=False
            )
            assert model is not None
            probs = predict_model_probs(model, df_features, V38_MATCHES)
        finally:
            set_canonical_teams([])

        missing = [m for m, p in zip(V38_MATCHES, probs) if p is None]
        assert missing == [], f"Matcher utan modellbidrag: {missing}"
        assert len(probs) == 13
        for p in probs:
            assert p.shape == (3,)
            assert abs(float(p.sum()) - 1.0) < 1e-6
