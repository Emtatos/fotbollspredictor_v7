"""Analysera användarens 13 fotbollsmatcher.

Fixar:
- Korrekt mapping av predict_proba (H/D/A = 0/1/2 -> 1/X/2)
- Bygger features konsekvent via single source of truth (schema.py)
- Använder "current team state" (state replay) i stället för senaste matchrad
- Räknar H2H för rätt lagpar (on-the-fly)
"""
from __future__ import annotations

import joblib
import pandas as pd

from utils import normalize_team_name
from state import build_current_team_states
from features import compute_h2h
from inference import build_feature_row, predict_match
from schema import FEATURE_COLUMNS


# Matcher att analysera
matches = [
    ("Charlton", "Chelsea"),
    ("Tottenham", "Aston Villa"),
    ("Newcastle", "Bournemouth"),
    ("Fulham", "Middlesbrough"),
    ("Bristol City", "Watford"),
    ("Stoke", "Coventry"),
    ("Doncaster", "Southampton"),
    ("Cambridge", "Birmingham"),
    ("Burnley", "Millwall"),
    ("Salford", "Swindon"),
    ("Bradford", "Rotherham"),
    ("Leyton Orient", "Cardiff"),
    ("Peterborough", "Bolton"),
]


MODEL_PATH = "models/xgboost_model_v7_2526.joblib"
FEATURES_PATH = "data/features.parquet"


def main() -> None:
    model = joblib.load(MODEL_PATH)
    df = pd.read_parquet(FEATURES_PATH)

    # För state-replay behövs grundkolumner
    need_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"features.parquet saknar nödvändiga kolumner för state-replay: {missing}")

    # Normalisera lagnamn så vi matchar dataset
    df = df.copy()
    df["HomeTeam"] = df["HomeTeam"].apply(normalize_team_name)
    df["AwayTeam"] = df["AwayTeam"].apply(normalize_team_name)

    states = build_current_team_states(df)

    print("=" * 80)
    print("ANALYS AV DINA 13 FOTBOLLSMATCHER")
    print("=" * 80)
    print()

    results = []

    for i, (home, away) in enumerate(matches, 1):
        home_norm = normalize_team_name(home)
        away_norm = normalize_team_name(away)

        print(f"Match {i}: {home} vs {away}")
        print(f"Normaliserat: {home_norm} vs {away_norm}")

        hs = states.get(home_norm)
        as_ = states.get(away_norm)

        if hs is None or as_ is None:
            print("⚠️  VARNING: Lag saknas i databasen!")
            if hs is None:
                print(f"   - {home_norm} finns inte")
            if as_ is None:
                print(f"   - {away_norm} finns inte")
            print()
            results.append({
                "Match": f"{home} - {away}",
                "Tips": "N/A",
                "Halvgardering": "N/A",
                "Status": "Lag saknas"
            })
            continue

        h2h_hw, h2h_d, h2h_aw, h2h_gd = compute_h2h(df, home_norm, away_norm)

        # Bygg feature row enligt kontraktet
        feature_dict = {
            "HomeFormPts": hs["FormPts"],
            "HomeFormGD": hs["FormGD"],
            "AwayFormPts": as_["FormPts"],
            "AwayFormGD": as_["FormGD"],
            "HomeFormHome": hs["FormHome"],
            "AwayFormAway": as_["FormAway"],
            "HomeGoalsFor": hs["GoalsFor"],
            "HomeGoalsAgainst": hs["GoalsAgainst"],
            "AwayGoalsFor": as_["GoalsFor"],
            "AwayGoalsAgainst": as_["GoalsAgainst"],
            "HomeStreak": hs["Streak"],
            "AwayStreak": as_["Streak"],
            "H2H_HomeWins": h2h_hw,
            "H2H_Draws": h2h_d,
            "H2H_AwayWins": h2h_aw,
            "H2H_HomeGoalDiff": h2h_gd,
            "HomePosition": hs["Position"],
            "AwayPosition": as_["Position"],
            "PositionDiff": as_["Position"] - hs["Position"],  # samma som i feature_engineering.py
            "HomeElo": hs["Elo"],
            "AwayElo": as_["Elo"],
            "League": hs.get("League", -1),
            # injury defaults (uppdateras i app med live-scraper)
            "InjuredPlayers_Home": 0,
            "InjuredPlayers_Away": 0,
            "KeyPlayersOut_Home": 0,
            "KeyPlayersOut_Away": 0,
            "InjurySeverity_Home": 0.0,
            "InjurySeverity_Away": 0.0,
        }

        row_df = build_feature_row(feature_dict)  # ordning + numeric coercion

        probs = predict_match(model, row_df)  # {"1","X","2"}
        prob_home = probs["1"]
        prob_draw = probs["X"]
        prob_away = probs["2"]

        # Bestäm tips
        items = [("1", prob_home, f"Hemmavinst ({home})"),
                 ("X", prob_draw, "Oavgjort"),
                 ("2", prob_away, f"Bortavinst ({away})")]
        items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
        tip, best_p, tip_text = items_sorted[0]
        second_p = items_sorted[1][1]
        confidence = best_p - second_p

        # Halvgardering om osäker
        if confidence < 0.15:
            halv = "".join(sorted([items_sorted[0][0], items_sorted[1][0]]))
            # behåll traditionell ordning 1X,12,X2
            order = {"1X": "1X", "12": "12", "2X": "X2", "X2": "X2", "1X": "1X"}
            if halv in ("2X",):
                halv = "X2"
        else:
            halv = tip

        print("📊 Sannolikheter:")
        print(f"   Hemmavinst (1): {prob_home:.1%}")
        print(f"   Oavgjort (X):   {prob_draw:.1%}")
        print(f"   Bortavinst (2): {prob_away:.1%}")
        print(f"💡 Tips: {tip} ({tip_text})")
        print(f"🎯 Halvgardering: {halv}")
        print(f"📈 Säkerhet: {confidence:.1%}")
        print()

        results.append({
            "Match": f"{home} - {away}",
            "Tips": tip,
            "Halvgardering": halv,
            "Hemma%": f"{prob_home:.1%}",
            "Oavgjort%": f"{prob_draw:.1%}",
            "Borta%": f"{prob_away:.1%}",
            "Säkerhet": f"{confidence:.1%}",
        })

    # Sammanfattning
    print("=" * 80)
    print("SAMMANFATTNING - TIPSRAD")
    print("=" * 80)
    print()

    tipsrad = " ".join([r["Tips"] for r in results if r.get("Tips") not in (None, "N/A")])
    print(f"Tipsrad (utan halvgarderingar): {tipsrad}")
    print()

    tipsrad_halv = " ".join([r.get("Halvgardering", "N/A") for r in results if r.get("Halvgardering") != "N/A"])
    print(f"Tipsrad (med halvgarderingar):  {tipsrad_halv}")
    print()

    # Spara till CSV
    pd.DataFrame(results).to_csv("tips_analysis_results.csv", index=False, encoding="utf-8-sig")
    print("✅ Resultat sparat i tips_analysis_results.csv")


if __name__ == "__main__":
    main()
