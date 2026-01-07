"""
Analysera användarens 13 fotbollsmatcher
"""
import pandas as pd
import joblib
from utils import normalize_team_name

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

# Ladda modell och features
model = joblib.load("models/xgboost_model_v7_2526.joblib")
features_df = pd.read_parquet("data/features.parquet")

# Skapa en lookup för senaste features per lag
def get_latest_team_stats(team_name, df):
    """Hämta senaste statistik för ett lag"""
    home_matches = df[df['HomeTeam'] == team_name].copy()
    away_matches = df[df['AwayTeam'] == team_name].copy()
    
    if not home_matches.empty:
        home_matches['Date'] = pd.to_datetime(home_matches['Date'])
        latest_home = home_matches.sort_values('Date', ascending=False).iloc[0]
        return {
            'form_points': latest_home['HomeFormPts'],
            'form_gd': latest_home['HomeFormGD'],
            'elo': latest_home['HomeElo']
        }
    elif not away_matches.empty:
        away_matches['Date'] = pd.to_datetime(away_matches['Date'])
        latest_away = away_matches.sort_values('Date', ascending=False).iloc[0]
        return {
            'form_points': latest_away['AwayFormPts'],
            'form_gd': latest_away['AwayFormGD'],
            'elo': latest_away['AwayElo']
        }
    else:
        return None

print("=" * 80)
print("ANALYS AV DINA 13 FOTBOLLSMATCHER")
print("=" * 80)
print()

results = []

for i, (home, away) in enumerate(matches, 1):
    # Normalisera lagnamn
    home_norm = normalize_team_name(home)
    away_norm = normalize_team_name(away)
    
    print(f"Match {i}: {home} vs {away}")
    print(f"Normaliserat: {home_norm} vs {away_norm}")
    
    # Hämta features för lagen
    home_stats = get_latest_team_stats(home_norm, features_df)
    away_stats = get_latest_team_stats(away_norm, features_df)
    
    if home_stats is None or away_stats is None:
        print(f"⚠️  VARNING: Lag saknas i databasen!")
        if home_stats is None:
            print(f"   - {home_norm} finns inte")
        if away_stats is None:
            print(f"   - {away_norm} finns inte")
        print()
        results.append({
            "Match": f"{home} - {away}",
            "Tips": "N/A",
            "Sannolikhet": "N/A",
            "Status": "Lag saknas"
        })
        continue
    
    # Skapa feature-vektor för matchen (använd samma namn som modellen tränades med)
    feature_cols = [
        'HomeFormPts', 'HomeFormGD', 'HomeElo',
        'AwayFormPts', 'AwayFormGD', 'AwayElo'
    ]
    
    X = pd.DataFrame([[
        home_stats['form_points'],
        home_stats['form_gd'],
        home_stats['elo'],
        away_stats['form_points'],
        away_stats['form_gd'],
        away_stats['elo']
    ]], columns=feature_cols)
    
    # Prediktion
    proba = model.predict_proba(X)[0]
    pred_class = model.predict(X)[0]
    
    # Sannolikheter: [Away, Draw, Home] (alfabetisk ordning: A, D, H)
    prob_away = proba[0]
    prob_draw = proba[1]
    prob_home = proba[2]
    
    # Bestäm tips
    max_prob = max(prob_home, prob_draw, prob_away)
    
    if max_prob == prob_home:
        tip = "1"
        tip_text = f"Hemmavinst ({home})"
    elif max_prob == prob_draw:
        tip = "X"
        tip_text = "Oavgjort"
    else:
        tip = "2"
        tip_text = f"Bortavinst ({away})"
    
    # Halvgardering?
    sorted_probs = sorted([prob_home, prob_draw, prob_away], reverse=True)
    confidence = sorted_probs[0] - sorted_probs[1]
    
    if confidence < 0.15:  # Osäker match
        # Hitta de två högsta sannolikheterna
        if prob_home > prob_away:
            if prob_draw > prob_away:
                halv = "1X"
            else:
                halv = "12"
        else:
            if prob_draw > prob_home:
                halv = "X2"
            else:
                halv = "12"
    else:
        halv = tip
    
    print(f"📊 Sannolikheter:")
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
        "Säkerhet": f"{confidence:.1%}"
    })

# Sammanfattning
print("=" * 80)
print("SAMMANFATTNING - TIPSRAD")
print("=" * 80)
print()

tipsrad = " ".join([r["Tips"] for r in results if r["Tips"] != "N/A"])
print(f"Tipsrad (utan halvgarderingar): {tipsrad}")
print()

tipsrad_halv = " ".join([r["Halvgardering"] for r in results if r.get("Halvgardering", "N/A") != "N/A"])
print(f"Tipsrad (med halvgarderingar):  {tipsrad_halv}")
print()

# Tabell
print("DETALJERAD TABELL:")
print("-" * 80)
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
print()

# Spara till fil
df_results.to_csv("/home/ubuntu/matcher_analys.csv", index=False)
print("✅ Resultat sparade till: /home/ubuntu/matcher_analys.csv")
