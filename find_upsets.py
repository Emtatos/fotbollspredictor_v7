"""
Script för att hitta potentiella skrällmatcher
"""
import pandas as pd
import numpy as np
from pathlib import Path
from model_handler import load_model
from news_scraper_v2 import IntelligentFootballAnalyzer
from prediction_with_context import adjust_probabilities_with_context
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_elo_difference(home_elo: float, away_elo: float) -> float:
    """Beräknar ELO-skillnad (positivt = hemmalaget starkare)"""
    return home_elo - away_elo


def get_upset_potential(
    home_team: str,
    away_team: str,
    df_features: pd.DataFrame,
    model
) -> dict:
    """
    Analyserar en matchs skrällpotential
    
    Returns:
        Dict med skrällanalys
    """
    # Hämta senaste data för lagen
    home_matches = df_features[(df_features['HomeTeam'] == home_team) | (df_features['AwayTeam'] == home_team)]
    away_matches = df_features[(df_features['HomeTeam'] == away_team) | (df_features['AwayTeam'] == away_team)]
    
    if home_matches.empty or away_matches.empty:
        return None
    
    home_latest = home_matches.iloc[-1]
    away_latest = away_matches.iloc[-1]
    
    # Extrahera features
    if home_latest['HomeTeam'] == home_team:
        h_form_pts = home_latest['HomeFormPts']
        h_form_gd = home_latest['HomeFormGD']
        h_elo = home_latest['HomeElo']
        h_goals_for = home_latest.get('HomeGoalsFor', 0)
        h_form_home = home_latest.get('HomeFormHome', 0)
    else:
        h_form_pts = home_latest['AwayFormPts']
        h_form_gd = home_latest['AwayFormGD']
        h_elo = home_latest['AwayElo']
        h_goals_for = home_latest.get('AwayGoalsFor', 0)
        h_form_home = home_latest.get('AwayFormAway', 0)
    
    if away_latest['HomeTeam'] == away_team:
        a_form_pts = away_latest['HomeFormPts']
        a_form_gd = away_latest['HomeFormGD']
        a_elo = away_latest['HomeElo']
        a_goals_for = away_latest.get('HomeGoalsFor', 0)
        a_form_away = away_latest.get('HomeFormHome', 0)
    else:
        a_form_pts = away_latest['AwayFormPts']
        a_form_gd = away_latest['AwayFormGD']
        a_elo = away_latest['AwayElo']
        a_goals_for = away_latest.get('AwayGoalsFor', 0)
        a_form_away = away_latest.get('AwayFormAway', 0)
    
    # Extrahera alla 21 features
    feature_cols = ['HomeFormPts', 'HomeFormGD', 'AwayFormPts', 'AwayFormGD', 'HomeFormHome', 'AwayFormAway',
                    'HomeGoalsFor', 'HomeGoalsAgainst', 'AwayGoalsFor', 'AwayGoalsAgainst',
                    'HomeStreak', 'AwayStreak', 'H2H_HomeWins', 'H2H_Draws', 'H2H_AwayWins', 'H2H_HomeGoalDiff',
                    'HomePosition', 'AwayPosition', 'PositionDiff', 'HomeElo', 'AwayElo']
    
    # Skapa feature vector från senaste matchdata
    feature_vector = []
    for col in feature_cols:
        if col.startswith('Home'):
            val = home_latest.get(col, 0)
        elif col.startswith('Away'):
            val = away_latest.get(col, 0)
        elif col.startswith('H2H') or col == 'PositionDiff':
            val = home_latest.get(col, 0)
        else:
            val = 0
        feature_vector.append(val)
    
    feature_vector = np.array([feature_vector])
    probs = model.predict_proba(feature_vector)[0]
    
    # Justera med AI-kontext
    adjusted_probs, context = adjust_probabilities_with_context(
        probs, home_team, away_team, use_context=True
    )
    
    # Beräkna skrällpotential
    elo_diff = calculate_elo_difference(h_elo, a_elo)
    form_diff = h_form_pts - a_form_pts
    
    # Identifiera underdog
    is_home_underdog = elo_diff < -50  # Hemmalaget är underdog
    is_away_underdog = elo_diff > 50   # Bortalaget är underdog
    
    upset_score = 0
    upset_reasons = []
    
    if is_home_underdog:
        # Hemmalaget är underdog - leta efter skrällpotential
        if h_form_pts > a_form_pts:
            upset_score += 2
            upset_reasons.append(f"Bättre form ({h_form_pts:.1f} vs {a_form_pts:.1f})")
        
        if h_form_home > 7:
            upset_score += 1
            upset_reasons.append(f"Stark hemmaform ({h_form_home:.1f}/10)")
        
        if context and context['source'] != 'fallback':
            if context['away_injuries'] > context['home_injuries'] + 2:
                upset_score += 2
                upset_reasons.append(f"Motståndaren har fler skador ({context['away_injuries']} vs {context['home_injuries']})")
            
            if context['away_form'] < 5:
                upset_score += 1
                upset_reasons.append(f"Motståndaren i dålig form ({context['away_form']}/10)")
        
        if adjusted_probs[0] > 0.30:  # Minst 30% chans för hemmavinst
            upset_score += 1
            upset_reasons.append(f"Modellen ger {adjusted_probs[0]:.1%} chans")
        
        return {
            'home': home_team,
            'away': away_team,
            'underdog': home_team,
            'upset_score': upset_score,
            'reasons': upset_reasons,
            'elo_diff': elo_diff,
            'form_diff': form_diff,
            'prob_home': adjusted_probs[0],
            'prob_draw': adjusted_probs[1],
            'prob_away': adjusted_probs[2],
            'context': context
        }
    
    elif is_away_underdog:
        # Bortalaget är underdog
        if a_form_pts > h_form_pts:
            upset_score += 2
            upset_reasons.append(f"Bättre form ({a_form_pts:.1f} vs {h_form_pts:.1f})")
        
        if a_form_away > 6:
            upset_score += 1
            upset_reasons.append(f"Stark bortaform ({a_form_away:.1f}/10)")
        
        if context and context['source'] != 'fallback':
            if context['home_injuries'] > context['away_injuries'] + 2:
                upset_score += 2
                upset_reasons.append(f"Motståndaren har fler skador ({context['home_injuries']} vs {context['away_injuries']})")
            
            if context['home_form'] < 5:
                upset_score += 1
                upset_reasons.append(f"Motståndaren i dålig form ({context['home_form']}/10)")
        
        if adjusted_probs[2] > 0.25:  # Minst 25% chans för bortavinst
            upset_score += 1
            upset_reasons.append(f"Modellen ger {adjusted_probs[2]:.1%} chans")
        
        return {
            'home': home_team,
            'away': away_team,
            'underdog': away_team,
            'upset_score': upset_score,
            'reasons': upset_reasons,
            'elo_diff': elo_diff,
            'form_diff': form_diff,
            'prob_home': adjusted_probs[0],
            'prob_draw': adjusted_probs[1],
            'prob_away': adjusted_probs[2],
            'context': context
        }
    
    return None


def find_upset_candidates(df_features: pd.DataFrame, model, top_n: int = 3) -> list:
    """
    Hittar de bästa skrällkandidaterna
    
    Args:
        df_features: Feature-data
        model: Tränad modell
        top_n: Antal kandidater att returnera
        
    Returns:
        Lista med skrällkandidater sorterade efter upset_score
    """
    logger.info("Analyserar potentiella skrällmatcher...")
    
    # Hämta alla unika lag
    all_teams = sorted(set(df_features['HomeTeam'].unique()) | set(df_features['AwayTeam'].unique()))
    
    # Generera intressanta matchups (stora ELO-skillnader)
    candidates = []
    
    # Testa några klassiska "David vs Goliath"-matcher
    test_matches = [
        # Premier League underdogs mot topplag
        ("Fulham", "Arsenal"),
        ("Brentford", "Liverpool"),
        ("Brighton", "Manchester City"),
        ("Bournemouth", "Chelsea"),
        ("Crystal Palace", "Tottenham"),
        ("Nottingham Forest", "Newcastle"),
        ("Wolves", "Man United"),
        
        # Championship underdogs
        ("Luton", "Leeds United"),
        ("Millwall", "Burnley"),
        ("Coventry", "Sheffield United"),
        ("Watford", "Norwich"),
        
        # League One/Two
        ("Cambridge", "Birmingham"),
        ("Salford", "Bolton"),
        ("Leyton Orient", "Charlton"),
    ]
    
    for home, away in test_matches:
        # Normalisera lagnamn
        home_matches = df_features[df_features['HomeTeam'].str.contains(home, case=False, na=False)]
        away_matches = df_features[df_features['AwayTeam'].str.contains(away, case=False, na=False)]
        
        if not home_matches.empty and not away_matches.empty:
            home_team = home_matches.iloc[0]['HomeTeam']
            away_team = away_matches.iloc[0]['AwayTeam']
            
            result = get_upset_potential(home_team, away_team, df_features, model)
            
            if result and result['upset_score'] > 0:
                candidates.append(result)
    
    # Sortera efter upset_score
    candidates.sort(key=lambda x: x['upset_score'], reverse=True)
    
    return candidates[:top_n]


def main():
    """Huvudfunktion"""
    # Ladda modell och data
    model_path = Path("models") / "xgboost_model_v7_2526.joblib"
    model = load_model(model_path)
    
    df_features = pd.read_parquet(Path("data") / "features.parquet")
    
    if model is None or df_features is None:
        print("❌ Kunde inte ladda modell eller data")
        return
    
    print("\n" + "="*80)
    print("🎯 SKRÄLLANALYS - TOP 3 UNDERDOGS MED VINSTCHANS")
    print("="*80 + "\n")
    
    candidates = find_upset_candidates(df_features, model, top_n=3)
    
    if not candidates:
        print("Inga skrällkandidater hittades.")
        return
    
    for i, candidate in enumerate(candidates, 1):
        print(f"\n{'='*80}")
        print(f"#{i} SKRÄLLKANDIDAT: {candidate['underdog']} (Underdog)")
        print(f"{'='*80}")
        print(f"Match: {candidate['home']} vs {candidate['away']}")
        print(f"Skrällpoäng: {candidate['upset_score']}/10 ⭐")
        print(f"\n📊 Odds:")
        print(f"  Hemmavinst: {candidate['prob_home']:.1%}")
        print(f"  Oavgjort: {candidate['prob_draw']:.1%}")
        print(f"  Bortavinst: {candidate['prob_away']:.1%}")
        print(f"\n📈 Statistik:")
        print(f"  ELO-skillnad: {candidate['elo_diff']:+.0f} (negativt = underdog)")
        print(f"  Form-skillnad: {candidate['form_diff']:+.1f} poäng")
        print(f"\n💡 Skrällfaktorer:")
        for reason in candidate['reasons']:
            print(f"  ✓ {reason}")
        
        if candidate['context'] and candidate['context']['source'] != 'fallback':
            ctx = candidate['context']
            print(f"\n🤖 AI-analys:")
            print(f"  {candidate['home']} skador: {ctx['home_injuries']}/10")
            print(f"  {candidate['away']} skador: {ctx['away_injuries']}/10")
            print(f"  {candidate['home']} form: {ctx['home_form']}/10")
            print(f"  {candidate['away']} form: {ctx['away_form']}/10")
            if ctx['home_issues'] != 'Ingen information tillgänglig':
                print(f"  Problem {candidate['home']}: {ctx['home_issues'][:80]}...")
            if ctx['away_issues'] != 'Ingen information tillgänglig':
                print(f"  Problem {candidate['away']}: {ctx['away_issues'][:80]}...")
    
    print(f"\n{'='*80}")
    print("💡 TIPS: Överväg dessa matcher för skrällspel!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
