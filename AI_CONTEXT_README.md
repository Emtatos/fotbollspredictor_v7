# AI-baserad Matchkontextanalys

## Översikt

Fotbollspredictor v7 har nu en intelligent AI-funktion som automatiskt analyserar aktuell matchinformation och justerar prediktionerna därefter.

## Funktioner

### 1. Automatisk Matchanalys

Systemet använder AI (GPT-4.1-mini) för att analysera:
- **Skador**: Antal skadade spelare per lag
- **Form**: Senaste prestationer och resultat
- **Tränarbyten**: Klubbproblem och tränarsituationer
- **Andra faktorer**: Motivation, tabellposition, etc.

### 2. Intelligent Justering av Prediktioner

Baserat på analysen justeras sannolikheterna automatiskt:
- Skador minskar chansen att vinna
- Bra form ökar chansen
- Tränarbyten och problem påverkar negativt

### 3. Gratis och Automatisk

- **Ingen manuell scraping** - AI hämtar information automatiskt
- **Inga API-kostnader** - Använder redan tillgänglig OpenAI-integration
- **Cachning** - Analys sparas i 1 timme för att undvika onödiga anrop

## Användning

### I Streamlit-appen

1. Aktivera "Visa AI-analys" i användargränssnittet
2. Systemet hämtar automatiskt aktuell information
3. Prediktionerna justeras baserat på analysen

### Programmatiskt

```python
from prediction_with_context import adjust_probabilities_with_context, format_context_summary
import numpy as np

# Ursprungliga sannolikheter från modellen
probs = np.array([0.45, 0.30, 0.25])  # H, D, A

# Justera med aktuell matchkontext
adjusted_probs, context = adjust_probabilities_with_context(
    probs,
    home_team="Arsenal",
    away_team="Liverpool",
    use_context=True
)

# Visa sammanfattning
summary = format_context_summary(context)
print(summary)
```

### Direkt AI-analys

```python
from news_scraper_v2 import get_match_context, IntelligentFootballAnalyzer

# Hämta matchkontext
context = get_match_context("Manchester City", "Chelsea", use_ai=True)

print(f"Hemmalag skador: {context['home_injuries']}/10")
print(f"Bortalag skador: {context['away_injuries']}/10")
print(f"Hemmalag form: {context['home_form']}/10")
print(f"Bortalag form: {context['away_form']}/10")
print(f"Problem hemma: {context['home_issues']}")
print(f"Problem borta: {context['away_issues']}")
```

## Exempel

### Exempel 1: Arsenal vs Liverpool

**Ursprungliga sannolikheter:**
- Hemmavinst: 45.0%
- Oavgjort: 30.0%
- Bortavinst: 25.0%

**AI-analys:**
- Arsenal skador: 2/10
- Liverpool skador: 3/10
- Arsenal form: 7/10
- Liverpool form: 6/10
- Bedömning: Gynnar Arsenal (+5%)

**Justerade sannolikheter:**
- Hemmavinst: 45.4% (+0.4%)
- Oavgjort: 29.9% (-0.1%)
- Bortavinst: 24.7% (-0.3%)

### Exempel 2: Manchester City vs Chelsea

**AI-analys:**
- Man City skador: 2/10
- Chelsea skador: 3/10
- Man City form: 8/10
- Chelsea form: 5/10
- Chelsea har tränarosäkerhet
- Bedömning: Gynnar Man City (+10%)

**Resultat:**
- Större justering till förmån för Man City
- Chelsea's problem minskar deras chanser ytterligare

## Teknisk Implementation

### Arkitektur

```
┌─────────────────┐
│  Streamlit App  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ prediction_with_context │
└────────┬────────────────┘
         │
         ▼
┌──────────────────┐
│ news_scraper_v2  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  OpenAI GPT-4.1  │
└──────────────────┘
```

### Moduler

1. **news_scraper_v2.py**
   - `IntelligentFootballAnalyzer`: Huvudklass för AI-analys
   - `get_match_context()`: Hämtar matchkontext med cachning

2. **prediction_with_context.py**
   - `adjust_probabilities_with_context()`: Justerar sannolikheter
   - `format_context_summary()`: Formaterar sammanfattning

3. **app.py**
   - Integrerar AI-analys i användargränssnittet
   - Toggle för att aktivera/inaktivera

## Konfiguration

### Miljövariabler

```bash
# OpenAI API-nyckel (redan konfigurerad i Manus)
OPENAI_API_KEY=your_key_here
```

### Streamlit Secrets

```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "your_key_here"
```

## Begränsningar

1. **AI-beroende**: Kräver OpenAI API-nyckel
2. **Kostnad**: Varje analys kostar ~$0.0001 (mycket billigt)
3. **Cachning**: Analys är giltig i 1 timme
4. **Noggrannhet**: AI kan inte ha 100% aktuell information

## Fallback-beteende

Om AI inte är tillgängligt:
- Systemet använder neutrala värden (0 skador, 5/10 form)
- Inga justeringar görs
- Prediktionerna baseras endast på historisk data

## Framtida Förbättringar

1. **Fler källor**: Integrera flera AI-modeller
2. **Realtidsdata**: Hämta från live-API:er
3. **Historisk validering**: Testa hur väl justeringarna fungerar
4. **Användarfeedback**: Låt användare rapportera felaktiga analyser

## Support

För frågor eller problem, se huvuddokumentationen i `README.md`.

---

**Version:** 7.3  
**Skapad:** 2026-01-06  
**AI-modell:** GPT-4.1-mini  
**Kostnad per analys:** ~$0.0001
