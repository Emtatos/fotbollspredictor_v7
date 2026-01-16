# Skade-funktionalitet i Fotbollspredictor v7

## Översikt

Fotbollspredictor v7 har nu stöd för att integrera **mänsklig påverkan** i prediktionerna, specifikt skador och suspenderingar. Detta förbättrar modellens noggrannhet genom att ta hänsyn till vilka spelare som saknas i varje match.

## Nya Features

Modellen använder nu **27 features** (upp från 21):

### Skade-relaterade features (6 nya)

| Feature | Beskrivning | Typ |
|:--------|:------------|:----|
| `InjuredPlayers_Home` | Totalt antal skadade spelare i hemmalaget | Integer |
| `InjuredPlayers_Away` | Totalt antal skadade spelare i bortalaget | Integer |
| `KeyPlayersOut_Home` | Antal skadade nyckelspelare i hemmalaget | Integer |
| `KeyPlayersOut_Away` | Antal skadade nyckelspelare i bortalaget | Integer |
| `InjurySeverity_Home` | Allvarlighetsgrad av skador (0-10 skala) | Float |
| `InjurySeverity_Away` | Allvarlighetsgrad av skador (0-10 skala) | Float |

### Hur severity beräknas

```python
severity = min(10, key_players_injured * 2 + (total_injured - key_players_injured) * 0.5)
```

- Varje skadad nyckelspelare ger **2 poäng**
- Varje annan skadad spelare ger **0.5 poäng**
- Maxvärde är **10**

## Datakälla

Skadedata hämtas från **API-Football** (https://www.api-football.com/):
- Gratis tier: 100 requests/dag
- Täcker Premier League, Championship, League One och League Two
- Uppdateras dagligen av API-leverantören

## Användning

### 1. Konfigurera API-nyckel

#### Lokalt (utveckling)
Skapa en `.env`-fil i projektets rot:
```bash
API_FOOTBALL_KEY=din_api_nyckel_här
```

#### På Render (produktion)
1. Gå till Render Dashboard
2. Välj din web service
3. Gå till "Environment"
4. Lägg till: `API_FOOTBALL_KEY` = `din_api_nyckel`

### 2. Uppdatera skadedata

#### Via Streamlit-appen (Rekommenderat)
1. Öppna appen i webbläsaren
2. Gå till sidomenyn
3. Klicka på **"🎪 Uppdatera skador & form"**
4. Vänta 10-30 sekunder medan data hämtas
5. Gör dina prediktioner med färsk data!

#### Via Python-kod
```python
from injury_scraper import update_injury_data

# Uppdatera skadedata
success = update_injury_data()

if success:
    print("✅ Skadedata uppdaterad!")
else:
    print("❌ Kunde inte uppdatera skadedata")
```

### 3. Workflow varje vecka

**Lördag kl 11:00** (1 timme innan matcher):
1. Öppna appen
2. Klicka "Uppdatera skador & form"
3. Gör prediktioner
4. Tippa! 🎯

## Teknisk implementation

### Moduler

#### `injury_scraper.py`
Huvudmodul för att hämta skadedata:
- `InjuryDataFetcher` - Klass för att hantera API-anrop
- `update_injury_data()` - Convenience-funktion för uppdatering
- `get_injury_features_for_match()` - Hämtar features för en specifik match

#### `feature_engineering.py`
Uppdaterad för att inkludera skade-features:
- `_add_injury_features()` - Lägger till skade-kolumner
- `create_features()` - Nu skapar 27 features istället för 21

#### `app.py`
Streamlit-appen med uppdateringsknapp:
- Visar status för skadedata i sidomenyn
- Knapp för att uppdatera data on-demand
- Integrerar skade-features i prediktioner automatiskt

### Dataflöde

```
1. Användare klickar "Uppdatera skador"
   ↓
2. injury_scraper.py hämtar data från API-Football
   ↓
3. Data sparas till data/injuries_latest.json
   ↓
4. Vid prediktion läser app.py skadedata
   ↓
5. get_injury_features_for_match() beräknar features
   ↓
6. Modellen får alla 27 features (inkl. skador)
   ↓
7. Prediktion görs med hänsyn till skador
```

## API-kostnad och begränsningar

### Gratis tier (API-Football)
- **100 requests/dag**
- **1 uppdatering/vecka ≈ 20 requests** (ett per lag i Premier League)
- **Du har gott om marginal!**

### Betald tier
Om du behöver fler requests:
- **Basic:** $10/månad (1000 requests/dag)
- **Pro:** $30/månad (10000 requests/dag)

### Tips för att spara requests
1. Uppdatera bara 1 gång per vecka (innan tipprunda)
2. Cacha data lokalt (görs automatiskt)
3. Använd `is_data_stale()` för att kolla om uppdatering behövs

## Felsökning

### "Skadedata saknas"
**Problem:** Ingen skadedata har hämtats än.
**Lösning:** Klicka "Uppdatera skador & form" i appen.

### "Kunde inte uppdatera skadedata"
**Problem:** API-nyckel saknas eller är ogiltig.
**Lösning:** 
1. Kontrollera att `API_FOOTBALL_KEY` är satt i `.env` eller Render
2. Verifiera att nyckeln är giltig på api-football.com
3. Kolla att du inte överskridit request-gränsen

### "Skadedata är gammal (>24h)"
**Problem:** Data är äldre än 24 timmar.
**Lösning:** Klicka "Uppdatera skador & form" för att hämta färsk data.

### Modellen ger samma resultat som innan
**Problem:** Modellen är inte omtränad med nya features.
**Lösning:** 
1. Klicka "Kör omträning av modell" i sidomenyn
2. Vänta 30-60 sekunder
3. Modellen använder nu alla 27 features

## Framtida förbättringar

Möjliga tillägg i framtida versioner:
- ✅ Skador och suspenderingar (implementerat)
- 🔄 Tränarbyte och "new manager bounce"
- 🔄 Spelarbetyg och form
- 🔄 Vilodagar och fixture congestion
- 🔄 Väder och spelförhållanden
- 🔄 Historisk skadedata (inte bara aktuell)

## Exempel

### Scenario: Arsenal vs Chelsea

**Utan skade-features:**
```
Arsenal vs Chelsea
1: 55% | X: 25% | 2: 20%
Tips: 1
```

**Med skade-features (Arsenal saknar 3 nyckelspelare):**
```
Arsenal vs Chelsea
Skador Arsenal: 3 nyckelspelare (Severity: 6.0)
Skador Chelsea: 0

1: 42% | X: 28% | 2: 30%
Tips: 1 (men osäkrare)
```

Modellen justerar sannolikheterna baserat på skador, vilket ger mer realistiska prediktioner!

## Support

För frågor eller problem:
1. Öppna en issue på GitHub
2. Kontakta utvecklaren
3. Se API-Football dokumentation: https://www.api-football.com/documentation-v3

---

**Version:** 7.6.0  
**Datum:** 2026-01-16  
**Utvecklare:** Manus AI
