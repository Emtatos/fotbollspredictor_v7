# ⚽ Fotbollspredictor v7

En maskininlärningsbaserad applikation för att förutsäga utfall i engelska fotbollsmatcher från Premier League (E0), Championship (E1) och League One (E2).

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Tests](https://img.shields.io/badge/tests-42%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

## 📋 Innehållsförteckning

- [Funktioner](#-funktioner)
- [Installation](#-installation)
- [Användning](#-användning)
- [Arkitektur](#-arkitektur)
- [Testning](#-testning)
- [Deployment](#-deployment)
- [Utveckling](#-utveckling)
- [Bidra](#-bidra)

## ✨ Funktioner

### Kärnfunktionalitet
- **Maskininlärning**: XGBoost-klassificerare tränad på historisk matchdata
- **Tre ligor**: Premier League, Championship och League One
- **Prediktioner**: Sannolikheter för hemmavinst (1), oavgjort (X) och bortavinst (2)
- **Halvgarderingar**: Intelligent val av osäkra matcher för dubbla tips

### Features
- **Form**: Beräknas från de senaste 5 matcherna (poäng och målskillnad)
- **ELO-rating**: Dynamisk styrkerating som uppdateras efter varje match
- **Statistik**: Detaljerad matchstatistik och jämförelser

### Avancerade funktioner
- **AI-analys**: OpenAI-driven matchanalys (valfritt)
- **Batch-prediktion**: Tippa flera matcher samtidigt
- **Interaktivt gränssnitt**: Webbaserad app byggd med Streamlit

## 🚀 Installation

### Förutsättningar
- Python 3.9 eller senare
- pip (Python package manager)

### Steg-för-steg

1. **Klona repositoriet**
```bash
git clone https://github.com/Emtatos/fotbollspredictor_v7.git
cd fotbollspredictor_v7
```

2. **Skapa virtuell miljö (rekommenderas)**
```bash
python -m venv venv
source venv/bin/activate  # På Windows: venv\Scripts\activate
```

3. **Installera beroenden**
```bash
pip install -r requirements.txt
```

4. **Konfigurera miljövariabler**
```bash
cp .env.example .env
# Redigera .env och lägg till dina API-nycklar
```

### API-nycklar (valfritt)

- **API-Football**: För att hämta live-fixtures från [api-football.com](https://www.api-football.com/)
- **OpenAI**: För AI-analys från [platform.openai.com](https://platform.openai.com/)

## 📖 Användning

### Träna modellen

Kör pipelinen för att hämta data och träna modellen:

```bash
python main.py
```

Detta kommer att:
1. Hämta data från football-data.co.uk
2. Bearbeta och normalisera datan
3. Skapa features (form och ELO)
4. Träna XGBoost-modellen
5. Spara modellen till `models/`

### Starta webbapplikationen

```bash
streamlit run app.py
```

Öppna din webbläsare på `http://localhost:8501`

### Använda applikationen

#### Enskild match
1. Välj hemmalag och bortalag från dropdown-menyerna
2. Välj om du vill ha halvgardering
3. Klicka på "Tippa Match"
4. Se sannolikheter, tips och statistik

#### Flera matcher
1. Gå till fliken "Flera Matcher"
2. Skriv in matcher (en per rad): `Hemmalag - Bortalag`
3. Välj antal halvgarderingar
4. Klicka på "Tippa Alla Matcher"
5. Kopiera tipsraden

## 🏗️ Arkitektur

### Projektstruktur

```
fotbollspredictor_v7/
├── app.py                      # Huvudapplikation (Streamlit)
├── main.py                     # Pipeline för databehandling och träning
├── config.py                   # Centraliserad konfiguration
├── data_loader.py              # Datahämtning från externa källor
├── data_processing.py          # Datarensning och normalisering
├── feature_engineering.py      # Feature-skapande (form, ELO)
├── model_handler.py            # Modellträning och hantering
├── utils.py                    # Hjälpfunktioner (namnnormalisering)
├── ui_utils.py                 # UI-specifika hjälpfunktioner
├── tests/                      # Testsvit
│   ├── test_utils.py
│   ├── test_data_processing.py
│   ├── test_feature_engineering.py
│   └── test_ui_utils.py
├── data/                       # Data (gitignorerad)
├── models/                     # Tränade modeller (gitignorerad)
├── requirements.txt            # Python-beroenden
├── pytest.ini                  # Pytest-konfiguration
├── .env.example                # Exempelmiljövariabler
├── .gitignore                  # Git-ignorerade filer
└── README.md                   # Denna fil
```

### Dataflöde

```
1. Datahämtning (data_loader.py)
   ↓
2. Datarensning (data_processing.py)
   ↓
3. Feature Engineering (feature_engineering.py)
   ↓
4. Modellträning (model_handler.py)
   ↓
5. Prediktion (app.py)
```

### Teknisk stack

- **Frontend**: Streamlit
- **ML**: XGBoost, scikit-learn
- **Data**: pandas, numpy
- **HTTP**: requests
- **Testning**: pytest
- **AI**: OpenAI (valfritt)

## 🧪 Testning

Projektet har en omfattande testsvit med 42 enhetstester.

### Kör alla tester

```bash
pytest tests/ -v
```

### Kör tester med coverage

```bash
pytest tests/ --cov=. --cov-report=html
```

Öppna `htmlcov/index.html` för att se coverage-rapport.

### Kör specifika tester

```bash
pytest tests/test_utils.py -v
pytest tests/test_feature_engineering.py::TestCreateFeatures::test_elo_initialization -v
```

### Teststruktur

- **test_utils.py**: Tester för namnnormalisering och hjälpfunktioner
- **test_data_processing.py**: Tester för datarensning och validering
- **test_feature_engineering.py**: Tester för form- och ELO-beräkningar
- **test_ui_utils.py**: Tester för UI-logik och halvgarderingar

## 🚢 Deployment

### Render.com

Projektet är konfigurerat för deployment på Render med `render.yaml`.

1. Skapa ett konto på [render.com](https://render.com)
2. Anslut ditt GitHub-repo
3. Render kommer automatiskt att upptäcka `render.yaml`
4. Lägg till miljövariabler i Render Dashboard:
   - `API_FOOTBALL_KEY`
   - `OPENAI_API_KEY`

### Docker (kommande)

En Dockerfile kommer att läggas till för containeriserad deployment.

## 👨‍💻 Utveckling

### Kodstil

Projektet följer PEP 8-riktlinjer. Använd dessa verktyg:

```bash
# Formatera kod
black .

# Sortera imports
isort .

# Linting
flake8 .
```

### Pre-commit hooks (rekommenderas)

```bash
pip install pre-commit
pre-commit install
```

### Bidra med ny funktionalitet

1. Skapa en ny branch: `git checkout -b feature/ny-funktion`
2. Implementera funktionen
3. Skriv tester
4. Kör testsviten: `pytest tests/ -v`
5. Commit: `git commit -m "Lägg till ny funktion"`
6. Push: `git push origin feature/ny-funktion`
7. Skapa en Pull Request

### Commit-meddelanden

Använd beskrivande commit-meddelanden:

- ✅ `Lägg till ELO-beräkning för bättre prediktioner`
- ✅ `Fixa bug i namnnormalisering för Sheffield-lag`
- ❌ `Update utils.py`
- ❌ `Fix`

## 🐛 Felsökning

### Vanliga problem

**Problem**: `ModuleNotFoundError: No module named 'streamlit'`
**Lösning**: Kör `pip install -r requirements.txt`

**Problem**: Modellen saknas
**Lösning**: Kör `python main.py` för att träna modellen

**Problem**: Inga lag visas i dropdown
**Lösning**: Kontrollera att data finns i `data/features.parquet`

**Problem**: API-nycklar fungerar inte
**Lösning**: Kontrollera att `.env` finns och innehåller giltiga nycklar

## 📊 Prestanda

- **Träning**: ~30 sekunder för 3 ligor (ca 1500 matcher)
- **Prediktion**: <100ms per match
- **Testsvit**: <1 sekund för 42 tester

## 📝 Licens

Detta projekt är licensierat under MIT-licensen - se LICENSE-filen för detaljer.

## 🙏 Erkännanden

- Data från [football-data.co.uk](https://www.football-data.co.uk/)
- API från [api-football.com](https://www.api-football.com/)
- AI från [OpenAI](https://openai.com/)

## 📧 Kontakt

För frågor eller feedback, öppna en issue på GitHub.

---

**Utvecklad med ❤️ av Emtatos**


## Backtest Report

Kör en walk-forward backtest för att utvärdera modellens prestanda.

### Användning med cache (default)

Scriptet använder lokal cache som default och laddar **inte** ner data automatiskt:

```bash
python backtest_report.py
```

### Uppdatera data

För att ladda ner färsk data, använd `--refresh-data` flaggan eller miljövariabeln:

```bash
python backtest_report.py --refresh-data
# eller
BACKTEST_REFRESH_DATA=1 python backtest_report.py
```

Cache-mapp: `data/cache/`

### Metrics

Rapporten genererar:
- **accuracy_top1**: Top-1 accuracy (argmax-prediktion)
- **accuracy_top2_on_halfguards**: Top-2 accuracy på entropy-valda halvgarderingar
- **combined_ticket_hit_rate**: Kombinerad träffprocent (top1 + top2 för HG)
- **logloss**: Multiclass log loss
- **brier**: Multiclass Brier score
- Per-liga breakdown för accuracy och logloss

## Training vs Inference Contract

- `schema.py` innehåller `FEATURE_COLUMNS` som är **single source of truth**.
- Alla prediktioner ska gå via `inference.predict_match()`.
- `state.build_current_team_states()` används för att ta fram aktuellt lagläge (inte senaste matchrad).
