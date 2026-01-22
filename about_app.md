## Om Fotbollspredictor v7.6

Fotbollspredictor v7.6 är en avancerad maskininlärningsapplikation designad för att prediktera fotbollsmatcher med hög noggrannhet. Appen kombinerar statistisk analys med realtidsdata för att ge insiktsfulla och datadrivna förutsägelser.

---

### 🧠 Hur fungerar modellen?

Modellen använder en **XGBoost-algoritm** (Extreme Gradient Boosting), en kraftfull och beprövad metod för prediktiv modellering. Den tränas på tusentals historiska matcher från Premier League, Championship och League One.

#### Features (27 totalt)

Modellen analyserar **27 olika features** för varje match. Dessa kan delas in i fem huvudkategorier:

| Kategori | Antal | Exempel på features |
|:---|:---|:---|
| **Form** | 6 | Genomsnittlig poäng, målskillnad (senaste 5 matcher) |
| **Målstatistik** | 4 | Genomsnitt gjorda/insläppta mål |
| **Momentum** | 2 | Vinst/förlust-streak |
| **Head-to-Head** | 4 | Tidigare möten mellan lagen |
| **Styrka & Position** | 5 | ELO-rating, ligaposition |
| **Mänsklig påverkan** | 6 | Skador, suspenderingar, nyckelspelare borta |

#### Nytt i v7.6: Mänsklig påverkan

Den senaste versionen integrerar **skador och suspenderingar** via API-Football. Detta ger en mer realistisk bild av lagens aktuella styrka.

- **Datakälla:** API-Football (uppdateras dagligen)
- **Nya features:** Antal skadade, antal nyckelspelare borta, allvarlighetsgrad (0-10)
- **Användning:** Klicka "Uppdatera skador & form" i sidomenyn för att hämta färsk data.

---

### 🎯 Funktioner i appen

- **Enskild match-prediktion:** Analysera en specifik match i detalj.
- **Flera matcher:** Tippa en hel omgång samtidigt.
- **Halvgarderingar:** Få förslag på vilka matcher som är mest osäkra.
- **AI-analys (valfritt):** OpenAI-driven textanalys av matchen.
- **On-demand data-uppdatering:** Hämta färsk skadedata med en knapptryckning.
- **Automatisk omträning:** Träna om modellen med den senaste datan.

---

### 🚀 Framtida förbättringsmöjligheter

För att ytterligare förbättra noggrannheten finns flera spännande möjligheter:

| Förbättring | Beskrivning | Potentiell påverkan |
|:---|:---|:---|
| **Tränarbyte** | Implementera "new manager bounce"-effekten. | Hög |
| **Spelarbetyg** | Använd individuell spelarform istället för bara lagform. | Hög |
| **Vilodagar** | Analysera hur tätt matchschema påverkar prestation. | Medel |
| **Väder** | Ta hänsyn till väderförhållanden (regn, vind, etc.). | Låg-Medel |
| **Historisk skadedata** | Träna modellen på historisk skadedata, inte bara aktuell. | Hög |
| **Live-odds** | Jämför modellens prediktioner med live-odds från spelbolag. | Medel |
| **Avancerad H2H** | Analysera taktiska mönster i tidigare möten. | Medel |

---

### 📊 Teknisk Stack

- **Frontend:** Streamlit
- **Backend:** Python
- **ML-modell:** XGBoost, scikit-learn
- **Datahantering:** pandas, numpy, pyarrow
- **API-integration:** requests, python-dotenv
- **Testning:** pytest, pytest-cov (46 tester)
- **Deployment:** Render, Docker

### 🔧 Utveckling & Kvalitet

Projektet följer moderna best practices:
- **Modulär arkitektur:** Lätt att underhålla och bygga ut.
- **Automatiserad testning:** 42 enhetstester och 4 integrationstester.
- **Prestandaoptimering:** 5-10x snabbare feature engineering.
- **CI/CD-redo:** Automatisk deployment via GitHub och Render.
- **Säkerhet:** API-nycklar hanteras via miljövariabler.

---

### 📝 Version

**v7.6.0** - "Human Impact" Edition

### 🐛 Felsökning

Om du stöter på problem:
1. **Uppdatera skadedata:** Klicka "Uppdatera skador & form" i sidomenyn.
2. **Kör omträning:** Klicka "Kör omträning av modell".
3. **Kontrollera API-nyckel:** Verifiera att `API_FOOTBALL_KEY` är korrekt i Render.
4. **Se loggar:** Kolla loggarna i Render Dashboard för felmeddelanden.

---

Utvecklad av **Manus AI** på uppdrag av **Emtatos**.
