# Changelog

Alla betydande ändringar i detta projekt dokumenteras i denna fil.

Formatet baseras på [Keep a Changelog](https://keepachangelog.com/sv/1.0.0/),
och detta projekt följer [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [7.1.0] - 2026-01-06

### Tillagt
- **Automatiserad testning**: Komplett testsvit med 42 enhetstester
  - `tests/test_utils.py`: Tester för namnnormalisering
  - `tests/test_data_processing.py`: Tester för datarensning
  - `tests/test_feature_engineering.py`: Tester för feature-skapande
  - `tests/test_ui_utils.py`: Tester för UI-logik
- **CI/CD**: GitHub Actions workflow för automatiska tester
- **Konsoliderad app**: Ny `app.py` som kombinerar det bästa från tidigare versioner
- **Centraliserad konfiguration**: `config.py` för all konfiguration
- **Säkerhetsförbättringar**:
  - `.env.example` som mall för miljövariabler
  - Uppdaterad `.gitignore` för att skydda hemligheter
  - Borttagen hårdkodad API-nyckel från `.env`
- **Dokumentation**:
  - Omfattande README.md med installationsinstruktioner
  - CHANGELOG.md för versionshistorik
  - Kodkommentarer och docstrings

### Ändrat
- **Användargränssnitt**: Förbättrad layout med flikar
  - Flik 1: Enskild match med AI-analys
  - Flik 2: Flera matcher med batch-prediktion
  - Flik 3: Om appen
- **Kodstruktur**: Mer modulär och underhållbar
- **Felhantering**: Förbättrad felhantering i alla moduler
- **Prestanda**: Optimerad caching i Streamlit

### Borttaget
- Dubbletter av applikationsfiler (backup som `*_old.py`)
- Hårdkodade API-nycklar från versionskontroll

### Säkerhet
- API-nycklar hanteras nu endast via miljövariabler
- `.env` läggs inte längre till i Git
- Förbättrad `.gitignore` för att förhindra läckage av hemligheter

## [7.0.0] - 2024-XX-XX

### Tillagt
- Initial version med grundläggande funktionalitet
- XGBoost-modell för matchprediktioner
- Stöd för E0, E1, E2 (engelska ligor)
- Form- och ELO-beräkningar
- Streamlit-baserat webbgränssnitt
- Halvgarderingar
- OpenAI-integration för matchanalys

### Känt problem
- Ingen automatiserad testning
- Dubbletter av applikationsfiler
- Hårdkodade API-nycklar i `.env`
- Generiska commit-meddelanden

---

## Versionsnumrering

Projektet använder Semantic Versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Inkompatibla API-ändringar
- **MINOR**: Nya funktioner (bakåtkompatibla)
- **PATCH**: Buggfixar (bakåtkompatibla)

## Kategorier

- **Tillagt**: Nya funktioner
- **Ändrat**: Ändringar i befintlig funktionalitet
- **Föråldrat**: Funktioner som snart tas bort
- **Borttaget**: Borttagna funktioner
- **Fixat**: Buggfixar
- **Säkerhet**: Säkerhetsrelaterade ändringar
