# Bidra till Fotbollspredictor v7

Tack för att du överväger att bidra till detta projekt! Alla bidrag är välkomna, oavsett om det är buggfixar, nya funktioner eller förbättringar av dokumentationen.

## Innehållsförteckning

- [Kodstil](#kodstil)
- [Utvecklingsmiljö](#utvecklingsmiljö)
- [Testning](#testning)
- [Pull Requests](#pull-requests)
- [Rapportera buggar](#rapportera-buggar)
- [Föreslå funktioner](#föreslå-funktioner)

## Kodstil

Projektet följer PEP 8-riktlinjer för Python-kod. Använd följande verktyg för att säkerställa kodkvalitet:

### Formatering

```bash
# Automatisk formatering med black
black .

# Sortera imports med isort
isort .
```

### Linting

```bash
# Kontrollera kodkvalitet med flake8
flake8 . --max-line-length=127
```

### Namnkonventioner

- **Funktioner och variabler**: `snake_case`
- **Klasser**: `PascalCase`
- **Konstanter**: `UPPER_SNAKE_CASE`
- **Privata metoder**: `_leading_underscore`

### Docstrings

Använd Google-stil docstrings:

```python
def calculate_elo(home_elo: float, away_elo: float, result: str) -> tuple[float, float]:
    """
    Beräknar nya ELO-ratings efter en match.
    
    Args:
        home_elo: Hemmalag ELO före matchen
        away_elo: Bortalag ELO före matchen
        result: Matchresultat ('H', 'D', eller 'A')
    
    Returns:
        Tuple med (ny_home_elo, ny_away_elo)
    
    Raises:
        ValueError: Om result inte är 'H', 'D' eller 'A'
    """
    # Implementation...
```

## Utvecklingsmiljö

### Sätta upp miljön

1. Forka repositoriet
2. Klona din fork:
```bash
git clone https://github.com/ditt-användarnamn/fotbollspredictor_v7.git
cd fotbollspredictor_v7
```

3. Skapa en virtuell miljö:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

4. Installera beroenden:
```bash
pip install -r requirements.txt
pip install black isort flake8 pre-commit
```

5. Installera pre-commit hooks:
```bash
pre-commit install
```

### Branching-strategi

- `main`: Stabil produktionskod
- `develop`: Utvecklingsbranch
- `feature/namn`: Nya funktioner
- `fix/namn`: Buggfixar
- `docs/namn`: Dokumentationsändringar

Skapa alltid en ny branch för dina ändringar:

```bash
git checkout -b feature/min-nya-funktion
```

## Testning

Alla nya funktioner och buggfixar måste ha tillhörande tester.

### Skriva tester

Placera tester i `tests/`-mappen. Använd pytest-konventioner:

```python
# tests/test_min_modul.py
import pytest
from min_modul import min_funktion


class TestMinFunktion:
    """Tester för min_funktion"""
    
    def test_grundläggande_funktionalitet(self):
        """Testar grundläggande användning"""
        result = min_funktion(input_data)
        assert result == expected_output
    
    def test_felhantering(self):
        """Testar felhantering"""
        with pytest.raises(ValueError):
            min_funktion(invalid_input)
```

### Köra tester

```bash
# Alla tester
pytest tests/ -v

# Specifik testfil
pytest tests/test_utils.py -v

# Med coverage
pytest tests/ --cov=. --cov-report=html
```

### Test-coverage

Sträva efter minst 80% coverage för ny kod. Kontrollera coverage:

```bash
pytest tests/ --cov=. --cov-report=term
```

## Pull Requests

### Innan du skickar en PR

1. **Kör testerna**: Säkerställ att alla tester passerar
```bash
pytest tests/ -v
```

2. **Kontrollera kodkvalitet**:
```bash
black --check .
isort --check-only .
flake8 .
```

3. **Uppdatera dokumentation**: Om du lägger till nya funktioner, uppdatera README.md

4. **Commit-meddelanden**: Skriv beskrivande commit-meddelanden
```bash
git commit -m "Lägg till ELO-beräkning för bättre prediktioner"
```

### PR-process

1. Push din branch till din fork:
```bash
git push origin feature/min-nya-funktion
```

2. Öppna en Pull Request på GitHub

3. Fyll i PR-mallen med:
   - Beskrivning av ändringarna
   - Relaterade issues (om tillämpligt)
   - Screenshots (för UI-ändringar)
   - Checklista för granskare

4. Vänta på code review

5. Adressera feedback från granskare

6. När PR är godkänd kommer den att mergas

### PR-mall

```markdown
## Beskrivning
Kort beskrivning av ändringarna

## Typ av ändring
- [ ] Buggfix
- [ ] Ny funktion
- [ ] Breaking change
- [ ] Dokumentation

## Hur har detta testats?
Beskriv hur du har testat ändringarna

## Checklista
- [ ] Koden följer projektets kodstil
- [ ] Jag har lagt till tester
- [ ] Alla tester passerar
- [ ] Jag har uppdaterat dokumentationen
- [ ] Inga nya varningar introduceras
```

## Rapportera buggar

Använd GitHub Issues för att rapportera buggar. Inkludera:

### Buggrapport-mall

```markdown
**Beskrivning av buggen**
En tydlig beskrivning av vad buggen är.

**Steg för att återskapa**
1. Gå till '...'
2. Klicka på '...'
3. Se felet

**Förväntat beteende**
Vad du förväntade dig skulle hända.

**Faktiskt beteende**
Vad som faktiskt hände.

**Screenshots**
Om tillämpligt, lägg till screenshots.

**Miljö**
- OS: [t.ex. Windows 10, macOS 13]
- Python-version: [t.ex. 3.11]
- Projektversion: [t.ex. 7.1.0]

**Ytterligare kontext**
Annan relevant information.
```

## Föreslå funktioner

Använd GitHub Issues för att föreslå nya funktioner. Inkludera:

### Feature request-mall

```markdown
**Är din feature request relaterad till ett problem?**
En tydlig beskrivning av problemet. Ex: "Jag blir frustrerad när..."

**Beskriv lösningen du vill ha**
En tydlig beskrivning av vad du vill ska hända.

**Beskriv alternativ du har övervägt**
Andra lösningar eller funktioner du har övervägt.

**Ytterligare kontext**
Annan relevant information, screenshots, etc.
```

## Kod av uppförande

### Våra förväntningar

- Var respektfull och inkluderande
- Acceptera konstruktiv kritik
- Fokusera på vad som är bäst för projektet
- Visa empati mot andra bidragsgivare

### Oacceptabelt beteende

- Trakasserier eller diskriminerande kommentarer
- Trolling eller nedsättande kommentarer
- Offentlig eller privat trakassering
- Publicering av andras privata information

## Frågor?

Om du har frågor, öppna en issue på GitHub eller kontakta projektägaren.

---

Tack för att du bidrar till Fotbollspredictor v7! 🎉
