import os
import time
from pathlib import Path
import requests
import logging

# Konfigurera en enkel logger för modulen
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Konstanter ===
DATA_DIR = Path("data")
BASE_URL = "https://www.football-data.co.uk/mmz4281"


def _http_get(url: str, session: requests.Session, timeout: float = 10.0) -> bytes | None:
    """
    Privat hjälpfunktion för robusta GET-anrop.

    Parametrar
    ----------
    url : str
        Fullständig URL att hämta.
    session : requests.Session
        En delad session (med ev. HTTPAdapter/Retry konfigurerad av anroparen).
    timeout : float, valfritt
        Timeout i sekunder per försök (default 10.0).

    Returnerar
    ----------
    bytes | None
        Sidans innehåll (response.content) om lyckat anrop, annars None.
    """
    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.content
    except requests.RequestException as exc:
        logging.error("HTTP GET misslyckades: %s | Fel: %s", url, exc)
        return None


def download_season_data(season_code: str, leagues: list[str]) -> list[Path]:
    """
    Ladda ner CSV-data för en säsong och en uppsättning ligor från football-data.co.uk.

    Parametrar
    ----------
    season_code : str
        Säsongskod enligt mmz4281-konventionen, t.ex. '2425'.
    leagues : list[str]
        Lista med ligakoder, t.ex. ['E0', 'E1', 'E2'].

    Returnerar
    ----------
    list[Path]
        Lista med paths till filer som laddades ner framgångsrikt och sparades i DATA_DIR.
    """
    # a) Se till att data-mappen finns
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # b) Skapa session + HTTPAdapter med Retry
    session = requests.Session()
    try:
        # Använd endast requests (utan extra imports):
        # - HTTPAdapter via requests.adapters.HTTPAdapter
        # - Retry via requests.packages.urllib3.util.retry.Retry
        HTTPAdapter = requests.adapters.HTTPAdapter
        Retry = requests.packages.urllib3.util.retry.Retry  # type: ignore[attr-defined]

        retry_strategy = Retry(
            total=3,                    # totalt antal omförsök
            backoff_factor=1.0,         # exponential backoff: 1s, 2s, 4s ...
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"])
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
    except Exception as exc:
        # Om något oväntat händer med Retry—fortsätt utan (men logga)
        logging.warning("Kunde inte konfigurera Retry/HTTPAdapter: %s. Fortsätter utan retry-adapter.", exc)

    saved_paths: list[Path] = []

    # c) Iterera över ligor och hämta filer
    for league in leagues:
        url = f"{BASE_URL}/{season_code}/{league}.csv"
        logging.info("Hämtar: %s", url)

        content = _http_get(url, session=session, timeout=10.0)
        if content is None:
            logging.warning("Nedladdning misslyckades för liga %s (season %s).", league, season_code)
            continue

        filename = f"{league}_{season_code}.csv"
        out_path = DATA_DIR / filename
        try:
            with open(out_path, "wb") as f:
                f.write(content)
            logging.info("Sparade fil: %s", out_path)
            saved_paths.append(out_path)
        except OSError as exc:
            logging.error("Kunde inte spara filen %s: %s", out_path, exc)

        # Liten paus är ibland vänlig mot källan (och spelar bra med backoff)
        time.sleep(0.1)

    # d) Returnera lista med sparade paths
    return saved_paths
