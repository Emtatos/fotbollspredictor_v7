from pathlib import Path
import re
import pandas as pd
import numpy as np
import logging


MATCHSTATS_COLS = [
    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
]

ODDS_COLS = [
    "B365H", "B365D", "B365A", "PSH", "PSD", "PSA",
]

_ODDS_RE = re.compile(
    r"^(B365|BW|IW|PS|WH|VC|Max|Avg|P)(H|D|A)$"
)


def normalize_csv_data(file_paths: list[Path]) -> pd.DataFrame:
    """
    Läser en lista av CSV-filer, normaliserar dem och returnerar en enda DataFrame.

    Behåller optional columns (matchstats, odds) om de finns i CSV:erna.
    Extraherar Season från filnamn (t.ex. E0_2425.csv → Season='2425').

    Parametrar
    ----------
    file_paths : list[Path]
        En lista med sökvägar till CSV-filer nedladdade från football-data.co.uk.

    Returnerar
    -------
    pd.DataFrame
        En sammanslagen och rensad DataFrame med standardiserade kolumner.
    """
    all_dfs = []
    required_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

    def _extract_league_code(p: Path) -> str:
        stem = p.stem
        league = stem.split("_")[0].strip()
        return league if league else "UNK"

    def _extract_season_code(p: Path) -> str:
        stem = p.stem
        parts = stem.split("_")
        if len(parts) >= 2:
            return parts[1].strip()
        return "UNK"

    def _discover_odds_cols(columns: pd.Index) -> list:
        found = []
        for c in columns:
            if _ODDS_RE.match(str(c)):
                found.append(c)
        return found

    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='warn')

            if not all(col in df.columns for col in required_cols):
                logging.warning("Filen %s saknar en eller flera nödvändiga kolumner. Hoppar över.", file_path)
                continue

            df = df.copy()

            league_code = _extract_league_code(Path(file_path))
            season_code = _extract_season_code(Path(file_path))
            df["League"] = league_code
            df["Season"] = season_code

            keep_cols = required_cols + ["League", "Season"]

            found_stats = [c for c in MATCHSTATS_COLS if c in df.columns]
            if found_stats:
                logging.info(
                    "Liga %s säsong %s: hittade matchstats-kolumner: %s",
                    league_code, season_code, found_stats
                )
                keep_cols += found_stats

            found_odds = [c for c in ODDS_COLS if c in df.columns]
            extra_odds = _discover_odds_cols(df.columns)
            all_odds = list(dict.fromkeys(found_odds + extra_odds))
            if all_odds:
                logging.info(
                    "Liga %s säsong %s: hittade odds-kolumner: %s",
                    league_code, season_code, all_odds
                )
                keep_cols += [c for c in all_odds if c not in keep_cols]

            keep_cols = [c for c in keep_cols if c in df.columns]
            all_dfs.append(df[keep_cols])

        except Exception as e:
            logging.error("Kunde inte läsa eller bearbeta filen %s: %s", file_path, e)

    if not all_dfs:
        logging.warning("Ingen data kunde laddas från de angivna filerna.")
        return pd.DataFrame()

    concatenated_df = pd.concat(all_dfs, ignore_index=True)

    concatenated_df['Date'] = pd.to_datetime(concatenated_df['Date'], dayfirst=True, errors='coerce')
    concatenated_df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTR'], inplace=True)

    for col in ['FTHG', 'FTAG']:
        concatenated_df[col] = pd.to_numeric(concatenated_df[col], errors='coerce')

    concatenated_df.dropna(subset=['FTHG', 'FTAG'], inplace=True)

    concatenated_df[['FTHG', 'FTAG']] = concatenated_df[['FTHG', 'FTAG']].astype(int)

    for col in MATCHSTATS_COLS:
        if col in concatenated_df.columns:
            concatenated_df[col] = pd.to_numeric(concatenated_df[col], errors='coerce')

    for col in concatenated_df.columns:
        if _ODDS_RE.match(str(col)) or col in ODDS_COLS:
            concatenated_df[col] = pd.to_numeric(concatenated_df[col], errors='coerce')

    logging.info(
        "Framgångsrikt laddat och normaliserat %d matcher från %d filer.",
        len(concatenated_df),
        len(all_dfs)
    )

    optional_present = [c for c in concatenated_df.columns if c not in required_cols + ["League", "Season"]]
    if optional_present:
        logging.info("Optional kolumner i slutgiltig DataFrame: %s", optional_present)

    return concatenated_df
