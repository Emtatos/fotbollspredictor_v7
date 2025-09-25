from pathlib import Path
import pandas as pd
import numpy as np
import logging


def normalize_csv_data(file_paths: list[Path]) -> pd.DataFrame:
    """
    Läser en lista av CSV-filer, normaliserar dem och returnerar en enda DataFrame.

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

    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='warn')

            # Kontrollera att nödvändiga kolumner finns
            if not all(col in df.columns for col in required_cols):
                logging.warning("Filen %s saknar en eller flera nödvändiga kolumner. Hoppar över.", file_path)
                continue

            all_dfs.append(df[required_cols])

        except Exception as e:
            logging.error("Kunde inte läsa eller bearbeta filen %s: %s", file_path, e)

    if not all_dfs:
        logging.warning("Ingen data kunde laddas från de angivna filerna.")
        return pd.DataFrame()

    # Slå ihop alla dataframes
    concatenated_df = pd.concat(all_dfs, ignore_index=True)

    # Datatvätt och standardisering
    concatenated_df['Date'] = pd.to_datetime(concatenated_df['Date'], dayfirst=True, errors='coerce')
    concatenated_df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTR'], inplace=True)

    for col in ['FTHG', 'FTAG']:
        concatenated_df[col] = pd.to_numeric(concatenated_df[col], errors='coerce')

    # Ta bort matcher där mål-data saknas efter konvertering
    concatenated_df.dropna(subset=['FTHG', 'FTAG'], inplace=True)

    # Konvertera mål till heltal
    concatenated_df[['FTHG', 'FTAG']] = concatenated_df[['FTHG', 'FTAG']].astype(int)

    logging.info(
        "Framgångsrikt laddat och normaliserat %d matcher från %d filer.",
        len(concatenated_df),
        len(all_dfs)
    )

    return concatenated_df
