"""
archive -- Postgres-/SQLite-baserat arkiv for Stryktipsomgangar.

Ersatter det filbaserade arkivet i `snapshot_storage.py` (deprecated).
Modulerna har ar rena Python-moduler utan Streamlit-beroenden sa att
samma funktioner kan anropas fran UI:t och fran ett framtida cron-jobb.

    archive.db      -- engine/session och schema (SQLAlchemy)
    archive.fetch   -- hamtfunktioner mot Svenska Spels API + skrivning
    archive.status  -- statusvy per omgang (harledd fran databasen)
    archive.legacy  -- import av aldre JSON-filer
"""
