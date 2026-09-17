"""
archive/db.py -- databasschema och anslutning for omgangsarkivet.

Produktion: Postgres via `DATABASE_URL` (satts av Render). Tester och lokal
utveckling: SQLite via samma kodvag. Ingen kod har forutsatter en specifik
dialekt; `raw_payload` anvander JSON portabelt med JSONB som
Postgres-variant.

Schemat skapas i produktion av Alembic (`alembic upgrade head`), inte av
`create_all()`. `create_schema()` finns for tester och lokal SQLite.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
)
from sqlalchemy import event
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import Session, sessionmaker

DATABASE_URL_ENV = "DATABASE_URL"

PARSER_VERSION = 1

ROUND_STATUS_OPEN = "open"
ROUND_STATUS_CLOSED = "closed"
ROUND_STATUS_FINALIZED = "finalized"

PRECISION_EXACT = "exact"
PRECISION_DATE_ONLY = "date_only"
PRECISION_UNKNOWN = "unknown"
CAPTURED_AT_PRECISIONS = (
    PRECISION_EXACT, PRECISION_DATE_ONLY, PRECISION_UNKNOWN,
)

SOURCE_API = "api"
SOURCE_IMAGE_SCAN = "image_scan"
SOURCE_PASTE = "paste"
SOURCE_MANUAL = "manual"
SOURCE_LEGACY_IMPORT = "legacy_import"
SNAPSHOT_SOURCES = (
    SOURCE_API, SOURCE_IMAGE_SCAN, SOURCE_PASTE, SOURCE_MANUAL,
    SOURCE_LEGACY_IMPORT,
)
RESULT_SOURCES = (SOURCE_API, SOURCE_MANUAL, SOURCE_LEGACY_IMPORT)

MATCH_COUNT = 13

# Namngivningskonvention sa att Alembic far deterministiska constraint-namn
# i bada dialekterna.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)

PortableJSON = JSON().with_variant(JSONB, "postgresql")
TimestampTZ = DateTime(timezone=True)


class DatabaseNotConfigured(RuntimeError):
    """`DATABASE_URL` saknas. UI:t ska visa felet, inte falla tillbaka."""


rounds = Table(
    "rounds", metadata,
    Column("draw_number", Integer, primary_key=True),
    Column("week_label", String),
    Column("reg_close_time", TimestampTZ),
    Column("created_at", TimestampTZ, nullable=False),
    Column("status", String, nullable=False),
)

round_matches = Table(
    "round_matches", metadata,
    Column(
        "draw_number", Integer,
        ForeignKey("rounds.draw_number"), primary_key=True,
    ),
    Column("position", Integer, primary_key=True),
    Column("home_team", String, nullable=False),
    Column("away_team", String, nullable=False),
    Column("home_team_canon", String),
    Column("away_team_canon", String),
    Column("league", String),
)

market_snapshots = Table(
    "market_snapshots", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "draw_number", Integer,
        ForeignKey("rounds.draw_number"), nullable=False,
    ),
    Column("captured_at", TimestampTZ, nullable=False),
    Column("captured_at_precision", String, nullable=False),
    Column("source", String, nullable=False),
    Column("parser_version", Integer, nullable=False),
    Column("raw_payload", PortableJSON),
)

market_snapshot_matches = Table(
    "market_snapshot_matches", metadata,
    Column(
        "snapshot_id", Integer,
        ForeignKey("market_snapshots.id"), primary_key=True,
    ),
    Column("position", Integer, primary_key=True),
    Column("streck_1", Float),
    Column("streck_x", Float),
    Column("streck_2", Float),
    Column("odds_1", Float),
    Column("odds_x", Float),
    Column("odds_2", Float),
    Column("startodds_1", Float),
    Column("startodds_x", Float),
    Column("startodds_2", Float),
    Column("favoritskap_1", Float),
    Column("favoritskap_x", Float),
    Column("favoritskap_2", Float),
)

played_systems = Table(
    "played_systems", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "draw_number", Integer,
        ForeignKey("rounds.draw_number"), nullable=False,
    ),
    Column("created_at", TimestampTZ, nullable=False),
    Column("snapshot_id", Integer, ForeignKey("market_snapshots.id")),
    Column("n_halfguards", Integer, nullable=False),
    Column("note", String, nullable=False, default=""),
)

played_system_matches = Table(
    "played_system_matches", metadata,
    Column(
        "played_system_id", Integer,
        ForeignKey("played_systems.id"), primary_key=True,
    ),
    Column("position", Integer, primary_key=True),
    Column("played_signs", String, nullable=False),
    Column("is_halfguard", Boolean, nullable=False),
    Column("combined_p1", Float),
    Column("combined_px", Float),
    Column("combined_p2", Float),
    Column("gain", Float),
    Column("sources_used", String),
)

results = Table(
    "results", metadata,
    Column(
        "draw_number", Integer,
        ForeignKey("rounds.draw_number"), primary_key=True,
    ),
    Column("fetched_at", TimestampTZ, nullable=False),
    Column("correct_row", String, nullable=False),
    Column("turnover", Float),
    Column("payout_13", Float),
    Column("payout_12", Float),
    Column("payout_11", Float),
    Column("payout_10", Float),
    Column("winners_13", Integer),
    Column("winners_12", Integer),
    Column("winners_11", Integer),
    Column("winners_10", Integer),
    Column("source", String, nullable=False),
    Column("parser_version", Integer, nullable=False),
    Column("raw_payload", PortableJSON),
)

result_matches = Table(
    "result_matches", metadata,
    Column(
        "draw_number", Integer,
        ForeignKey("results.draw_number"), primary_key=True,
    ),
    Column("position", Integer, primary_key=True),
    Column("home_score", Integer),
    Column("away_score", Integer),
    Column("outcome", String, nullable=False),
)


# ---------------------------------------------------------------------------
# Anslutning
# ---------------------------------------------------------------------------

_engine: Optional[Engine] = None
_engine_url: Optional[str] = None


def normalize_database_url(url: str) -> str:
    """Render ger `postgres://`; SQLAlchemy 2 kraver `postgresql://`."""
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://"):]
    return url


def database_url() -> Optional[str]:
    """`DATABASE_URL` fran miljon, normaliserad, eller None."""
    raw = os.environ.get(DATABASE_URL_ENV, "").strip()
    return normalize_database_url(raw) if raw else None


def make_engine(url: str) -> Engine:
    """Skapar en engine for given URL (SQLite eller Postgres)."""
    kwargs = {"future": True}
    is_sqlite = url.startswith("sqlite")
    if is_sqlite:
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url or url in ("sqlite://", "sqlite:///"):
            kwargs["poolclass"] = StaticPool
    else:
        kwargs["pool_pre_ping"] = True
    engine = create_engine(url, **kwargs)
    if is_sqlite:
        @event.listens_for(engine, "connect")
        def _enable_sqlite_fks(dbapi_connection, _record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    return engine


def get_engine(url: Optional[str] = None) -> Engine:
    """
    Delad engine. Utan `url` anvands `DATABASE_URL`; saknas den kastas
    DatabaseNotConfigured -- ingen tyst fallback till fil eller SQLite.
    """
    global _engine, _engine_url
    target = normalize_database_url(url) if url else database_url()
    if not target:
        raise DatabaseNotConfigured(
            f"Miljovariabeln {DATABASE_URL_ENV} saknas. Arkivet kraver en "
            "databas (Render Postgres i produktion)."
        )
    if _engine is None or _engine_url != target:
        if _engine is not None:
            _engine.dispose()
        _engine = make_engine(target)
        _engine_url = target
    return _engine


def set_engine(engine: Optional[Engine]) -> None:
    """Byter delad engine (tester)."""
    global _engine, _engine_url
    _engine = engine
    _engine_url = str(engine.url) if engine is not None else None


def create_schema(engine: Engine) -> None:
    """Skapar schemat direkt (tester/lokal SQLite). Produktion: Alembic."""
    metadata.create_all(engine)


@contextmanager
def session_scope(engine: Optional[Engine] = None) -> Iterator[Session]:
    """Transaktion: commit vid lyckat block, rollback vid undantag."""
    target = engine if engine is not None else get_engine()
    factory = sessionmaker(bind=target, future=True)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
