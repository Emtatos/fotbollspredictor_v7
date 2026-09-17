"""round archive: rounds, snapshots, played systems, results

Revision ID: 0001_round_archive
Revises:
Create Date: 2026-09-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001_round_archive"
down_revision = None
branch_labels = None
depends_on = None

_JSON = sa.JSON().with_variant(JSONB, "postgresql")
_TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "rounds",
        sa.Column("draw_number", sa.Integer, nullable=False),
        sa.Column("week_label", sa.String),
        sa.Column("reg_close_time", _TS),
        sa.Column("created_at", _TS, nullable=False),
        sa.Column("status", sa.String, nullable=False),
        sa.PrimaryKeyConstraint("draw_number", name="pk_rounds"),
    )
    op.create_table(
        "round_matches",
        sa.Column("draw_number", sa.Integer, nullable=False),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("home_team", sa.String, nullable=False),
        sa.Column("away_team", sa.String, nullable=False),
        sa.Column("home_team_canon", sa.String),
        sa.Column("away_team_canon", sa.String),
        sa.Column("league", sa.String),
        sa.ForeignKeyConstraint(
            ["draw_number"], ["rounds.draw_number"],
            name="fk_round_matches_draw_number_rounds",
        ),
        sa.PrimaryKeyConstraint(
            "draw_number", "position", name="pk_round_matches",
        ),
    )
    op.create_table(
        "market_snapshots",
        sa.Column("id", sa.Integer, nullable=False, autoincrement=True),
        sa.Column("draw_number", sa.Integer, nullable=False),
        sa.Column("captured_at", _TS, nullable=False),
        sa.Column("captured_at_precision", sa.String, nullable=False),
        sa.Column("source", sa.String, nullable=False),
        sa.Column("parser_version", sa.Integer, nullable=False),
        sa.Column("raw_payload", _JSON),
        sa.ForeignKeyConstraint(
            ["draw_number"], ["rounds.draw_number"],
            name="fk_market_snapshots_draw_number_rounds",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_market_snapshots"),
    )
    op.create_table(
        "market_snapshot_matches",
        sa.Column("snapshot_id", sa.Integer, nullable=False),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("streck_1", sa.Float),
        sa.Column("streck_x", sa.Float),
        sa.Column("streck_2", sa.Float),
        sa.Column("odds_1", sa.Float),
        sa.Column("odds_x", sa.Float),
        sa.Column("odds_2", sa.Float),
        sa.Column("startodds_1", sa.Float),
        sa.Column("startodds_x", sa.Float),
        sa.Column("startodds_2", sa.Float),
        sa.Column("favoritskap_1", sa.Float),
        sa.Column("favoritskap_x", sa.Float),
        sa.Column("favoritskap_2", sa.Float),
        sa.ForeignKeyConstraint(
            ["snapshot_id"], ["market_snapshots.id"],
            name="fk_market_snapshot_matches_snapshot_id_market_snapshots",
        ),
        sa.PrimaryKeyConstraint(
            "snapshot_id", "position", name="pk_market_snapshot_matches",
        ),
    )
    op.create_table(
        "played_systems",
        sa.Column("id", sa.Integer, nullable=False, autoincrement=True),
        sa.Column("draw_number", sa.Integer, nullable=False),
        sa.Column("created_at", _TS, nullable=False),
        sa.Column("snapshot_id", sa.Integer),
        sa.Column("n_halfguards", sa.Integer, nullable=False),
        sa.Column("note", sa.String, nullable=False),
        sa.ForeignKeyConstraint(
            ["draw_number"], ["rounds.draw_number"],
            name="fk_played_systems_draw_number_rounds",
        ),
        sa.ForeignKeyConstraint(
            ["snapshot_id"], ["market_snapshots.id"],
            name="fk_played_systems_snapshot_id_market_snapshots",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_played_systems"),
    )
    op.create_table(
        "played_system_matches",
        sa.Column("played_system_id", sa.Integer, nullable=False),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("played_signs", sa.String, nullable=False),
        sa.Column("is_halfguard", sa.Boolean, nullable=False),
        sa.Column("combined_p1", sa.Float),
        sa.Column("combined_px", sa.Float),
        sa.Column("combined_p2", sa.Float),
        sa.Column("gain", sa.Float),
        sa.Column("sources_used", sa.String),
        sa.ForeignKeyConstraint(
            ["played_system_id"], ["played_systems.id"],
            name="fk_played_system_matches_played_system_id_played_systems",
        ),
        sa.PrimaryKeyConstraint(
            "played_system_id", "position", name="pk_played_system_matches",
        ),
    )
    op.create_table(
        "results",
        sa.Column("draw_number", sa.Integer, nullable=False),
        sa.Column("fetched_at", _TS, nullable=False),
        sa.Column("correct_row", sa.String, nullable=False),
        sa.Column("turnover", sa.Float),
        sa.Column("payout_13", sa.Float),
        sa.Column("payout_12", sa.Float),
        sa.Column("payout_11", sa.Float),
        sa.Column("payout_10", sa.Float),
        sa.Column("winners_13", sa.Integer),
        sa.Column("winners_12", sa.Integer),
        sa.Column("winners_11", sa.Integer),
        sa.Column("winners_10", sa.Integer),
        sa.Column("source", sa.String, nullable=False),
        sa.Column("parser_version", sa.Integer, nullable=False),
        sa.Column("raw_payload", _JSON),
        sa.ForeignKeyConstraint(
            ["draw_number"], ["rounds.draw_number"],
            name="fk_results_draw_number_rounds",
        ),
        sa.PrimaryKeyConstraint("draw_number", name="pk_results"),
    )
    op.create_table(
        "result_matches",
        sa.Column("draw_number", sa.Integer, nullable=False),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("home_score", sa.Integer),
        sa.Column("away_score", sa.Integer),
        sa.Column("outcome", sa.String, nullable=False),
        sa.ForeignKeyConstraint(
            ["draw_number"], ["results.draw_number"],
            name="fk_result_matches_draw_number_results",
        ),
        sa.PrimaryKeyConstraint(
            "draw_number", "position", name="pk_result_matches",
        ),
    )


def downgrade() -> None:
    for name in (
        "result_matches", "results", "played_system_matches",
        "played_systems", "market_snapshot_matches", "market_snapshots",
        "round_matches", "rounds",
    ):
        op.drop_table(name)
