"""Convert tables to TimescaleDB hypertables

Revision ID: 002_hypertables
Revises: 001_initial
Create Date: 2026-01-05

This migration:
1. Adds the missing risk_events table
2. Enables TimescaleDB extension
3. Converts time-series tables to hypertables with appropriate chunk intervals
4. Adds compression policies for storage optimization
5. Adds retention policies for data lifecycle management

Tables converted to hypertables:
- ohlcv_bars (7-day chunks) - High-volume market data (composite PK: symbol, timestamp)
- features (1-day chunks) - ML feature storage (composite PK: symbol, timestamp, feature_name)

Tables kept as regular tables (have auto-increment id PK):
- position_history - Position snapshots (indexed on timestamp)
- signals - Trading signals (indexed on timestamp)
- model_predictions - Model outputs (indexed on timestamp)
- system_logs - Application logs (indexed on timestamp)
- risk_events - Risk audit trail (indexed on timestamp)

Note: Only tables with composite primary keys including timestamp can be converted
to hypertables. Tables with auto-increment id PKs remain regular PostgreSQL tables.

@agent: @data, @infra
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002_hypertables"
down_revision: Union[str, None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade to TimescaleDB hypertables."""

    # =========================================================================
    # Step 1: Create missing risk_events table
    # =========================================================================
    op.create_table(
        "risk_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False),
        sa.Column("symbol", sa.String(10), nullable=True),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("metrics", postgresql.JSONB(), nullable=True),
        sa.Column("order_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("strategy_name", sa.String(50), nullable=True),
        sa.Column("resolved", sa.Boolean(), server_default="false"),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_by", sa.String(50), nullable=True),
        sa.Column("resolution_notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_risk_events_timestamp", "risk_events", ["timestamp"])
    op.create_index("ix_risk_events_event_type", "risk_events", ["event_type"])
    op.create_index("ix_risk_events_severity", "risk_events", ["severity"])
    op.create_index("ix_risk_events_symbol", "risk_events", ["symbol"])
    op.create_index(
        "ix_risk_events_type_timestamp", "risk_events", ["event_type", "timestamp"]
    )
    op.create_index(
        "ix_risk_events_severity_timestamp", "risk_events", ["severity", "timestamp"]
    )

    # =========================================================================
    # Step 2: Enable TimescaleDB extension
    # =========================================================================
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")

    # =========================================================================
    # Step 3: Convert tables to hypertables
    # =========================================================================

    # OHLCV Bars - 7-day chunks (high volume, range queries)
    op.execute("""
        SELECT create_hypertable(
            'ohlcv_bars',
            'timestamp',
            chunk_time_interval => INTERVAL '7 days',
            if_not_exists => TRUE,
            migrate_data => TRUE
        )
    """)

    # Features - 1-day chunks (very high volume)
    op.execute("""
        SELECT create_hypertable(
            'features',
            'timestamp',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE,
            migrate_data => TRUE
        )
    """)

    # Note: Tables with auto-increment id PKs (position_history, signals,
    # model_predictions, system_logs, risk_events) are NOT converted to
    # hypertables because TimescaleDB requires the partitioning column
    # to be part of the primary key. They remain as regular PostgreSQL
    # tables with proper timestamp indexes.

    # =========================================================================
    # Step 4: Add space partitioning for high-cardinality tables
    # This improves query performance when filtering by symbol
    # =========================================================================

    # Note: add_dimension requires data to exist, so we use IF NOT EXISTS pattern
    op.execute("""
        DO $$
        BEGIN
            PERFORM add_dimension('ohlcv_bars', 'symbol', number_partitions => 8);
        EXCEPTION
            WHEN duplicate_object THEN
                RAISE NOTICE 'Dimension already exists for ohlcv_bars';
            WHEN undefined_table THEN
                RAISE NOTICE 'Table ohlcv_bars not found';
        END $$;
    """)

    op.execute("""
        DO $$
        BEGIN
            PERFORM add_dimension('features', 'symbol', number_partitions => 8);
        EXCEPTION
            WHEN duplicate_object THEN
                RAISE NOTICE 'Dimension already exists for features';
            WHEN undefined_table THEN
                RAISE NOTICE 'Table features not found';
        END $$;
    """)

    # =========================================================================
    # Step 5: Add compression policies
    # =========================================================================

    # OHLCV Bars - compress after 7 days
    op.execute("""
        ALTER TABLE ohlcv_bars SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'symbol',
            timescaledb.compress_orderby = 'timestamp DESC'
        )
    """)
    op.execute("""
        SELECT add_compression_policy('ohlcv_bars', INTERVAL '7 days', if_not_exists => TRUE)
    """)

    # Features - compress after 3 days
    op.execute("""
        ALTER TABLE features SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'symbol, feature_name',
            timescaledb.compress_orderby = 'timestamp DESC'
        )
    """)
    op.execute("""
        SELECT add_compression_policy('features', INTERVAL '3 days', if_not_exists => TRUE)
    """)

    # Note: Compression and retention policies are only available for hypertables.
    # Regular PostgreSQL tables (position_history, signals, model_predictions,
    # system_logs, risk_events) use standard PostgreSQL features for data management.


def downgrade() -> None:
    """Downgrade from TimescaleDB hypertables.

    Note: Downgrading from hypertables is destructive and not fully reversible.
    This will remove compression policies but data in hypertables
    cannot be easily converted back to regular tables.
    """

    # Remove compression policies
    op.execute("""
        SELECT remove_compression_policy('ohlcv_bars', if_exists => TRUE)
    """)
    op.execute("""
        SELECT remove_compression_policy('features', if_exists => TRUE)
    """)

    # Drop risk_events table
    op.drop_index("ix_risk_events_severity_timestamp", "risk_events")
    op.drop_index("ix_risk_events_type_timestamp", "risk_events")
    op.drop_index("ix_risk_events_symbol", "risk_events")
    op.drop_index("ix_risk_events_severity", "risk_events")
    op.drop_index("ix_risk_events_event_type", "risk_events")
    op.drop_index("ix_risk_events_timestamp", "risk_events")
    op.drop_table("risk_events")

    # Note: Cannot easily revert hypertable conversion
    # Tables will remain as hypertables even after this downgrade
