"""Add timeframe-aware OHLCV keys and universe-scoped feature cache columns.

Revision ID: 003_timeframe_scope
Revises: 002_hypertables
Create Date: 2026-03-12
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "003_timeframe_scope"
down_revision: Union[str, None] = "002_hypertables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema for timeframe-aware bars and universe-scoped features."""
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        raise RuntimeError("003_timeframe_scope migration requires PostgreSQL.")

    op.execute("ALTER TABLE ohlcv_bars ADD COLUMN IF NOT EXISTS timeframe VARCHAR(16)")
    op.execute("UPDATE ohlcv_bars SET timeframe = '15Min' WHERE timeframe IS NULL")
    op.execute("ALTER TABLE ohlcv_bars ALTER COLUMN timeframe SET DEFAULT '15Min'")
    op.execute("ALTER TABLE ohlcv_bars ALTER COLUMN timeframe SET NOT NULL")
    op.execute("DROP INDEX IF EXISTS ix_ohlcv_bars_symbol")
    op.execute("DROP INDEX IF EXISTS ix_ohlcv_bars_symbol_timeframe")
    op.execute("CREATE INDEX IF NOT EXISTS ix_ohlcv_bars_symbol ON ohlcv_bars (symbol)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_ohlcv_bars_symbol_timeframe "
        "ON ohlcv_bars (symbol, timeframe)"
    )
    op.execute("ALTER TABLE ohlcv_bars DROP CONSTRAINT IF EXISTS ohlcv_bars_pkey")
    op.execute(
        "ALTER TABLE ohlcv_bars "
        "ADD CONSTRAINT ohlcv_bars_pkey PRIMARY KEY (symbol, timestamp, timeframe)"
    )
    op.execute(
        """
        ALTER TABLE ohlcv_bars SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'symbol, timeframe',
            timescaledb.compress_orderby = 'timestamp DESC'
        )
        """
    )

    op.execute("ALTER TABLE features ADD COLUMN IF NOT EXISTS timeframe VARCHAR(16)")
    op.execute("ALTER TABLE features ADD COLUMN IF NOT EXISTS feature_set_id VARCHAR(64)")
    op.execute("UPDATE features SET timeframe = '15Min' WHERE timeframe IS NULL")
    op.execute(
        "UPDATE features SET feature_set_id = 'default' "
        "WHERE feature_set_id IS NULL OR btrim(feature_set_id) = ''"
    )
    op.execute("ALTER TABLE features ALTER COLUMN timeframe SET DEFAULT '15Min'")
    op.execute("ALTER TABLE features ALTER COLUMN timeframe SET NOT NULL")
    op.execute("ALTER TABLE features ALTER COLUMN feature_set_id SET DEFAULT 'default'")
    op.execute("ALTER TABLE features ALTER COLUMN feature_set_id SET NOT NULL")
    op.execute("DROP INDEX IF EXISTS ix_features_symbol_timestamp")
    op.execute("DROP INDEX IF EXISTS ix_features_symbol_feature_set")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_features_symbol_timestamp "
        "ON features (symbol, timeframe, timestamp)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_features_symbol_feature_set "
        "ON features (symbol, timeframe, feature_set_id)"
    )
    op.execute("ALTER TABLE features DROP CONSTRAINT IF EXISTS features_pkey")
    op.execute(
        "ALTER TABLE features ADD CONSTRAINT features_pkey "
        "PRIMARY KEY (symbol, timestamp, timeframe, feature_name, feature_set_id)"
    )
    op.execute(
        """
        ALTER TABLE features SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'symbol, timeframe, feature_name, feature_set_id',
            timescaledb.compress_orderby = 'timestamp DESC'
        )
        """
    )


def downgrade() -> None:
    """Revert timeframe and universe-scope columns."""
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        raise RuntimeError("003_timeframe_scope migration requires PostgreSQL.")

    op.execute("ALTER TABLE features DROP CONSTRAINT IF EXISTS features_pkey")
    op.execute(
        "ALTER TABLE features ADD CONSTRAINT features_pkey "
        "PRIMARY KEY (symbol, timestamp, feature_name)"
    )
    op.execute("DROP INDEX IF EXISTS ix_features_symbol_feature_set")
    op.execute("DROP INDEX IF EXISTS ix_features_symbol_timestamp")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_features_symbol_timestamp "
        "ON features (symbol, timestamp)"
    )
    op.execute("ALTER TABLE features DROP COLUMN IF EXISTS feature_set_id")
    op.execute("ALTER TABLE features DROP COLUMN IF EXISTS timeframe")
    op.execute(
        """
        ALTER TABLE features SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'symbol, feature_name',
            timescaledb.compress_orderby = 'timestamp DESC'
        )
        """
    )

    op.execute("ALTER TABLE ohlcv_bars DROP CONSTRAINT IF EXISTS ohlcv_bars_pkey")
    op.execute(
        "ALTER TABLE ohlcv_bars ADD CONSTRAINT ohlcv_bars_pkey PRIMARY KEY (symbol, timestamp)"
    )
    op.execute("DROP INDEX IF EXISTS ix_ohlcv_bars_symbol_timeframe")
    op.execute("CREATE INDEX IF NOT EXISTS ix_ohlcv_bars_symbol ON ohlcv_bars (symbol)")
    op.execute("ALTER TABLE ohlcv_bars DROP COLUMN IF EXISTS timeframe")
    op.execute(
        """
        ALTER TABLE ohlcv_bars SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'symbol',
            timescaledb.compress_orderby = 'timestamp DESC'
        )
        """
    )
