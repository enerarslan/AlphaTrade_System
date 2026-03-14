"""Add immutable earnings availability timestamps.

Revision ID: 004_earnings_availability
Revises: 003_timeframe_scope
Create Date: 2026-03-14
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "004_earnings_availability"
down_revision: Union[str, None] = "003_timeframe_scope"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add first-seen and availability timestamps for earnings events."""
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        raise RuntimeError("004_earnings_availability migration requires PostgreSQL.")

    op.execute(
        "ALTER TABLE earnings_events ADD COLUMN IF NOT EXISTS announcement_timestamp TIMESTAMPTZ"
    )
    op.execute(
        "ALTER TABLE earnings_events ADD COLUMN IF NOT EXISTS availability_timestamp TIMESTAMPTZ"
    )
    op.execute("ALTER TABLE earnings_events ADD COLUMN IF NOT EXISTS first_seen_at TIMESTAMPTZ")
    op.execute(
        "UPDATE earnings_events SET first_seen_at = created_at "
        "WHERE first_seen_at IS NULL AND created_at IS NOT NULL"
    )
    op.execute(
        "UPDATE earnings_events SET reported_date = NULL "
        "WHERE announcement_timestamp IS NULL "
        "AND availability_timestamp IS NULL "
        "AND first_seen_at IS NOT NULL "
        "AND reported_date = fiscal_date_ending"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_earnings_events_symbol_availability "
        "ON earnings_events (symbol, availability_timestamp)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_earnings_events_symbol_first_seen "
        "ON earnings_events (symbol, first_seen_at)"
    )


def downgrade() -> None:
    """Drop immutable earnings availability timestamps."""
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        raise RuntimeError("004_earnings_availability migration requires PostgreSQL.")

    op.execute("DROP INDEX IF EXISTS ix_earnings_events_symbol_first_seen")
    op.execute("DROP INDEX IF EXISTS ix_earnings_events_symbol_availability")
    op.execute("ALTER TABLE earnings_events DROP COLUMN IF EXISTS first_seen_at")
    op.execute("ALTER TABLE earnings_events DROP COLUMN IF EXISTS availability_timestamp")
    op.execute("ALTER TABLE earnings_events DROP COLUMN IF EXISTS announcement_timestamp")
