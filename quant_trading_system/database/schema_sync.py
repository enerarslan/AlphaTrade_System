"""Lightweight schema sync helpers for reference-data extensions."""

from __future__ import annotations

from sqlalchemy import text

from quant_trading_system.database.connection import DatabaseManager, get_db_manager

_REFERENCE_SCHEMA_SYNCED = False


def ensure_reference_schema_extensions(
    db_manager: DatabaseManager | None = None,
    *,
    force: bool = False,
) -> None:
    """Ensure additive reference-data columns/indexes exist on PostgreSQL."""
    global _REFERENCE_SCHEMA_SYNCED
    if _REFERENCE_SCHEMA_SYNCED and not force:
        return

    manager = db_manager or get_db_manager()
    statements = (
        """
        ALTER TABLE earnings_events
        ADD COLUMN IF NOT EXISTS announcement_timestamp TIMESTAMPTZ
        """,
        """
        ALTER TABLE earnings_events
        ADD COLUMN IF NOT EXISTS availability_timestamp TIMESTAMPTZ
        """,
        """
        ALTER TABLE earnings_events
        ADD COLUMN IF NOT EXISTS first_seen_at TIMESTAMPTZ
        """,
        """
        UPDATE earnings_events
        SET first_seen_at = created_at
        WHERE first_seen_at IS NULL AND created_at IS NOT NULL
        """,
        """
        UPDATE earnings_events
        SET reported_date = NULL
        WHERE announcement_timestamp IS NULL
          AND availability_timestamp IS NULL
          AND first_seen_at IS NOT NULL
          AND reported_date = fiscal_date_ending
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_earnings_events_symbol_availability
        ON earnings_events (symbol, availability_timestamp)
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_earnings_events_symbol_first_seen
        ON earnings_events (symbol, first_seen_at)
        """,
    )

    with manager.engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))

    _REFERENCE_SCHEMA_SYNCED = True
