"""Create reference and research data tables missing from the migration chain.

Revision ID: 005_reference_data
Revises: 004_earnings_availability
Create Date: 2026-03-17
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

from quant_trading_system.database.models import (
    CorporateAction,
    EarningsEvent,
    FailsToDeliver,
    FundamentalSnapshot,
    MacroObservation,
    MacroVintageObservation,
    NewsArticle,
    SECFiling,
    SecurityMaster,
    ShortSaleVolume,
    StockQuote,
    StockTrade,
)

# revision identifiers, used by Alembic.
revision: str = "005_reference_data"
down_revision: Union[str, None] = "004_earnings_availability"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _reference_tables():
    return (
        SecurityMaster.__table__,
        CorporateAction.__table__,
        FundamentalSnapshot.__table__,
        EarningsEvent.__table__,
        SECFiling.__table__,
        MacroObservation.__table__,
        NewsArticle.__table__,
        StockQuote.__table__,
        StockTrade.__table__,
        ShortSaleVolume.__table__,
        FailsToDeliver.__table__,
        MacroVintageObservation.__table__,
    )


def upgrade() -> None:
    """Create the reference-data tables expected by training and bootstrap flows."""
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        raise RuntimeError("005_reference_data migration requires PostgreSQL.")

    for table in _reference_tables():
        table.create(bind=bind, checkfirst=True)


def downgrade() -> None:
    """Drop the reference-data tables introduced by this migration."""
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        raise RuntimeError("005_reference_data migration requires PostgreSQL.")

    for table in reversed(_reference_tables()):
        table.drop(bind=bind, checkfirst=True)
