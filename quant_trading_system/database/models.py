"""
SQLAlchemy ORM models for the trading system.

Defines all database tables including:
- Market data (OHLCV bars, features)
- Trading (orders, trades, positions)
- Signals and predictions
- Performance tracking
- System logs and alerts
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from quant_trading_system.data.timeframe import DEFAULT_TIMEFRAME


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    @declared_attr.directive
    def __tablename__(cls) -> str:
        """Generate table name from class name."""
        # Convert CamelCase to snake_case
        name = cls.__name__
        return "".join(["_" + c.lower() if c.isupper() else c for c in name]).lstrip("_")


def _utc_now() -> datetime:
    """Get current UTC time. Replaces deprecated _utc_now()."""
    return datetime.now(timezone.utc)


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utc_now,
        onupdate=_utc_now,
        nullable=False,
    )


# =============================================================================
# Market Data Models
# =============================================================================


class OHLCVBar(Base):
    """OHLCV bar data for market prices.

    This table is intended to be a TimescaleDB hypertable
    partitioned by timestamp (weekly).
    """

    __tablename__ = "ohlcv_bars"

    symbol: Mapped[str] = mapped_column(String(10), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        primary_key=True,
    )
    timeframe: Mapped[str] = mapped_column(
        String(16),
        primary_key=True,
        default=DEFAULT_TIMEFRAME,
    )
    open: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)
    vwap: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    trade_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    __table_args__ = (
        Index("ix_ohlcv_bars_timestamp", "timestamp", postgresql_using="btree"),
        Index("ix_ohlcv_bars_symbol", "symbol", postgresql_using="btree"),
        Index("ix_ohlcv_bars_symbol_timeframe", "symbol", "timeframe", postgresql_using="btree"),
    )

    def __repr__(self) -> str:
        return f"<OHLCVBar({self.symbol}, {self.timestamp}, C={self.close})>"

    def to_dict(self, preserve_decimal_precision: bool = True) -> dict[str, Any]:
        """Convert to dictionary.

        Args:
            preserve_decimal_precision: If True, converts Decimal to str to preserve
                financial precision. If False, converts to float (may lose precision).
                Default True for JPMorgan-level accuracy.

        Returns:
            Dictionary representation of the bar.
        """
        if preserve_decimal_precision:
            return {
                "symbol": self.symbol,
                "timestamp": self.timestamp.isoformat() if self.timestamp else None,
                "timeframe": self.timeframe,
                "open": str(self.open) if self.open else None,
                "high": str(self.high) if self.high else None,
                "low": str(self.low) if self.low else None,
                "close": str(self.close) if self.close else None,
                "volume": self.volume,
                "vwap": str(self.vwap) if self.vwap else None,
                "trade_count": self.trade_count,
            }
        else:
            return {
                "symbol": self.symbol,
                "timestamp": self.timestamp.isoformat() if self.timestamp else None,
                "timeframe": self.timeframe,
                "open": float(self.open) if self.open else None,
                "high": float(self.high) if self.high else None,
                "low": float(self.low) if self.low else None,
                "close": float(self.close) if self.close else None,
                "volume": self.volume,
                "vwap": float(self.vwap) if self.vwap else None,
                "trade_count": self.trade_count,
            }


class Feature(Base):
    """Computed features for machine learning.

    This table is intended to be a TimescaleDB hypertable
    partitioned by timestamp (daily).
    """

    __tablename__ = "features"

    symbol: Mapped[str] = mapped_column(String(10), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        primary_key=True,
    )
    timeframe: Mapped[str] = mapped_column(
        String(16),
        primary_key=True,
        default=DEFAULT_TIMEFRAME,
    )
    feature_name: Mapped[str] = mapped_column(String(100), primary_key=True)
    feature_set_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default="default",
    )
    value: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        Index("ix_features_symbol_timestamp", "symbol", "timeframe", "timestamp"),
        Index("ix_features_symbol_feature_set", "symbol", "timeframe", "feature_set_id"),
        Index("ix_features_feature_name", "feature_name"),
    )

    def __repr__(self) -> str:
        return f"<Feature({self.symbol}, {self.feature_name}={self.value})>"


# =============================================================================
# Market Intelligence / Reference Data Models
# =============================================================================


class SecurityMaster(Base, TimestampMixin):
    """Reference master for tradable instruments."""

    __tablename__ = "security_master"

    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    exchange: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    asset_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    ipo_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    delisting_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    currency: Mapped[str | None] = mapped_column(String(16), nullable=True)
    country: Mapped[str | None] = mapped_column(String(64), nullable=True)
    sector: Mapped[str | None] = mapped_column(String(128), nullable=True)
    industry: Mapped[str | None] = mapped_column(String(255), nullable=True)
    market_cap: Mapped[Decimal | None] = mapped_column(Numeric(20, 2), nullable=True)
    shares_outstanding: Mapped[Decimal | None] = mapped_column(Numeric(20, 4), nullable=True)
    cik: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="mixed", index=True)
    security_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        Index("ix_security_master_exchange_status", "exchange", "status"),
    )


class CorporateAction(Base, TimestampMixin):
    """Dividend and split events for point-in-time adjustment workflows."""

    __tablename__ = "corporate_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    action_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    ex_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    record_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    payment_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    declaration_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    amount: Mapped[Decimal | None] = mapped_column(Numeric(20, 6), nullable=True)
    split_from: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)
    split_to: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)
    currency: Mapped[str | None] = mapped_column(String(16), nullable=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="mixed", index=True)
    action_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "action_type",
            "ex_date",
            "source",
            name="uq_corporate_actions_symbol_type_date_source",
        ),
        Index("ix_corporate_actions_symbol_type_date", "symbol", "action_type", "ex_date"),
    )


class FundamentalSnapshot(Base, TimestampMixin):
    """Daily snapshots of slowly changing fundamental state."""

    __tablename__ = "fundamental_snapshots"

    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    as_of_date: Mapped[date] = mapped_column(Date, primary_key=True)
    source: Mapped[str] = mapped_column(String(32), primary_key=True, default="mixed")
    market_cap: Mapped[Decimal | None] = mapped_column(Numeric(20, 2), nullable=True)
    shares_outstanding: Mapped[Decimal | None] = mapped_column(Numeric(20, 4), nullable=True)
    pe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    peg_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_to_book: Mapped[float | None] = mapped_column(Float, nullable=True)
    eps: Mapped[float | None] = mapped_column(Float, nullable=True)
    book_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    dividend_per_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    dividend_yield: Mapped[float | None] = mapped_column(Float, nullable=True)
    revenue_ttm: Mapped[Decimal | None] = mapped_column(Numeric(20, 2), nullable=True)
    gross_profit_ttm: Mapped[Decimal | None] = mapped_column(Numeric(20, 2), nullable=True)
    operating_margin_ttm: Mapped[float | None] = mapped_column(Float, nullable=True)
    profit_margin: Mapped[float | None] = mapped_column(Float, nullable=True)
    beta: Mapped[float | None] = mapped_column(Float, nullable=True)
    week_52_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    week_52_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    analyst_target_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    fundamentals_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        Index("ix_fundamental_snapshots_symbol_source", "symbol", "source"),
    )


class EarningsEvent(Base, TimestampMixin):
    """Earnings history and event metadata."""

    __tablename__ = "earnings_events"

    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    fiscal_date_ending: Mapped[date] = mapped_column(Date, primary_key=True)
    source: Mapped[str] = mapped_column(String(32), primary_key=True, default="mixed")
    reported_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    announcement_timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    availability_timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    first_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    reported_eps: Mapped[float | None] = mapped_column(Float, nullable=True)
    estimated_eps: Mapped[float | None] = mapped_column(Float, nullable=True)
    surprise: Mapped[float | None] = mapped_column(Float, nullable=True)
    surprise_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    earnings_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        Index("ix_earnings_events_symbol_reported", "symbol", "reported_date"),
    )


class SECFiling(Base, TimestampMixin):
    """Recent SEC filing metadata for event-driven research."""

    __tablename__ = "sec_filings"

    accession_number: Mapped[str] = mapped_column(String(32), primary_key=True)
    symbol: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    cik: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    form: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    filed_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    accepted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    report_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    filing_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    filing_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        Index("ix_sec_filings_symbol_form_date", "symbol", "form", "filed_date"),
    )


class MacroObservation(Base, TimestampMixin):
    """Macro and volatility series used for regime detection."""

    __tablename__ = "macro_observations"

    series_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    observation_date: Mapped[date] = mapped_column(Date, primary_key=True)
    source: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    interval: Mapped[str | None] = mapped_column(String(32), nullable=True)
    unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    value: Mapped[Decimal | None] = mapped_column(Numeric(20, 6), nullable=True)
    macro_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        Index("ix_macro_observations_source_date", "source", "observation_date"),
    )


class NewsArticle(Base, TimestampMixin):
    """News metadata stored for NLP and event studies."""

    __tablename__ = "news_articles"

    article_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    headline: Mapped[str | None] = mapped_column(String(512), nullable=True)
    author: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at_source: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    updated_at_source: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    symbols: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    sentiment: Mapped[float | None] = mapped_column(Float, nullable=True)
    news_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        Index("ix_news_articles_source_created", "source", "created_at_source"),
    )


# =============================================================================
# Market Microstructure / Short Pressure Models
# =============================================================================


class StockQuote(Base, TimestampMixin):
    """Historical top-of-book quote updates for market microstructure research."""

    __tablename__ = "stock_quotes"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="alpaca_sip", index=True)
    bid_price: Mapped[Decimal | None] = mapped_column(Numeric(16, 6), nullable=True)
    bid_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bid_exchange: Mapped[str | None] = mapped_column(String(8), nullable=True)
    ask_price: Mapped[Decimal | None] = mapped_column(Numeric(16, 6), nullable=True)
    ask_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ask_exchange: Mapped[str | None] = mapped_column(String(8), nullable=True)
    tape: Mapped[str | None] = mapped_column(String(8), nullable=True)
    condition_codes: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    quote_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "timestamp",
            "bid_price",
            "bid_size",
            "bid_exchange",
            "ask_price",
            "ask_size",
            "ask_exchange",
            "source",
            name="uq_stock_quotes_event",
        ),
        Index("ix_stock_quotes_symbol_timestamp", "symbol", "timestamp"),
    )


class StockTrade(Base, TimestampMixin):
    """Historical trade prints for market microstructure research."""

    __tablename__ = "stock_trades"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="alpaca_sip", index=True)
    trade_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    price: Mapped[Decimal | None] = mapped_column(Numeric(16, 6), nullable=True)
    size: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    exchange: Mapped[str | None] = mapped_column(String(8), nullable=True)
    tape: Mapped[str | None] = mapped_column(String(8), nullable=True)
    condition_codes: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    trade_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "timestamp",
            "trade_id",
            "exchange",
            "source",
            name="uq_stock_trades_event",
        ),
        Index("ix_stock_trades_symbol_timestamp", "symbol", "timestamp"),
    )


class ShortSaleVolume(Base, TimestampMixin):
    """FINRA Reg SHO short sale volume aggregates."""

    __tablename__ = "short_sale_volumes"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    market: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="finra_regsho", index=True)
    short_volume: Mapped[Decimal | None] = mapped_column(Numeric(20, 6), nullable=True)
    short_exempt_volume: Mapped[Decimal | None] = mapped_column(Numeric(20, 6), nullable=True)
    total_volume: Mapped[Decimal | None] = mapped_column(Numeric(20, 6), nullable=True)
    volume_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "trade_date",
            "market",
            "source",
            name="uq_short_sale_volumes_symbol_date_market_source",
        ),
        Index("ix_short_sale_volumes_symbol_date", "symbol", "trade_date"),
    )


class FailsToDeliver(Base, TimestampMixin):
    """SEC fails-to-deliver dataset for settlement stress analysis."""

    __tablename__ = "fails_to_deliver"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    settlement_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    cusip: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    symbol: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="sec_ftd", index=True)
    quantity: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    price: Mapped[Decimal | None] = mapped_column(Numeric(16, 6), nullable=True)
    ftd_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "settlement_date",
            "cusip",
            "symbol",
            "source",
            name="uq_fails_to_deliver_settlement_cusip_symbol_source",
        ),
        Index("ix_fails_to_deliver_symbol_settlement", "symbol", "settlement_date"),
    )


class MacroVintageObservation(Base, TimestampMixin):
    """Point-in-time macro vintages from ALFRED for revision-aware training."""

    __tablename__ = "macro_vintage_observations"

    series_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    observation_date: Mapped[date] = mapped_column(Date, primary_key=True)
    realtime_start: Mapped[date] = mapped_column(Date, primary_key=True)
    realtime_end: Mapped[date] = mapped_column(Date, primary_key=True)
    source: Mapped[str] = mapped_column(String(32), primary_key=True, default="alfred")
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    interval: Mapped[str | None] = mapped_column(String(32), nullable=True)
    unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    value: Mapped[Decimal | None] = mapped_column(Numeric(20, 6), nullable=True)
    vintage_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        Index("ix_macro_vintage_observations_series_date", "series_id", "observation_date"),
        Index("ix_macro_vintage_observations_realtime", "source", "realtime_start"),
    )


# =============================================================================
# Trading Models
# =============================================================================


class Order(Base, TimestampMixin):
    """Order records for trade execution."""

    __tablename__ = "orders"

    order_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    client_order_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    broker_order_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(4), nullable=False)  # BUY/SELL
    order_type: Mapped[str] = mapped_column(String(20), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    limit_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    stop_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    time_in_force: Mapped[str] = mapped_column(String(10), nullable=False, default="DAY")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="PENDING", index=True)
    filled_qty: Mapped[Decimal] = mapped_column(Numeric(12, 4), default=Decimal("0"))
    filled_avg_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    filled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    strategy_name: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    signal_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), nullable=True)
    order_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Relationships
    trades: Mapped[list["Trade"]] = relationship("Trade", back_populates="order")

    __table_args__ = (
        Index("ix_orders_symbol_created", "symbol", "created_at"),
        Index("ix_orders_strategy_created", "strategy_name", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Order({self.order_id[:8]}, {self.symbol}, {self.side}, {self.status})>"

    def to_dict(self, preserve_decimal_precision: bool = True) -> dict[str, Any]:
        """Convert to dictionary.

        Args:
            preserve_decimal_precision: If True, converts Decimal to str to preserve
                financial precision. If False, converts to float (may lose precision).
                Default True for JPMorgan-level accuracy.

        Returns:
            Dictionary representation of the order.
        """
        if preserve_decimal_precision:
            return {
                "order_id": self.order_id,
                "client_order_id": self.client_order_id,
                "broker_order_id": self.broker_order_id,
                "symbol": self.symbol,
                "side": self.side,
                "order_type": self.order_type,
                "quantity": str(self.quantity) if self.quantity else None,
                "limit_price": str(self.limit_price) if self.limit_price else None,
                "stop_price": str(self.stop_price) if self.stop_price else None,
                "time_in_force": self.time_in_force,
                "status": self.status,
                "filled_qty": str(self.filled_qty) if self.filled_qty else "0",
                "filled_avg_price": str(self.filled_avg_price) if self.filled_avg_price else None,
            }
        else:
            return {
                "order_id": self.order_id,
                "client_order_id": self.client_order_id,
                "broker_order_id": self.broker_order_id,
                "symbol": self.symbol,
                "side": self.side,
                "order_type": self.order_type,
                "quantity": float(self.quantity) if self.quantity else None,
                "limit_price": float(self.limit_price) if self.limit_price else None,
                "stop_price": float(self.stop_price) if self.stop_price else None,
                "time_in_force": self.time_in_force,
                "status": self.status,
                "filled_qty": float(self.filled_qty) if self.filled_qty else 0,
                "filled_avg_price": float(self.filled_avg_price) if self.filled_avg_price else None,
            }


class Trade(Base):
    """Individual trade executions (fills)."""

    __tablename__ = "trades"

    trade_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    order_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("orders.order_id"),
        nullable=False,
    )
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(4), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    commission: Mapped[Decimal] = mapped_column(Numeric(10, 4), default=Decimal("0"))
    executed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    # Relationships
    order: Mapped["Order"] = relationship("Order", back_populates="trades")

    __table_args__ = (
        Index("ix_trades_symbol_executed", "symbol", "executed_at"),
    )

    def __repr__(self) -> str:
        return f"<Trade({self.trade_id[:8]}, {self.symbol}, {self.quantity}@{self.price})>"


class Position(Base):
    """Current positions in the portfolio."""

    __tablename__ = "positions"

    symbol: Mapped[str] = mapped_column(String(10), primary_key=True)
    quantity: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    avg_entry_price: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    cost_basis: Mapped[Decimal] = mapped_column(Numeric(14, 4), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utc_now,
        onupdate=_utc_now,
    )

    def __repr__(self) -> str:
        return f"<Position({self.symbol}, qty={self.quantity})>"


class PositionHistory(Base):
    """Historical position snapshots.

    This table is intended to be a TimescaleDB hypertable
    partitioned by timestamp (daily).
    """

    __tablename__ = "position_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    quantity: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    avg_entry_price: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    def __repr__(self) -> str:
        return f"<PositionHistory({self.symbol}, {self.timestamp})>"


# =============================================================================
# Signals and Predictions
# =============================================================================


class Signal(Base):
    """Trading signals generated by models.

    This table is intended to be a TimescaleDB hypertable
    partitioned by timestamp (daily).
    """

    __tablename__ = "signals"

    signal_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)  # LONG/SHORT/FLAT
    strength: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    horizon: Mapped[int] = mapped_column(Integer, nullable=False)
    model_source: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    features_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        Index("ix_signals_symbol_timestamp", "symbol", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<Signal({self.symbol}, {self.direction}, strength={self.strength:.2f})>"


class ModelPrediction(Base):
    """Model predictions for tracking and analysis.

    This table is intended to be a TimescaleDB hypertable
    partitioned by timestamp (daily).
    """

    __tablename__ = "model_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(20), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    prediction: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    actual: Mapped[float | None] = mapped_column(Float, nullable=True)  # Filled later
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utc_now,
    )

    def __repr__(self) -> str:
        return f"<ModelPrediction({self.model_name}, {self.symbol}, pred={self.prediction:.4f})>"


# =============================================================================
# Performance Tracking
# =============================================================================


class DailyPerformance(Base):
    """Daily performance metrics."""

    __tablename__ = "daily_performance"

    date: Mapped[datetime] = mapped_column(Date, primary_key=True)
    starting_equity: Mapped[Decimal] = mapped_column(Numeric(14, 4), nullable=False)
    ending_equity: Mapped[Decimal] = mapped_column(Numeric(14, 4), nullable=False)
    pnl: Mapped[Decimal] = mapped_column(Numeric(14, 4), nullable=False)
    pnl_percent: Mapped[float] = mapped_column(Float, nullable=False)
    trades_count: Mapped[int] = mapped_column(Integer, default=0)
    win_count: Mapped[int] = mapped_column(Integer, default=0)
    loss_count: Mapped[int] = mapped_column(Integer, default=0)
    max_drawdown: Mapped[float] = mapped_column(Float, default=0.0)
    sharpe_estimate: Mapped[float | None] = mapped_column(Float, nullable=True)
    perf_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    def __repr__(self) -> str:
        return f"<DailyPerformance({self.date}, pnl={self.pnl_percent:.2f}%)>"


class TradeLog(Base):
    """Complete trade log for round-trip trades."""

    __tablename__ = "trade_log"

    trade_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    exit_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    side: Mapped[str] = mapped_column(String(4), nullable=False)
    entry_quantity: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    exit_quantity: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    entry_price: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    exit_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    pnl: Mapped[Decimal | None] = mapped_column(Numeric(14, 4), nullable=True)
    pnl_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    commission_total: Mapped[Decimal] = mapped_column(Numeric(10, 4), default=Decimal("0"))
    slippage: Mapped[float | None] = mapped_column(Float, nullable=True)
    signals: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    strategy: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<TradeLog({self.symbol}, {self.side}, pnl={self.pnl})>"


# =============================================================================
# System Models
# =============================================================================


class SystemLog(Base):
    """System log entries.

    This table is intended to be a TimescaleDB hypertable
    partitioned by timestamp (daily) with 30-day retention.
    """

    __tablename__ = "system_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utc_now,
        nullable=False,
        index=True,
    )
    level: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    correlation_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), nullable=True)
    context: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    def __repr__(self) -> str:
        return f"<SystemLog({self.level}, {self.category}, {self.message[:50]})>"


class Alert(Base):
    """System alerts for monitoring."""

    __tablename__ = "alerts"

    alert_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utc_now,
        nullable=False,
        index=True,
    )
    severity: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    acknowledged_by: Mapped[str | None] = mapped_column(String(50), nullable=True)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)

    def __repr__(self) -> str:
        return f"<Alert({self.severity}, {self.title[:50]})>"


class RiskEvent(Base):
    """Risk events for audit trail and monitoring.

    Records all risk-related events including:
    - Kill switch activations/deactivations
    - Position limit breaches
    - Drawdown warnings
    - Risk parameter changes
    - Order rejections due to risk

    This table is intended to be a TimescaleDB hypertable
    partitioned by timestamp (daily) with 90-day retention.
    """

    __tablename__ = "risk_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utc_now,
        nullable=False,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )  # KILL_SWITCH_ACTIVATED, POSITION_LIMIT_BREACH, DRAWDOWN_WARNING, etc.
    severity: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
    )  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    symbol: Mapped[str | None] = mapped_column(String(10), nullable=True, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)

    # Metrics at time of event
    metrics: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )  # Current risk metrics snapshot

    # Related identifiers
    order_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), nullable=True)
    strategy_name: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Resolution tracking
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(50), nullable=True)
    resolution_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_risk_events_type_timestamp", "event_type", "timestamp"),
        Index("ix_risk_events_severity_timestamp", "severity", "timestamp"),
        Index("ix_risk_events_symbol_timestamp", "symbol", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<RiskEvent({self.event_type}, {self.severity}, {self.timestamp})>"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "event_type": self.event_type,
            "severity": self.severity,
            "symbol": self.symbol,
            "description": self.description,
            "metrics": self.metrics,
            "order_id": self.order_id,
            "strategy_name": self.strategy_name,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "resolution_notes": self.resolution_notes,
        }


# Export all models
__all__ = [
    "Base",
    "OHLCVBar",
    "Feature",
    "Order",
    "Trade",
    "Position",
    "PositionHistory",
    "Signal",
    "ModelPrediction",
    "DailyPerformance",
    "TradeLog",
    "SystemLog",
    "Alert",
    "RiskEvent",
]
