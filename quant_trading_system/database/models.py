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

from datetime import datetime
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


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    @declared_attr.directive
    def __tablename__(cls) -> str:
        """Generate table name from class name."""
        # Convert CamelCase to snake_case
        name = cls.__name__
        return "".join(["_" + c.lower() if c.isupper() else c for c in name]).lstrip("_")


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
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
    )

    def __repr__(self) -> str:
        return f"<OHLCVBar({self.symbol}, {self.timestamp}, C={self.close})>"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
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
    feature_name: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        Index("ix_features_symbol_timestamp", "symbol", "timestamp"),
        Index("ix_features_feature_name", "feature_name"),
    )

    def __repr__(self) -> str:
        return f"<Feature({self.symbol}, {self.feature_name}={self.value})>"


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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
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
        default=datetime.utcnow,
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
        default=datetime.utcnow,
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
        default=datetime.utcnow,
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
]
