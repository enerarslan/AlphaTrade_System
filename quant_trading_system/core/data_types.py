"""
Pydantic models and type definitions for the trading system.

Defines strict type contracts for all data flowing through the system,
including OHLCV bars, trade signals, orders, positions, and portfolio state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _serialize_decimal(value: Decimal) -> str:
    """Serialize Decimal to string for JSON."""
    return str(value)


def _serialize_uuid(value: UUID) -> str:
    """Serialize UUID to string for JSON."""
    return str(value)


def _serialize_datetime(value: datetime) -> str:
    """Serialize datetime to ISO format string for JSON."""
    return value.isoformat()


class Direction(str, Enum):
    """Trade direction enum."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class OrderSide(str, Enum):
    """Order side enum."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type enum."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class OrderStatus(str, Enum):
    """Order status enum."""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    FILLED = "FILLED"
    PARTIAL_FILLED = "PARTIAL_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class TimeInForce(str, Enum):
    """Time in force enum."""

    DAY = "DAY"
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    OPG = "OPG"  # Market on Open
    CLS = "CLS"  # Market on Close


class OHLCVBar(BaseModel):
    """OHLCV bar data model with strict validation."""

    symbol: str = Field(..., min_length=1, max_length=10, description="Ticker symbol")
    timestamp: datetime = Field(..., description="Bar timestamp in UTC")
    open: Decimal = Field(..., ge=0, decimal_places=8, description="Open price")
    high: Decimal = Field(..., ge=0, decimal_places=8, description="High price")
    low: Decimal = Field(..., ge=0, decimal_places=8, description="Low price")
    close: Decimal = Field(..., ge=0, decimal_places=8, description="Close price")
    volume: int = Field(..., ge=0, description="Trading volume")
    vwap: Decimal | None = Field(default=None, ge=0, description="Volume weighted average price")
    trade_count: int | None = Field(default=None, ge=0, description="Number of trades")

    @model_validator(mode="after")
    def validate_ohlc_relationship(self) -> "OHLCVBar":
        """Validate OHLC relationship: low <= open,close <= high."""
        if self.low > self.open or self.low > self.close:
            raise ValueError(f"Low price ({self.low}) must be <= open ({self.open}) and close ({self.close})")
        if self.high < self.open or self.high < self.close:
            raise ValueError(f"High price ({self.high}) must be >= open ({self.open}) and close ({self.close})")
        if self.low > self.high:
            raise ValueError(f"Low price ({self.low}) must be <= high price ({self.high})")
        return self

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        return v.upper().strip()

    def to_dict(self, preserve_decimal_precision: bool = True) -> dict[str, Any]:
        """Convert to dictionary with serializable types.

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
                "timestamp": self.timestamp.isoformat(),
                "open": str(self.open),
                "high": str(self.high),
                "low": str(self.low),
                "close": str(self.close),
                "volume": self.volume,
                "vwap": str(self.vwap) if self.vwap else None,
                "trade_count": self.trade_count,
            }
        else:
            # Legacy float conversion (may lose precision)
            return {
                "symbol": self.symbol,
                "timestamp": self.timestamp.isoformat(),
                "open": float(self.open),
                "high": float(self.high),
                "low": float(self.low),
                "close": float(self.close),
                "volume": self.volume,
                "vwap": float(self.vwap) if self.vwap else None,
                "trade_count": self.trade_count,
            }

    model_config = ConfigDict(frozen=True)


class TradeSignal(BaseModel):
    """Trade signal generated by models."""

    signal_id: UUID = Field(default_factory=uuid4, description="Unique signal identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Signal generation time")
    symbol: str = Field(..., min_length=1, max_length=10, description="Ticker symbol")
    direction: Direction = Field(..., description="Signal direction")
    strength: float = Field(..., ge=-1.0, le=1.0, description="Signal strength (-1.0 to 1.0)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence (0.0 to 1.0)")
    horizon: int = Field(..., ge=1, description="Forecast horizon in bars")
    model_source: str = Field(..., description="Model that generated the signal")
    features_snapshot: dict[str, Any] = Field(default_factory=dict, description="Features at signal time")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        return v.upper().strip()

    def is_actionable(self, min_confidence: float = 0.5, min_strength: float = 0.3) -> bool:
        """Check if signal meets minimum thresholds for action."""
        return (
            self.confidence >= min_confidence
            and abs(self.strength) >= min_strength
            and self.direction != Direction.FLAT
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with JSON-serializable types."""
        return {
            "signal_id": str(self.signal_id),
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "direction": self.direction.value,
            "strength": self.strength,
            "confidence": self.confidence,
            "horizon": self.horizon,
            "model_source": self.model_source,
            "features_snapshot": self.features_snapshot,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TradeSignal":
        """Create TradeSignal from dictionary."""
        return cls(
            signal_id=UUID(data["signal_id"]) if isinstance(data.get("signal_id"), str) else data.get("signal_id", uuid4()),
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else data.get("timestamp", datetime.now(timezone.utc)),
            symbol=data["symbol"],
            direction=Direction(data["direction"]),
            strength=data["strength"],
            confidence=data["confidence"],
            horizon=data["horizon"],
            model_source=data["model_source"],
            features_snapshot=data.get("features_snapshot", {}),
            metadata=data.get("metadata", {}),
        )

    model_config = ConfigDict(frozen=True)


class Order(BaseModel):
    """Order model for trade execution."""

    order_id: UUID = Field(default_factory=uuid4, description="Internal order identifier")
    client_order_id: str = Field(default="", description="Client-specified order ID")
    broker_order_id: str | None = Field(default=None, description="Broker-assigned order ID")
    symbol: str = Field(..., min_length=1, max_length=10, description="Ticker symbol")
    side: OrderSide = Field(..., description="Order side (BUY/SELL)")
    order_type: OrderType = Field(default=OrderType.MARKET, description="Order type")
    quantity: Decimal = Field(..., gt=0, description="Order quantity")
    limit_price: Decimal | None = Field(default=None, ge=0, description="Limit price")
    stop_price: Decimal | None = Field(default=None, ge=0, description="Stop price")
    time_in_force: TimeInForce = Field(default=TimeInForce.DAY, description="Time in force")
    status: OrderStatus = Field(default=OrderStatus.PENDING, description="Order status")
    filled_qty: Decimal = Field(default=Decimal("0"), ge=0, description="Filled quantity")
    filled_avg_price: Decimal | None = Field(default=None, ge=0, description="Average fill price")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Order creation time")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Last update time")
    signal_id: UUID | None = Field(default=None, description="Associated signal ID")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        return v.upper().strip()

    @model_validator(mode="after")
    def validate_order_prices(self) -> "Order":
        """Validate order prices based on order type."""
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit orders require a limit price")
        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("Stop orders require a stop price")
        if self.order_type == OrderType.STOP_LIMIT:
            if self.limit_price is None or self.stop_price is None:
                raise ValueError("Stop-limit orders require both limit and stop prices")
        return self

    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.ACCEPTED, OrderStatus.PARTIAL_FILLED)

    @property
    def is_terminal(self) -> bool:
        """Check if order is in a terminal state."""
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED)

    @property
    def remaining_qty(self) -> Decimal:
        """Get remaining quantity to fill."""
        return self.quantity - self.filled_qty

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with JSON-serializable types."""
        return {
            "order_id": str(self.order_id),
            "client_order_id": self.client_order_id,
            "broker_order_id": self.broker_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": str(self.quantity),
            "limit_price": str(self.limit_price) if self.limit_price else None,
            "stop_price": str(self.stop_price) if self.stop_price else None,
            "time_in_force": self.time_in_force.value,
            "status": self.status.value,
            "filled_qty": str(self.filled_qty),
            "filled_avg_price": str(self.filled_avg_price) if self.filled_avg_price else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "signal_id": str(self.signal_id) if self.signal_id else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Order":
        """Create Order from dictionary."""
        return cls(
            order_id=UUID(data["order_id"]) if isinstance(data.get("order_id"), str) else data.get("order_id"),
            client_order_id=data.get("client_order_id", ""),
            broker_order_id=data.get("broker_order_id"),
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            order_type=OrderType(data.get("order_type", "MARKET")),
            quantity=Decimal(data["quantity"]),
            limit_price=Decimal(data["limit_price"]) if data.get("limit_price") else None,
            stop_price=Decimal(data["stop_price"]) if data.get("stop_price") else None,
            time_in_force=TimeInForce(data.get("time_in_force", "DAY")),
            status=OrderStatus(data.get("status", "PENDING")),
            filled_qty=Decimal(data.get("filled_qty", "0")),
            filled_avg_price=Decimal(data["filled_avg_price"]) if data.get("filled_avg_price") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.now(timezone.utc)),
            updated_at=datetime.fromisoformat(data["updated_at"]) if isinstance(data.get("updated_at"), str) else data.get("updated_at", datetime.now(timezone.utc)),
            signal_id=UUID(data["signal_id"]) if isinstance(data.get("signal_id"), str) else data.get("signal_id"),
        )


class Position(BaseModel):
    """Position model for portfolio tracking."""

    symbol: str = Field(..., min_length=1, max_length=10, description="Ticker symbol")
    quantity: Decimal = Field(..., description="Position quantity (positive=long, negative=short)")
    avg_entry_price: Decimal = Field(..., ge=0, description="Average entry price")
    current_price: Decimal = Field(..., ge=0, description="Current market price")
    cost_basis: Decimal = Field(..., description="Total cost basis")
    market_value: Decimal = Field(..., description="Current market value")
    unrealized_pnl: Decimal = Field(default=Decimal("0"), description="Unrealized P&L")
    realized_pnl: Decimal = Field(default=Decimal("0"), description="Realized P&L")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Last update time")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        return v.upper().strip()

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        """Check if position is flat (no position)."""
        return self.quantity == 0

    @property
    def abs_quantity(self) -> Decimal:
        """Get absolute quantity."""
        return abs(self.quantity)

    @property
    def unrealized_pnl_pct(self) -> float:
        """Get unrealized P&L as percentage."""
        if self.cost_basis == 0:
            return 0.0
        return float(self.unrealized_pnl / abs(self.cost_basis)) * 100

    def update_price(self, new_price: Decimal) -> "Position":
        """Update position with new price and recalculate values."""
        market_value = self.quantity * new_price
        unrealized_pnl = market_value - self.cost_basis
        return Position(
            symbol=self.symbol,
            quantity=self.quantity,
            avg_entry_price=self.avg_entry_price,
            current_price=new_price,
            cost_basis=self.cost_basis,
            market_value=market_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self.realized_pnl,
            last_updated=datetime.now(timezone.utc),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with JSON-serializable types."""
        return {
            "symbol": self.symbol,
            "quantity": str(self.quantity),
            "avg_entry_price": str(self.avg_entry_price),
            "current_price": str(self.current_price),
            "cost_basis": str(self.cost_basis),
            "market_value": str(self.market_value),
            "unrealized_pnl": str(self.unrealized_pnl),
            "realized_pnl": str(self.realized_pnl),
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Position":
        """Create Position from dictionary."""
        return cls(
            symbol=data["symbol"],
            quantity=Decimal(data["quantity"]),
            avg_entry_price=Decimal(data["avg_entry_price"]),
            current_price=Decimal(data["current_price"]),
            cost_basis=Decimal(data["cost_basis"]),
            market_value=Decimal(data["market_value"]),
            unrealized_pnl=Decimal(data.get("unrealized_pnl", "0")),
            realized_pnl=Decimal(data.get("realized_pnl", "0")),
            last_updated=datetime.fromisoformat(data["last_updated"]) if isinstance(data.get("last_updated"), str) else data.get("last_updated", datetime.now(timezone.utc)),
        )


class Portfolio(BaseModel):
    """Portfolio state model."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Portfolio snapshot time")
    equity: Decimal = Field(..., ge=0, description="Total equity value")
    cash: Decimal = Field(..., description="Cash balance")
    buying_power: Decimal = Field(..., ge=0, description="Available buying power")
    positions: dict[str, Position] = Field(default_factory=dict, description="Current positions by symbol")
    pending_orders: list[Order] = Field(default_factory=list, description="Pending orders")
    daily_pnl: Decimal = Field(default=Decimal("0"), description="Daily P&L")
    total_pnl: Decimal = Field(default=Decimal("0"), description="Total P&L")
    margin_used: Decimal = Field(default=Decimal("0"), ge=0, description="Margin in use")

    @property
    def position_count(self) -> int:
        """Get number of open positions."""
        return len([p for p in self.positions.values() if not p.is_flat])

    @property
    def total_market_value(self) -> Decimal:
        """Get total market value of all positions."""
        return sum(p.market_value for p in self.positions.values())

    @property
    def total_unrealized_pnl(self) -> Decimal:
        """Get total unrealized P&L."""
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def total_realized_pnl(self) -> Decimal:
        """Get total realized P&L."""
        return sum(p.realized_pnl for p in self.positions.values())

    @property
    def long_exposure(self) -> Decimal:
        """Get total long exposure."""
        return sum(p.market_value for p in self.positions.values() if p.is_long)

    @property
    def short_exposure(self) -> Decimal:
        """Get total short exposure (absolute value)."""
        return abs(sum(p.market_value for p in self.positions.values() if p.is_short))

    @property
    def net_exposure(self) -> Decimal:
        """Get net exposure (long - short)."""
        return self.long_exposure - self.short_exposure

    @property
    def gross_exposure(self) -> Decimal:
        """Get gross exposure (long + short)."""
        return self.long_exposure + self.short_exposure

    def get_position(self, symbol: str) -> Position | None:
        """Get position for a symbol."""
        return self.positions.get(symbol.upper())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with JSON-serializable types."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "equity": str(self.equity),
            "cash": str(self.cash),
            "buying_power": str(self.buying_power),
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "pending_orders": [o.to_dict() for o in self.pending_orders],
            "daily_pnl": str(self.daily_pnl),
            "total_pnl": str(self.total_pnl),
            "margin_used": str(self.margin_used),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Portfolio":
        """Create Portfolio from dictionary."""
        positions = {k: Position.from_dict(v) for k, v in data.get("positions", {}).items()}
        pending_orders = [Order.from_dict(o) for o in data.get("pending_orders", [])]
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else data.get("timestamp", datetime.now(timezone.utc)),
            equity=Decimal(data["equity"]),
            cash=Decimal(data["cash"]),
            buying_power=Decimal(data["buying_power"]),
            positions=positions,
            pending_orders=pending_orders,
            daily_pnl=Decimal(data.get("daily_pnl", "0")),
            total_pnl=Decimal(data.get("total_pnl", "0")),
            margin_used=Decimal(data.get("margin_used", "0")),
        )


class RiskMetrics(BaseModel):
    """Risk metrics model for portfolio monitoring."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Metrics calculation time")
    portfolio_var_95: Decimal = Field(default=Decimal("0"), description="95% VaR")
    portfolio_var_99: Decimal = Field(default=Decimal("0"), description="99% VaR")
    portfolio_cvar: Decimal = Field(default=Decimal("0"), description="Conditional VaR (Expected Shortfall)")
    sharpe_ratio: float = Field(default=0.0, description="Sharpe ratio")
    sortino_ratio: float = Field(default=0.0, description="Sortino ratio")
    max_drawdown: float = Field(default=0.0, ge=0.0, le=1.0, description="Maximum drawdown")
    current_drawdown: float = Field(default=0.0, ge=0.0, le=1.0, description="Current drawdown")
    beta: float = Field(default=1.0, description="Portfolio beta")
    correlation_matrix: list[list[float]] | None = Field(default=None, description="Correlation matrix")
    sector_exposures: dict[str, Decimal] = Field(default_factory=dict, description="Sector exposure map")
    volatility_annual: float = Field(default=0.0, ge=0.0, description="Annualized volatility")
    calmar_ratio: float = Field(default=0.0, description="Calmar ratio")
    information_ratio: float = Field(default=0.0, description="Information ratio")

    def is_within_limits(
        self,
        max_var_95: Decimal | None = None,
        max_drawdown: float | None = None,
        max_volatility: float | None = None,
    ) -> bool:
        """Check if metrics are within specified limits."""
        if max_var_95 is not None and self.portfolio_var_95 > max_var_95:
            return False
        if max_drawdown is not None and self.current_drawdown > max_drawdown:
            return False
        if max_volatility is not None and self.volatility_annual > max_volatility:
            return False
        return True


class FeatureVector(BaseModel):
    """Feature vector for model input."""

    symbol: str = Field(..., description="Ticker symbol")
    timestamp: datetime = Field(..., description="Feature calculation time")
    features: dict[str, float] = Field(..., description="Feature name to value mapping")
    feature_names: list[str] = Field(default_factory=list, description="Ordered feature names")

    @property
    def values(self) -> list[float]:
        """Get feature values in order of feature_names."""
        if self.feature_names:
            return [self.features.get(name, 0.0) for name in self.feature_names]
        return list(self.features.values())

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self.values, dtype=np.float32)


class ModelPrediction(BaseModel):
    """Model prediction output."""

    model_name: str = Field(..., description="Model identifier")
    symbol: str = Field(..., description="Ticker symbol")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Prediction time")
    prediction: float = Field(..., description="Raw prediction value")
    probability: float | None = Field(default=None, ge=0.0, le=1.0, description="Prediction probability")
    direction: Direction = Field(..., description="Predicted direction")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence")
    horizon: int = Field(..., ge=1, description="Prediction horizon in bars")
    feature_importance: dict[str, float] = Field(default_factory=dict, description="Feature importance scores")
