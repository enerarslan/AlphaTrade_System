"""
Structured logging system for the trading system.

Provides:
- JSON and human-readable log formats
- Contextual metadata and correlation IDs
- Log categories for different components
- Trade logging for audit trails
- Rotating file handlers with retention
- TRACE level logging for ultra-detailed debugging
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# =============================================================================
# TRACE Level Logging (below DEBUG)
# =============================================================================

# Define TRACE level - lower than DEBUG (10), we use 5
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


def trace(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Log a message at TRACE level.

    TRACE level is for ultra-detailed debugging, such as:
    - Every tick/bar received
    - Individual feature calculations
    - Model layer activations
    - Order book updates
    - Memory allocation details

    Args:
        message: Log message.
        *args: Positional arguments for message formatting.
        **kwargs: Keyword arguments for logging.
    """
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kwargs)


# Add trace method to Logger class
logging.Logger.trace = trace  # type: ignore[attr-defined]


class LogCategory(str, Enum):
    """Log categories for different components."""

    SYSTEM = "SYSTEM"
    DATA = "DATA"
    MODEL = "MODEL"
    TRADING = "TRADING"
    ORDER = "ORDER"
    RISK = "RISK"
    ALERT = "ALERT"


class LogFormat(str, Enum):
    """Log output formats."""

    JSON = "json"
    TEXT = "text"


class TradeLogEntry(BaseModel):
    """Trade log entry for audit trail."""

    entry_time: datetime
    exit_time: datetime | None = None
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float | None = None
    pnl: float | None = None
    pnl_percent: float | None = None
    signals: list[str] = Field(default_factory=list)
    model_predictions: dict[str, float] = Field(default_factory=dict)
    risk_metrics: dict[str, float] = Field(default_factory=dict)
    execution_details: dict[str, Any] = Field(default_factory=dict)


class StructuredLogRecord(BaseModel):
    """Structured log record with metadata."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    level: str
    category: LogCategory
    message: str
    correlation_id: str | None = None
    symbol: str | None = None
    order_id: str | None = None
    model_name: str | None = None
    extra_data: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON string."""
        data = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "category": self.category.value,
            "message": self.message,
        }
        if self.correlation_id:
            data["correlation_id"] = self.correlation_id
        if self.symbol:
            data["symbol"] = self.symbol
        if self.order_id:
            data["order_id"] = self.order_id
        if self.model_name:
            data["model_name"] = self.model_name
        if self.extra_data:
            data["extra_data"] = self.extra_data
        return json.dumps(data)

    def to_text(self) -> str:
        """Convert to human-readable text."""
        parts = [
            self.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            f"[{self.level:8s}]",
            f"[{self.category.value:8s}]",
        ]
        if self.correlation_id:
            parts.append(f"[{self.correlation_id[:8]}]")
        if self.symbol:
            parts.append(f"[{self.symbol}]")
        parts.append(self.message)
        if self.extra_data:
            parts.append(f"| {self.extra_data}")
        return " ".join(parts)


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def __init__(self, category: LogCategory = LogCategory.SYSTEM) -> None:
        """Initialize the formatter.

        Args:
            category: Default log category.
        """
        super().__init__()
        self.category = category

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON.

        Args:
            record: Log record to format.

        Returns:
            JSON formatted string.
        """
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "category": getattr(record, "category", self.category.value),
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add optional fields
        if hasattr(record, "correlation_id") and record.correlation_id:
            log_data["correlation_id"] = record.correlation_id
        if hasattr(record, "symbol") and record.symbol:
            log_data["symbol"] = record.symbol
        if hasattr(record, "order_id") and record.order_id:
            log_data["order_id"] = record.order_id
        if hasattr(record, "model_name") and record.model_name:
            log_data["model_name"] = record.model_name
        if hasattr(record, "extra_data") and record.extra_data:
            log_data["extra_data"] = record.extra_data

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter for development."""

    def __init__(self, category: LogCategory = LogCategory.SYSTEM) -> None:
        """Initialize the formatter.

        Args:
            category: Default log category.
        """
        super().__init__()
        self.category = category

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as human-readable text.

        Args:
            record: Log record to format.

        Returns:
            Formatted string.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        category = getattr(record, "category", self.category.value)
        level = record.levelname

        parts = [
            timestamp,
            f"[{level:8s}]",
            f"[{category:8s}]",
        ]

        if hasattr(record, "correlation_id") and record.correlation_id:
            parts.append(f"[{record.correlation_id[:8]}]")
        if hasattr(record, "symbol") and record.symbol:
            parts.append(f"[{record.symbol}]")

        parts.append(record.getMessage())

        if hasattr(record, "extra_data") and record.extra_data:
            parts.append(f"| {record.extra_data}")

        message = " ".join(parts)

        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)

        return message


class ContextLogger(logging.LoggerAdapter):
    """Logger adapter with context support."""

    def __init__(
        self,
        logger: logging.Logger,
        category: LogCategory = LogCategory.SYSTEM,
        correlation_id: str | None = None,
    ) -> None:
        """Initialize the context logger.

        Args:
            logger: Base logger instance.
            category: Log category.
            correlation_id: Optional correlation ID for request tracing.
        """
        super().__init__(logger, {})
        self.category = category
        self.correlation_id = correlation_id or str(uuid4())

    def process(
        self,
        msg: str,
        kwargs: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Process the logging call to add context.

        Args:
            msg: Log message.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple of (message, kwargs).
        """
        extra = kwargs.get("extra", {})
        extra["category"] = self.category.value
        extra["correlation_id"] = self.correlation_id
        kwargs["extra"] = extra
        return msg, kwargs

    def with_context(
        self,
        symbol: str | None = None,
        order_id: str | None = None,
        model_name: str | None = None,
        **extra_data: Any,
    ) -> "ContextLogger":
        """Create a new logger with additional context.

        Args:
            symbol: Trading symbol.
            order_id: Order ID.
            model_name: Model name.
            **extra_data: Additional context data.

        Returns:
            New ContextLogger with added context.
        """

        class EnhancedContextLogger(ContextLogger):
            def __init__(
                self,
                logger: logging.Logger,
                category: LogCategory,
                correlation_id: str,
                ctx_symbol: str | None,
                ctx_order_id: str | None,
                ctx_model_name: str | None,
                ctx_extra_data: dict[str, Any],
            ) -> None:
                super().__init__(logger, category, correlation_id)
                self._symbol = ctx_symbol
                self._order_id = ctx_order_id
                self._model_name = ctx_model_name
                self._extra_data = ctx_extra_data

            def process(
                self,
                msg: str,
                kwargs: dict[str, Any],
            ) -> tuple[str, dict[str, Any]]:
                msg, kwargs = super().process(msg, kwargs)
                extra = kwargs.get("extra", {})
                if self._symbol:
                    extra["symbol"] = self._symbol
                if self._order_id:
                    extra["order_id"] = self._order_id
                if self._model_name:
                    extra["model_name"] = self._model_name
                if self._extra_data:
                    extra["extra_data"] = self._extra_data
                kwargs["extra"] = extra
                return msg, kwargs

        return EnhancedContextLogger(
            self.logger,
            self.category,
            self.correlation_id,
            symbol,
            order_id,
            model_name,
            extra_data,
        )


class TradeLogger:
    """Specialized logger for trade events and audit trail."""

    def __init__(
        self,
        log_dir: Path | None = None,
        retention_days: int = 365,
    ) -> None:
        """Initialize the trade logger.

        Args:
            log_dir: Directory for trade logs.
            retention_days: Number of days to retain logs.
        """
        self.log_dir = log_dir or Path("logs/trades")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self._logger = get_logger("trade", LogCategory.TRADING)

    def log_trade_entry(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        signals: list[str] | None = None,
        model_predictions: dict[str, float] | None = None,
        risk_metrics: dict[str, float] | None = None,
        execution_details: dict[str, Any] | None = None,
    ) -> TradeLogEntry:
        """Log a trade entry.

        Args:
            symbol: Trading symbol.
            side: Trade side (BUY/SELL).
            quantity: Trade quantity.
            entry_price: Entry price.
            signals: List of signals that triggered the trade.
            model_predictions: Model predictions at entry.
            risk_metrics: Risk metrics at entry.
            execution_details: Execution details.

        Returns:
            TradeLogEntry for tracking.
        """
        entry = TradeLogEntry(
            entry_time=datetime.now(timezone.utc),
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            signals=signals or [],
            model_predictions=model_predictions or {},
            risk_metrics=risk_metrics or {},
            execution_details=execution_details or {},
        )

        self._logger.info(
            f"Trade entry: {side} {quantity} {symbol} @ {entry_price}",
            extra={
                "symbol": symbol,
                "extra_data": {
                    "side": side,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "signals": signals,
                    "model_predictions": model_predictions,
                },
            },
        )

        self._write_to_file(entry)
        return entry

    def log_trade_exit(
        self,
        entry: TradeLogEntry,
        exit_price: float,
        execution_details: dict[str, Any] | None = None,
    ) -> TradeLogEntry:
        """Log a trade exit.

        Args:
            entry: Original trade entry.
            exit_price: Exit price.
            execution_details: Execution details.

        Returns:
            Updated TradeLogEntry.
        """
        pnl = (exit_price - entry.entry_price) * entry.quantity
        if entry.side == "SELL":
            pnl = -pnl

        pnl_percent = (pnl / (entry.entry_price * entry.quantity)) * 100

        updated_entry = TradeLogEntry(
            entry_time=entry.entry_time,
            exit_time=datetime.now(timezone.utc),
            symbol=entry.symbol,
            side=entry.side,
            quantity=entry.quantity,
            entry_price=entry.entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_percent=pnl_percent,
            signals=entry.signals,
            model_predictions=entry.model_predictions,
            risk_metrics=entry.risk_metrics,
            execution_details={**entry.execution_details, **(execution_details or {})},
        )

        self._logger.info(
            f"Trade exit: {entry.side} {entry.quantity} {entry.symbol} @ {exit_price} | P&L: {pnl:.2f} ({pnl_percent:.2f}%)",
            extra={
                "symbol": entry.symbol,
                "extra_data": {
                    "entry_price": entry.entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "pnl_percent": pnl_percent,
                },
            },
        )

        self._write_to_file(updated_entry)
        return updated_entry

    def _write_to_file(self, entry: TradeLogEntry) -> None:
        """Write trade entry to file.

        Args:
            entry: Trade log entry to write.
        """
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        file_path = self.log_dir / f"trades_{date_str}.jsonl"

        with open(file_path, "a") as f:
            f.write(entry.model_dump_json() + "\n")


def setup_logging(
    level: str = "INFO",
    log_format: LogFormat = LogFormat.JSON,
    log_file: Path | None = None,
    rotation: str = "1 day",
    retention: str = "30 days",
) -> None:
    """Set up logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Log format (json or text).
        log_file: Optional log file path.
        rotation: Log rotation period.
        retention: Log retention period.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    if log_format == LogFormat.JSON:
        formatter = JsonFormatter()
    else:
        formatter = TextFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=30,
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            root_logger.warning(f"Failed to set up file logging: {e}")


@lru_cache(maxsize=32)
def get_logger(
    name: str,
    category: LogCategory = LogCategory.SYSTEM,
    correlation_id: str | None = None,
) -> ContextLogger:
    """Get a context logger for a component.

    Args:
        name: Logger name.
        category: Log category.
        correlation_id: Optional correlation ID.

    Returns:
        ContextLogger instance.
    """
    base_logger = logging.getLogger(name)
    return ContextLogger(base_logger, category, correlation_id)


def get_trade_logger(log_dir: Path | None = None) -> TradeLogger:
    """Get the trade logger instance.

    Args:
        log_dir: Optional log directory.

    Returns:
        TradeLogger instance.
    """
    return TradeLogger(log_dir)


# Convenience functions for quick logging
def log_system(message: str, level: str = "INFO", **kwargs: Any) -> None:
    """Log a system message."""
    logger = get_logger("system", LogCategory.SYSTEM)
    getattr(logger, level.lower())(message, extra={"extra_data": kwargs})


def log_data(message: str, level: str = "INFO", **kwargs: Any) -> None:
    """Log a data pipeline message."""
    logger = get_logger("data", LogCategory.DATA)
    getattr(logger, level.lower())(message, extra={"extra_data": kwargs})


def log_model(message: str, model_name: str, level: str = "INFO", **kwargs: Any) -> None:
    """Log a model message."""
    logger = get_logger("model", LogCategory.MODEL)
    getattr(logger, level.lower())(
        message,
        extra={"model_name": model_name, "extra_data": kwargs},
    )


def log_trading(message: str, symbol: str | None = None, level: str = "INFO", **kwargs: Any) -> None:
    """Log a trading message."""
    logger = get_logger("trading", LogCategory.TRADING)
    extra: dict[str, Any] = {"extra_data": kwargs}
    if symbol:
        extra["symbol"] = symbol
    getattr(logger, level.lower())(message, extra=extra)


def log_order(
    message: str,
    order_id: str,
    symbol: str | None = None,
    level: str = "INFO",
    **kwargs: Any,
) -> None:
    """Log an order message."""
    logger = get_logger("order", LogCategory.ORDER)
    extra: dict[str, Any] = {"order_id": order_id, "extra_data": kwargs}
    if symbol:
        extra["symbol"] = symbol
    getattr(logger, level.lower())(message, extra=extra)


def log_risk(message: str, level: str = "WARNING", **kwargs: Any) -> None:
    """Log a risk message."""
    logger = get_logger("risk", LogCategory.RISK)
    getattr(logger, level.lower())(message, extra={"extra_data": kwargs})


def log_alert(message: str, level: str = "WARNING", **kwargs: Any) -> None:
    """Log an alert message."""
    logger = get_logger("alert", LogCategory.ALERT)
    getattr(logger, level.lower())(message, extra={"extra_data": kwargs})


# =============================================================================
# TRACE Level Convenience Functions
# =============================================================================


def log_trace(
    category: LogCategory,
    message: str,
    **kwargs: Any,
) -> None:
    """Log a trace-level message for ultra-detailed debugging.

    Use for:
    - Individual tick/bar data
    - Feature calculation steps
    - Model inference details
    - Order book updates
    - Memory/performance profiling

    Args:
        category: Log category.
        message: Log message.
        **kwargs: Additional context data.
    """
    logger = get_logger(category.value.lower(), category)
    logger.logger.log(TRACE, message, extra={"extra_data": kwargs})


def trace_bar(symbol: str, bar_data: dict[str, Any]) -> None:
    """Trace-log a bar update.

    Args:
        symbol: Trading symbol.
        bar_data: Bar OHLCV data.
    """
    log_trace(
        LogCategory.DATA,
        f"Bar: {symbol}",
        symbol=symbol,
        **bar_data,
    )


def trace_feature(
    feature_name: str,
    value: float,
    computation_time_ms: float | None = None,
) -> None:
    """Trace-log a feature calculation.

    Args:
        feature_name: Name of the feature.
        value: Calculated value.
        computation_time_ms: Optional computation time in milliseconds.
    """
    data: dict[str, Any] = {"feature": feature_name, "value": value}
    if computation_time_ms is not None:
        data["computation_time_ms"] = computation_time_ms
    log_trace(LogCategory.DATA, f"Feature: {feature_name}={value:.6f}", **data)


def trace_model_layer(
    model_name: str,
    layer_name: str,
    output_shape: tuple[int, ...] | None = None,
    activation_stats: dict[str, float] | None = None,
) -> None:
    """Trace-log model layer execution.

    Args:
        model_name: Name of the model.
        layer_name: Name of the layer.
        output_shape: Output tensor shape.
        activation_stats: Activation statistics (min, max, mean, std).
    """
    data: dict[str, Any] = {"model": model_name, "layer": layer_name}
    if output_shape:
        data["output_shape"] = output_shape
    if activation_stats:
        data["activation_stats"] = activation_stats
    log_trace(LogCategory.MODEL, f"Layer: {model_name}.{layer_name}", **data)


def trace_order_book(
    symbol: str,
    bid: float,
    ask: float,
    bid_size: int,
    ask_size: int,
) -> None:
    """Trace-log order book update.

    Args:
        symbol: Trading symbol.
        bid: Best bid price.
        ask: Best ask price.
        bid_size: Bid size.
        ask_size: Ask size.
    """
    spread_bps = ((ask - bid) / bid) * 10000 if bid > 0 else 0
    log_trace(
        LogCategory.DATA,
        f"Book: {symbol} bid={bid:.2f}x{bid_size} ask={ask:.2f}x{ask_size} spread={spread_bps:.1f}bps",
        symbol=symbol,
        bid=bid,
        ask=ask,
        bid_size=bid_size,
        ask_size=ask_size,
        spread_bps=spread_bps,
    )


def trace_latency(
    operation: str,
    latency_ms: float,
    component: str | None = None,
) -> None:
    """Trace-log operation latency.

    Args:
        operation: Operation name.
        latency_ms: Latency in milliseconds.
        component: Optional component name.
    """
    category = LogCategory.SYSTEM
    message = f"Latency: {operation}={latency_ms:.3f}ms"
    if component:
        message = f"[{component}] {message}"
    log_trace(
        category,
        message,
        operation=operation,
        latency_ms=latency_ms,
        component=component,
    )
