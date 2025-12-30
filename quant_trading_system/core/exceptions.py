"""
Custom exception hierarchy for the trading system.

Provides a structured exception hierarchy for different error categories:
- Data errors (loading, validation, corruption)
- Model errors (training, prediction, loading)
- Execution errors (order submission, cancellation)
- Risk errors (limit breaches, margin calls)
- Configuration errors (invalid, missing, parsing)
"""

from __future__ import annotations

from typing import Any


class TradingSystemError(Exception):
    """Base exception for all trading system errors.

    All custom exceptions in the system should inherit from this class.
    Provides structured error information including error code, message,
    and additional context.
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message.
            error_code: Optional error code for programmatic handling.
            details: Optional dictionary with additional error context.
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}

    def __str__(self) -> str:
        """Return formatted error string."""
        if self.details:
            return f"[{self.error_code}] {self.message} - Details: {self.details}"
        return f"[{self.error_code}] {self.message}"

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(TradingSystemError):
    """Raised when parameter or input validation fails.

    This is a general-purpose validation error used by validation decorators
    and input validation throughout the system.

    Examples:
        - Invalid order parameters (negative quantity, invalid symbol)
        - ML model input validation failures (NaN values, wrong dimensions)
        - Configuration validation failures
    """

    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        invalid_value: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the validation error.

        Args:
            message: Human-readable error message.
            field_name: Name of the field that failed validation.
            invalid_value: The value that failed validation.
            **kwargs: Additional context passed to parent.
        """
        details = kwargs.pop("details", {})
        if field_name:
            details["field_name"] = field_name
        if invalid_value is not None:
            details["invalid_value"] = str(invalid_value)
        super().__init__(message, details=details, **kwargs)
        self.field_name = field_name
        self.invalid_value = invalid_value


# =============================================================================
# Data Errors
# =============================================================================


class DataError(TradingSystemError):
    """Base exception for data-related errors."""

    pass


class DataNotFoundError(DataError):
    """Raised when requested data is not found.

    Examples:
        - Historical data file not found
        - Symbol not in database
        - Missing feature data
    """

    def __init__(
        self,
        message: str,
        symbol: str | None = None,
        data_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if symbol:
            details["symbol"] = symbol
        if data_type:
            details["data_type"] = data_type
        super().__init__(message, details=details, **kwargs)
        self.symbol = symbol
        self.data_type = data_type


class DataValidationError(DataError):
    """Raised when data fails validation checks.

    Examples:
        - Invalid OHLC relationship
        - Missing required fields
        - Out-of-range values
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        expected: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        if expected:
            details["expected"] = expected
        super().__init__(message, details=details, **kwargs)
        self.field = field
        self.value = value
        self.expected = expected


class DataCorruptionError(DataError):
    """Raised when data appears to be corrupted.

    Examples:
        - Checksum mismatch
        - Unreadable file format
        - Inconsistent data structure
    """

    def __init__(
        self,
        message: str,
        source: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if source:
            details["source"] = source
        super().__init__(message, details=details, **kwargs)
        self.source = source


class DataConnectionError(DataError):
    """Raised when unable to connect to data source.

    Examples:
        - Database connection failed
        - API endpoint unreachable
        - Network timeout
    """

    def __init__(
        self,
        message: str,
        source: str | None = None,
        retry_after: int | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if source:
            details["source"] = source
        if retry_after is not None:
            details["retry_after"] = retry_after
        super().__init__(message, details=details, **kwargs)
        self.source = source
        self.retry_after = retry_after


class DataLoadError(DataError):
    """Raised when data loading fails.

    Examples:
        - File read error
        - API fetch failed
        - Data format incompatible
    """

    def __init__(
        self,
        message: str,
        source: str | None = None,
        symbol: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if source:
            details["source"] = source
        if symbol:
            details["symbol"] = symbol
        super().__init__(message, details=details, **kwargs)
        self.source = source
        self.symbol = symbol


class DataStreamError(DataError):
    """Raised when real-time data streaming fails.

    Examples:
        - WebSocket connection dropped
        - Stream timeout
        - Malformed stream data
    """

    def __init__(
        self,
        message: str,
        stream_type: str | None = None,
        symbols: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if stream_type:
            details["stream_type"] = stream_type
        if symbols:
            details["symbols"] = symbols
        super().__init__(message, details=details, **kwargs)
        self.stream_type = stream_type
        self.symbols = symbols


# =============================================================================
# Model Errors
# =============================================================================


class ModelError(TradingSystemError):
    """Base exception for model-related errors."""

    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested model is not found.

    Examples:
        - Model file doesn't exist
        - Model not registered in registry
        - Version not available
    """

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        version: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if model_name:
            details["model_name"] = model_name
        if version:
            details["version"] = version
        super().__init__(message, details=details, **kwargs)
        self.model_name = model_name
        self.version = version


class ModelLoadError(ModelError):
    """Raised when a model fails to load.

    Examples:
        - Incompatible model format
        - Missing dependencies
        - Corrupted model file
    """

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        model_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if model_name:
            details["model_name"] = model_name
        if model_path:
            details["model_path"] = model_path
        super().__init__(message, details=details, **kwargs)
        self.model_name = model_name
        self.model_path = model_path


class PredictionError(ModelError):
    """Raised when model prediction fails.

    Examples:
        - Invalid input features
        - Model inference error
        - Output validation failed
    """

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        symbol: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if model_name:
            details["model_name"] = model_name
        if symbol:
            details["symbol"] = symbol
        super().__init__(message, details=details, **kwargs)
        self.model_name = model_name
        self.symbol = symbol


class TrainingError(ModelError):
    """Raised when model training fails.

    Examples:
        - Insufficient training data
        - Training diverged
        - Resource exhausted
    """

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        epoch: int | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if model_name:
            details["model_name"] = model_name
        if epoch is not None:
            details["epoch"] = epoch
        super().__init__(message, details=details, **kwargs)
        self.model_name = model_name
        self.epoch = epoch


# =============================================================================
# Execution Errors
# =============================================================================


class ExecutionError(TradingSystemError):
    """Base exception for execution-related errors."""

    pass


class OrderSubmissionError(ExecutionError):
    """Raised when order submission fails.

    Examples:
        - Invalid order parameters
        - Market closed
        - Symbol not tradeable
    """

    def __init__(
        self,
        message: str,
        order_id: str | None = None,
        symbol: str | None = None,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if order_id:
            details["order_id"] = order_id
        if symbol:
            details["symbol"] = symbol
        if reason:
            details["reason"] = reason
        super().__init__(message, details=details, **kwargs)
        self.order_id = order_id
        self.symbol = symbol
        self.reason = reason


class OrderCancellationError(ExecutionError):
    """Raised when order cancellation fails.

    Examples:
        - Order already filled
        - Order not found
        - Cancellation rejected
    """

    def __init__(
        self,
        message: str,
        order_id: str | None = None,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if order_id:
            details["order_id"] = order_id
        if reason:
            details["reason"] = reason
        super().__init__(message, details=details, **kwargs)
        self.order_id = order_id
        self.reason = reason


class InsufficientFundsError(ExecutionError):
    """Raised when there are insufficient funds for an order.

    Examples:
        - Not enough buying power
        - Margin requirement not met
        - Cash balance too low
    """

    def __init__(
        self,
        message: str,
        required: float | None = None,
        available: float | None = None,
        symbol: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if required is not None:
            details["required"] = required
        if available is not None:
            details["available"] = available
        if symbol:
            details["symbol"] = symbol
        super().__init__(message, details=details, **kwargs)
        self.required = required
        self.available = available
        self.symbol = symbol


class BrokerConnectionError(ExecutionError):
    """Raised when broker connection fails.

    Examples:
        - API authentication failed
        - Network connectivity lost
        - Broker service unavailable
    """

    def __init__(
        self,
        message: str,
        broker: str | None = None,
        retry_after: int | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if broker:
            details["broker"] = broker
        if retry_after is not None:
            details["retry_after"] = retry_after
        super().__init__(message, details=details, **kwargs)
        self.broker = broker
        self.retry_after = retry_after


# =============================================================================
# Risk Errors
# =============================================================================


class RiskError(TradingSystemError):
    """Base exception for risk-related errors."""

    pass


class PositionLimitError(RiskError):
    """Raised when position limit is breached.

    Examples:
        - Max position size exceeded
        - Max number of positions exceeded
        - Concentration limit breached
    """

    def __init__(
        self,
        message: str,
        symbol: str | None = None,
        current_value: float | None = None,
        limit_value: float | None = None,
        limit_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if symbol:
            details["symbol"] = symbol
        if current_value is not None:
            details["current_value"] = current_value
        if limit_value is not None:
            details["limit_value"] = limit_value
        if limit_type:
            details["limit_type"] = limit_type
        super().__init__(message, details=details, **kwargs)
        self.symbol = symbol
        self.current_value = current_value
        self.limit_value = limit_value
        self.limit_type = limit_type


class DrawdownLimitError(RiskError):
    """Raised when drawdown limit is breached.

    Examples:
        - Daily loss limit exceeded
        - Maximum drawdown breached
        - Weekly loss limit exceeded
    """

    def __init__(
        self,
        message: str,
        current_drawdown: float | None = None,
        limit: float | None = None,
        period: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if current_drawdown is not None:
            details["current_drawdown"] = current_drawdown
        if limit is not None:
            details["limit"] = limit
        if period:
            details["period"] = period
        super().__init__(message, details=details, **kwargs)
        self.current_drawdown = current_drawdown
        self.limit = limit
        self.period = period


class ExposureLimitError(RiskError):
    """Raised when exposure limit is breached.

    Examples:
        - Sector exposure too high
        - Correlation exposure exceeded
        - Gross exposure limit breached
    """

    def __init__(
        self,
        message: str,
        exposure_type: str | None = None,
        current_exposure: float | None = None,
        limit: float | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if exposure_type:
            details["exposure_type"] = exposure_type
        if current_exposure is not None:
            details["current_exposure"] = current_exposure
        if limit is not None:
            details["limit"] = limit
        super().__init__(message, details=details, **kwargs)
        self.exposure_type = exposure_type
        self.current_exposure = current_exposure
        self.limit = limit


class MarginCallError(RiskError):
    """Raised when margin call occurs.

    Examples:
        - Maintenance margin not met
        - Margin requirement increased
        - Forced liquidation imminent
    """

    def __init__(
        self,
        message: str,
        margin_required: float | None = None,
        margin_available: float | None = None,
        deadline: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if margin_required is not None:
            details["margin_required"] = margin_required
        if margin_available is not None:
            details["margin_available"] = margin_available
        if deadline:
            details["deadline"] = deadline
        super().__init__(message, details=details, **kwargs)
        self.margin_required = margin_required
        self.margin_available = margin_available
        self.deadline = deadline


class KillSwitchActiveError(RiskError):
    """Raised when trading is attempted while kill switch is active.

    This is a critical safety error that prevents ALL trading operations
    when the kill switch has been triggered due to:
    - Maximum drawdown breach
    - Rapid P&L decline
    - System errors
    - Data feed failures
    - Manual activation

    Examples:
        - Order submission while kill switch active
        - Position modification while halted
    """

    def __init__(
        self,
        message: str = "Trading halted - kill switch is active",
        reason: str | None = None,
        activated_at: str | None = None,
        activated_by: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if reason:
            details["reason"] = reason
        if activated_at:
            details["activated_at"] = activated_at
        if activated_by:
            details["activated_by"] = activated_by
        super().__init__(message, error_code="KILL_SWITCH_ACTIVE", details=details, **kwargs)
        self.reason = reason
        self.activated_at = activated_at
        self.activated_by = activated_by


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(TradingSystemError):
    """Base exception for configuration-related errors."""

    pass


class InvalidConfigError(ConfigurationError):
    """Raised when configuration is invalid.

    Examples:
        - Invalid parameter value
        - Conflicting settings
        - Schema validation failed
    """

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        value: Any = None,
        expected: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if config_key:
            details["config_key"] = config_key
        if value is not None:
            details["value"] = str(value)
        if expected:
            details["expected"] = expected
        super().__init__(message, details=details, **kwargs)
        self.config_key = config_key
        self.value = value
        self.expected = expected


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing.

    Examples:
        - Required environment variable not set
        - Configuration file not found
        - Required key missing from config
    """

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        config_file: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if config_key:
            details["config_key"] = config_key
        if config_file:
            details["config_file"] = config_file
        super().__init__(message, details=details, **kwargs)
        self.config_key = config_key
        self.config_file = config_file


class ConfigParseError(ConfigurationError):
    """Raised when configuration parsing fails.

    Examples:
        - Invalid YAML syntax
        - JSON parse error
        - Type conversion failed
    """

    def __init__(
        self,
        message: str,
        config_file: str | None = None,
        line_number: int | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if config_file:
            details["config_file"] = config_file
        if line_number is not None:
            details["line_number"] = line_number
        super().__init__(message, details=details, **kwargs)
        self.config_file = config_file
        self.line_number = line_number
