"""
Unit tests for core/exceptions.py
"""

import pytest

from quant_trading_system.core.exceptions import (
    BrokerConnectionError,
    ConfigParseError,
    ConfigurationError,
    DataConnectionError,
    DataCorruptionError,
    DataError,
    DataNotFoundError,
    DataValidationError,
    DrawdownLimitError,
    ExecutionError,
    ExposureLimitError,
    InsufficientFundsError,
    InvalidConfigError,
    MarginCallError,
    MissingConfigError,
    ModelError,
    ModelLoadError,
    ModelNotFoundError,
    OrderCancellationError,
    OrderSubmissionError,
    PositionLimitError,
    PredictionError,
    RiskError,
    TrainingError,
    TradingSystemError,
)


class TestTradingSystemError:
    """Tests for base TradingSystemError."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = TradingSystemError("Something went wrong")
        assert str(error) == "[TradingSystemError] Something went wrong"
        assert error.message == "Something went wrong"
        assert error.error_code == "TradingSystemError"

    def test_error_with_code(self):
        """Test error with custom error code."""
        error = TradingSystemError("Failed", error_code="ERR001")
        assert error.error_code == "ERR001"
        assert "[ERR001]" in str(error)

    def test_error_with_details(self):
        """Test error with details."""
        error = TradingSystemError(
            "Operation failed",
            details={"key": "value", "count": 42}
        )
        assert error.details["key"] == "value"
        assert "Details:" in str(error)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        error = TradingSystemError(
            "Test error",
            error_code="TEST001",
            details={"foo": "bar"}
        )
        d = error.to_dict()
        assert d["error_type"] == "TradingSystemError"
        assert d["error_code"] == "TEST001"
        assert d["message"] == "Test error"
        assert d["details"]["foo"] == "bar"


class TestDataErrors:
    """Tests for data-related errors."""

    def test_data_not_found_error(self):
        """Test DataNotFoundError."""
        error = DataNotFoundError(
            "Data not found",
            symbol="AAPL",
            data_type="historical"
        )
        assert error.symbol == "AAPL"
        assert error.data_type == "historical"
        assert error.details["symbol"] == "AAPL"
        assert isinstance(error, DataError)
        assert isinstance(error, TradingSystemError)

    def test_data_validation_error(self):
        """Test DataValidationError."""
        error = DataValidationError(
            "Invalid price",
            field="close",
            value=-100.0,
            expected="positive number"
        )
        assert error.field == "close"
        assert error.value == -100.0
        assert error.expected == "positive number"

    def test_data_corruption_error(self):
        """Test DataCorruptionError."""
        error = DataCorruptionError(
            "File corrupted",
            source="/path/to/file.parquet"
        )
        assert error.source == "/path/to/file.parquet"

    def test_data_connection_error(self):
        """Test DataConnectionError."""
        error = DataConnectionError(
            "Connection failed",
            source="PostgreSQL",
            retry_after=30
        )
        assert error.source == "PostgreSQL"
        assert error.retry_after == 30


class TestModelErrors:
    """Tests for model-related errors."""

    def test_model_not_found_error(self):
        """Test ModelNotFoundError."""
        error = ModelNotFoundError(
            "Model not found",
            model_name="xgboost_v1",
            version="1.0.0"
        )
        assert error.model_name == "xgboost_v1"
        assert error.version == "1.0.0"
        assert isinstance(error, ModelError)

    def test_model_load_error(self):
        """Test ModelLoadError."""
        error = ModelLoadError(
            "Failed to load model",
            model_name="lstm_v2",
            model_path="/models/lstm_v2.pt"
        )
        assert error.model_name == "lstm_v2"
        assert error.model_path == "/models/lstm_v2.pt"

    def test_prediction_error(self):
        """Test PredictionError."""
        error = PredictionError(
            "Prediction failed",
            model_name="ensemble",
            symbol="MSFT"
        )
        assert error.model_name == "ensemble"
        assert error.symbol == "MSFT"

    def test_training_error(self):
        """Test TrainingError."""
        error = TrainingError(
            "Training diverged",
            model_name="transformer",
            epoch=50
        )
        assert error.model_name == "transformer"
        assert error.epoch == 50


class TestExecutionErrors:
    """Tests for execution-related errors."""

    def test_order_submission_error(self):
        """Test OrderSubmissionError."""
        error = OrderSubmissionError(
            "Order rejected",
            order_id="ORD-12345",
            symbol="AAPL",
            reason="Market closed"
        )
        assert error.order_id == "ORD-12345"
        assert error.symbol == "AAPL"
        assert error.reason == "Market closed"
        assert isinstance(error, ExecutionError)

    def test_order_cancellation_error(self):
        """Test OrderCancellationError."""
        error = OrderCancellationError(
            "Cannot cancel",
            order_id="ORD-67890",
            reason="Already filled"
        )
        assert error.order_id == "ORD-67890"
        assert error.reason == "Already filled"

    def test_insufficient_funds_error(self):
        """Test InsufficientFundsError."""
        error = InsufficientFundsError(
            "Not enough buying power",
            required=50000.0,
            available=25000.0,
            symbol="TSLA"
        )
        assert error.required == 50000.0
        assert error.available == 25000.0
        assert error.symbol == "TSLA"

    def test_broker_connection_error(self):
        """Test BrokerConnectionError."""
        error = BrokerConnectionError(
            "Connection lost",
            broker="Alpaca",
            retry_after=60
        )
        assert error.broker == "Alpaca"
        assert error.retry_after == 60


class TestRiskErrors:
    """Tests for risk-related errors."""

    def test_position_limit_error(self):
        """Test PositionLimitError."""
        error = PositionLimitError(
            "Position limit exceeded",
            symbol="NVDA",
            current_value=15000.0,
            limit_value=10000.0,
            limit_type="max_position_value"
        )
        assert error.symbol == "NVDA"
        assert error.current_value == 15000.0
        assert error.limit_value == 10000.0
        assert error.limit_type == "max_position_value"
        assert isinstance(error, RiskError)

    def test_drawdown_limit_error(self):
        """Test DrawdownLimitError."""
        error = DrawdownLimitError(
            "Maximum drawdown breached",
            current_drawdown=0.18,
            limit=0.15,
            period="monthly"
        )
        assert error.current_drawdown == 0.18
        assert error.limit == 0.15
        assert error.period == "monthly"

    def test_exposure_limit_error(self):
        """Test ExposureLimitError."""
        error = ExposureLimitError(
            "Sector exposure too high",
            exposure_type="technology",
            current_exposure=0.35,
            limit=0.25
        )
        assert error.exposure_type == "technology"
        assert error.current_exposure == 0.35
        assert error.limit == 0.25

    def test_margin_call_error(self):
        """Test MarginCallError."""
        error = MarginCallError(
            "Margin call issued",
            margin_required=100000.0,
            margin_available=75000.0,
            deadline="2024-01-15T16:00:00"
        )
        assert error.margin_required == 100000.0
        assert error.margin_available == 75000.0
        assert error.deadline == "2024-01-15T16:00:00"


class TestConfigurationErrors:
    """Tests for configuration-related errors."""

    def test_invalid_config_error(self):
        """Test InvalidConfigError."""
        error = InvalidConfigError(
            "Invalid value",
            config_key="max_position_pct",
            value=1.5,
            expected="value between 0 and 1"
        )
        assert error.config_key == "max_position_pct"
        assert error.value == 1.5
        assert error.expected == "value between 0 and 1"
        assert isinstance(error, ConfigurationError)

    def test_missing_config_error(self):
        """Test MissingConfigError."""
        error = MissingConfigError(
            "Required config missing",
            config_key="ALPACA_API_KEY",
            config_file=".env"
        )
        assert error.config_key == "ALPACA_API_KEY"
        assert error.config_file == ".env"

    def test_config_parse_error(self):
        """Test ConfigParseError."""
        error = ConfigParseError(
            "YAML parse error",
            config_file="config/settings.yaml",
            line_number=42
        )
        assert error.config_file == "config/settings.yaml"
        assert error.line_number == 42


class TestExceptionHierarchy:
    """Tests for exception inheritance hierarchy."""

    def test_data_errors_inherit_from_trading_system_error(self):
        """Test data errors inheritance."""
        assert issubclass(DataError, TradingSystemError)
        assert issubclass(DataNotFoundError, DataError)
        assert issubclass(DataValidationError, DataError)
        assert issubclass(DataCorruptionError, DataError)
        assert issubclass(DataConnectionError, DataError)

    def test_model_errors_inherit_from_trading_system_error(self):
        """Test model errors inheritance."""
        assert issubclass(ModelError, TradingSystemError)
        assert issubclass(ModelNotFoundError, ModelError)
        assert issubclass(ModelLoadError, ModelError)
        assert issubclass(PredictionError, ModelError)
        assert issubclass(TrainingError, ModelError)

    def test_execution_errors_inherit_from_trading_system_error(self):
        """Test execution errors inheritance."""
        assert issubclass(ExecutionError, TradingSystemError)
        assert issubclass(OrderSubmissionError, ExecutionError)
        assert issubclass(OrderCancellationError, ExecutionError)
        assert issubclass(InsufficientFundsError, ExecutionError)
        assert issubclass(BrokerConnectionError, ExecutionError)

    def test_risk_errors_inherit_from_trading_system_error(self):
        """Test risk errors inheritance."""
        assert issubclass(RiskError, TradingSystemError)
        assert issubclass(PositionLimitError, RiskError)
        assert issubclass(DrawdownLimitError, RiskError)
        assert issubclass(ExposureLimitError, RiskError)
        assert issubclass(MarginCallError, RiskError)

    def test_config_errors_inherit_from_trading_system_error(self):
        """Test configuration errors inheritance."""
        assert issubclass(ConfigurationError, TradingSystemError)
        assert issubclass(InvalidConfigError, ConfigurationError)
        assert issubclass(MissingConfigError, ConfigurationError)
        assert issubclass(ConfigParseError, ConfigurationError)

    def test_can_catch_by_base_class(self):
        """Test that errors can be caught by base class."""
        with pytest.raises(TradingSystemError):
            raise DataNotFoundError("Not found")

        with pytest.raises(DataError):
            raise DataValidationError("Invalid")

        with pytest.raises(TradingSystemError):
            raise OrderSubmissionError("Failed")
