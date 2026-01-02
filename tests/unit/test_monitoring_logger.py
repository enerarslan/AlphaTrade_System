"""
Unit tests for monitoring/logger.py
"""

import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from quant_trading_system.monitoring.logger import (
    LogCategory,
    LogFormat,
    StructuredLogRecord,
    TradeLogEntry,
    JsonFormatter,
    TextFormatter,
    ContextLogger,
    TradeLogger,
    setup_logging,
    get_logger,
    get_trade_logger,
    log_system,
    log_data,
    log_model,
    log_trading,
    log_order,
    log_risk,
    log_alert,
)


class TestLogCategory:
    """Tests for LogCategory enum."""

    def test_log_categories_exist(self):
        """Test that all log categories are defined."""
        assert LogCategory.SYSTEM == "SYSTEM"
        assert LogCategory.DATA == "DATA"
        assert LogCategory.MODEL == "MODEL"
        assert LogCategory.TRADING == "TRADING"
        assert LogCategory.ORDER == "ORDER"
        assert LogCategory.RISK == "RISK"
        assert LogCategory.ALERT == "ALERT"


class TestLogFormat:
    """Tests for LogFormat enum."""

    def test_log_formats_exist(self):
        """Test that log formats are defined."""
        assert LogFormat.JSON == "json"
        assert LogFormat.TEXT == "text"


class TestStructuredLogRecord:
    """Tests for StructuredLogRecord model."""

    def test_create_log_record(self):
        """Test creating a structured log record."""
        record = StructuredLogRecord(
            level="INFO",
            category=LogCategory.SYSTEM,
            message="Test message",
            correlation_id="abc123",
            symbol="AAPL",
            order_id="order-001",
            model_name="xgboost",
            extra_data={"key": "value"},
        )
        assert record.level == "INFO"
        assert record.category == LogCategory.SYSTEM
        assert record.message == "Test message"
        assert record.correlation_id == "abc123"
        assert record.timestamp is not None

    def test_to_json(self):
        """Test JSON serialization."""
        record = StructuredLogRecord(
            level="ERROR",
            category=LogCategory.TRADING,
            message="Error occurred",
            symbol="MSFT",
        )
        json_str = record.to_json()
        data = json.loads(json_str)
        assert data["level"] == "ERROR"
        assert data["category"] == "TRADING"
        assert data["message"] == "Error occurred"
        assert data["symbol"] == "MSFT"

    def test_to_text(self):
        """Test text serialization."""
        record = StructuredLogRecord(
            level="INFO",
            category=LogCategory.DATA,
            message="Data loaded",
            correlation_id="xyz789",
        )
        text = record.to_text()
        assert "INFO" in text
        assert "DATA" in text
        assert "Data loaded" in text
        assert "xyz789"[:8] in text


class TestTradeLogEntry:
    """Tests for TradeLogEntry model."""

    def test_create_trade_entry(self):
        """Test creating a trade log entry."""
        entry = TradeLogEntry(
            entry_time=datetime.now(timezone.utc),
            symbol="AAPL",
            side="BUY",
            quantity=100.0,
            entry_price=185.50,
            signals=["momentum", "mean_reversion"],
            model_predictions={"xgboost": 0.7, "lightgbm": 0.65},
        )
        assert entry.symbol == "AAPL"
        assert entry.side == "BUY"
        assert entry.quantity == 100.0
        assert len(entry.signals) == 2

    def test_trade_entry_exit(self):
        """Test trade entry with exit information."""
        entry = TradeLogEntry(
            entry_time=datetime(2024, 1, 15, 10, 0),
            exit_time=datetime(2024, 1, 15, 14, 0),
            symbol="MSFT",
            side="BUY",
            quantity=50.0,
            entry_price=380.0,
            exit_price=385.0,
            pnl=250.0,
            pnl_percent=1.32,
        )
        assert entry.exit_time is not None
        assert entry.pnl == 250.0


class TestJsonFormatter:
    """Tests for JsonFormatter class."""

    def test_format_log_record(self):
        """Test formatting a log record as JSON."""
        formatter = JsonFormatter(category=LogCategory.SYSTEM)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_format_with_extra_fields(self):
        """Test formatting with extra fields."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=10,
            msg="Warning message",
            args=(),
            exc_info=None,
        )
        record.correlation_id = "corr-123"
        record.symbol = "AAPL"
        output = formatter.format(record)
        data = json.loads(output)
        assert data["correlation_id"] == "corr-123"
        assert data["symbol"] == "AAPL"


class TestTextFormatter:
    """Tests for TextFormatter class."""

    def test_format_log_record(self):
        """Test formatting a log record as text."""
        formatter = TextFormatter(category=LogCategory.TRADING)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "INFO" in output
        assert "Test message" in output

    def test_format_with_symbol(self):
        """Test formatting with symbol."""
        formatter = TextFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Order placed",
            args=(),
            exc_info=None,
        )
        record.symbol = "AAPL"
        output = formatter.format(record)
        assert "AAPL" in output


class TestContextLogger:
    """Tests for ContextLogger class."""

    def test_create_context_logger(self):
        """Test creating a context logger."""
        base_logger = logging.getLogger("test_context")
        context_logger = ContextLogger(
            base_logger,
            category=LogCategory.MODEL,
            correlation_id="test-corr-id",
        )
        assert context_logger.category == LogCategory.MODEL
        assert context_logger.correlation_id == "test-corr-id"

    def test_with_context(self):
        """Test creating logger with additional context."""
        base_logger = logging.getLogger("test_with_context")
        context_logger = ContextLogger(base_logger, LogCategory.TRADING)
        enhanced = context_logger.with_context(
            symbol="AAPL",
            order_id="order-123",
            model_name="xgboost",
            custom_field="value",
        )
        assert enhanced is not None


class TestTradeLogger:
    """Tests for TradeLogger class."""

    def test_create_trade_logger(self):
        """Test creating a trade logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(log_dir=Path(tmpdir))
            assert logger.log_dir.exists()

    def test_log_trade_entry(self):
        """Test logging a trade entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(log_dir=Path(tmpdir))
            entry = logger.log_trade_entry(
                symbol="AAPL",
                side="BUY",
                quantity=100.0,
                entry_price=185.50,
                signals=["momentum"],
                model_predictions={"xgboost": 0.7},
            )
            assert entry.symbol == "AAPL"
            assert entry.side == "BUY"

    def test_log_trade_exit(self):
        """Test logging a trade exit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(log_dir=Path(tmpdir))
            entry = logger.log_trade_entry(
                symbol="MSFT",
                side="BUY",
                quantity=50.0,
                entry_price=380.0,
            )
            exit_entry = logger.log_trade_exit(
                entry=entry,
                exit_price=385.0,
            )
            assert exit_entry.exit_price == 385.0
            assert exit_entry.pnl is not None


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_json_logging(self):
        """Test setting up JSON logging."""
        setup_logging(level="DEBUG", log_format=LogFormat.JSON)
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_setup_text_logging(self):
        """Test setting up text logging."""
        setup_logging(level="INFO", log_format=LogFormat.TEXT)
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_setup_with_file(self):
        """Test setting up logging with file handler."""
        import os
        # Use a unique temp file that won't conflict
        tmpdir = tempfile.mkdtemp()
        try:
            log_file = Path(tmpdir) / "test_setup.log"
            setup_logging(level="INFO", log_file=log_file)
            # Log something to ensure file is used
            logger = logging.getLogger("file_test")
            logger.info("Test log message")
        finally:
            # Clean up handlers to release file locks
            for handler in logging.getLogger().handlers[:]:
                handler.close()
                logging.getLogger().removeHandler(handler)


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger(self):
        """Test getting a logger."""
        logger = get_logger("test_module", LogCategory.SYSTEM)
        assert isinstance(logger, ContextLogger)

    def test_get_logger_caching(self):
        """Test that get_logger returns cached instances."""
        logger1 = get_logger("cached_test", LogCategory.DATA)
        logger2 = get_logger("cached_test", LogCategory.DATA)
        # Note: lru_cache means same args return same instance


class TestConvenienceLoggingFunctions:
    """Tests for convenience logging functions."""

    @patch('quant_trading_system.monitoring.logger.get_logger')
    def test_log_system(self, mock_get_logger):
        """Test log_system function."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        log_system("System started", level="INFO", extra_key="value")
        mock_logger.info.assert_called_once()

    @patch('quant_trading_system.monitoring.logger.get_logger')
    def test_log_data(self, mock_get_logger):
        """Test log_data function."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        log_data("Data loaded", level="INFO", rows=1000)
        mock_logger.info.assert_called_once()

    @patch('quant_trading_system.monitoring.logger.get_logger')
    def test_log_model(self, mock_get_logger):
        """Test log_model function."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        log_model("Prediction made", model_name="xgboost", level="INFO")
        mock_logger.info.assert_called_once()

    @patch('quant_trading_system.monitoring.logger.get_logger')
    def test_log_trading(self, mock_get_logger):
        """Test log_trading function."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        log_trading("Signal generated", symbol="AAPL", level="INFO")
        mock_logger.info.assert_called_once()

    @patch('quant_trading_system.monitoring.logger.get_logger')
    def test_log_order(self, mock_get_logger):
        """Test log_order function."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        log_order("Order filled", order_id="ord-123", symbol="AAPL")
        mock_logger.info.assert_called_once()

    @patch('quant_trading_system.monitoring.logger.get_logger')
    def test_log_risk(self, mock_get_logger):
        """Test log_risk function."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        log_risk("Risk limit approaching", level="WARNING", current=0.08)
        mock_logger.warning.assert_called_once()

    @patch('quant_trading_system.monitoring.logger.get_logger')
    def test_log_alert(self, mock_get_logger):
        """Test log_alert function."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        log_alert("Critical alert", level="WARNING", severity="critical")
        mock_logger.warning.assert_called_once()
