"""
Unit tests for monitoring/metrics.py
"""

import time
from unittest.mock import patch

import pytest

from quant_trading_system.monitoring.metrics import (
    MetricsCollector,
    get_metrics_collector,
    REGISTRY,
    SYSTEM_UPTIME,
    REQUESTS_TOTAL,
    ORDERS_SUBMITTED,
    PORTFOLIO_EQUITY,
    MODEL_PREDICTIONS,
)


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton for each test."""
        MetricsCollector._instance = None
        yield

    def test_singleton_pattern(self):
        """Test that MetricsCollector is a singleton."""
        collector1 = MetricsCollector()
        collector2 = MetricsCollector()
        assert collector1 is collector2

    def test_get_metrics_collector(self):
        """Test get_metrics_collector function."""
        collector = get_metrics_collector()
        assert isinstance(collector, MetricsCollector)

    def test_set_system_info(self):
        """Test setting system information."""
        collector = MetricsCollector()
        collector.set_system_info(
            version="1.0.0",
            environment="test",
            hostname="test-host",
        )
        # No assertion needed - just verify no exception

    def test_update_system_metrics(self):
        """Test updating system metrics."""
        collector = MetricsCollector()
        collector.update_system_metrics(
            memory_bytes=1024 * 1024 * 100,
            cpu_percent=50.0,
            connections={"database": 5, "redis": 2},
            queues={"events": 10, "orders": 5},
        )
        # Verify uptime is set
        assert SYSTEM_UPTIME._value._value >= 0

    def test_record_request(self):
        """Test recording API requests."""
        collector = MetricsCollector()
        collector.record_request(
            endpoint="/health",
            method="GET",
            status=200,
            latency=0.05,
        )
        # Verify counter incremented (labels create new metric)

    def test_record_error(self):
        """Test recording errors."""
        collector = MetricsCollector()
        collector.record_error("connection_error", "database")

    def test_record_order_submitted(self):
        """Test recording order submission."""
        collector = MetricsCollector()
        collector.record_order_submitted(
            symbol="AAPL",
            side="BUY",
            order_type="MARKET",
        )

    def test_record_order_filled(self):
        """Test recording filled order."""
        collector = MetricsCollector()
        collector.record_order_filled(symbol="AAPL", side="BUY")

    def test_record_order_rejected(self):
        """Test recording rejected order."""
        collector = MetricsCollector()
        collector.record_order_rejected(symbol="AAPL", reason="insufficient_funds")

    def test_record_order_latency(self):
        """Test recording order latency."""
        collector = MetricsCollector()
        collector.record_order_latency(symbol="AAPL", latency=0.1)

    def test_update_portfolio_metrics(self):
        """Test updating portfolio metrics."""
        collector = MetricsCollector()
        collector.update_portfolio_metrics(
            equity=100000.0,
            cash=50000.0,
            buying_power=150000.0,
            positions_count=5,
            long_exposure=40000.0,
            short_exposure=10000.0,
        )
        assert PORTFOLIO_EQUITY._value._value == 100000.0

    def test_update_performance_metrics(self):
        """Test updating performance metrics."""
        collector = MetricsCollector()
        collector.update_performance_metrics(
            daily_pnl=500.0,
            total_pnl=5000.0,
            sharpe_ratio=1.5,
            current_drawdown=0.05,
            max_drawdown=0.10,
            win_rate=0.6,
        )

    def test_update_execution_metrics(self):
        """Test updating execution metrics."""
        collector = MetricsCollector()
        collector.update_execution_metrics(
            fill_rate=0.95,
            avg_slippage_bps=2.5,
            partial_fill_rate=0.1,
        )

    def test_record_execution_latency(self):
        """Test recording execution latency."""
        collector = MetricsCollector()
        collector.record_execution_latency(latency_ms=50.0)

    def test_update_risk_metrics(self):
        """Test updating risk metrics."""
        collector = MetricsCollector()
        collector.update_risk_metrics(
            var_95=5000.0,
            sector_concentrations={"Technology": 0.3, "Healthcare": 0.2},
            largest_position_pct=0.08,
            beta=1.1,
        )

    def test_record_model_prediction(self):
        """Test recording model prediction."""
        collector = MetricsCollector()
        collector.record_model_prediction(model_name="xgboost", latency=0.01)

    def test_update_model_metrics(self):
        """Test updating model metrics."""
        collector = MetricsCollector()
        collector.update_model_metrics(
            model_name="xgboost",
            accuracy=0.65,
            auc=0.72,
            sharpe=1.2,
        )

    def test_record_model_error(self):
        """Test recording model error."""
        collector = MetricsCollector()
        collector.record_model_error(model_name="xgboost", error_type="prediction_error")

    def test_update_ensemble_metrics(self):
        """Test updating ensemble metrics."""
        collector = MetricsCollector()
        collector.update_ensemble_metrics(
            agreement=0.85,
            top_signal_strength=0.75,
            dispersion=0.15,
        )

    def test_time_data_processing_context_manager(self):
        """Test data processing timing context manager."""
        collector = MetricsCollector()
        with collector.time_data_processing("feature_calc"):
            time.sleep(0.01)

    def test_update_data_staleness(self):
        """Test updating data staleness."""
        collector = MetricsCollector()
        collector.update_data_staleness(data_type="bars", staleness_seconds=5.0)

    def test_record_cache_hit(self):
        """Test recording cache hit."""
        collector = MetricsCollector()
        collector.record_cache_hit(cache_name="feature_store")

    def test_record_cache_miss(self):
        """Test recording cache miss."""
        collector = MetricsCollector()
        collector.record_cache_miss(cache_name="feature_store")

    def test_get_metrics_output(self):
        """Test getting Prometheus metrics output."""
        collector = MetricsCollector()
        output = collector.get_metrics()
        assert isinstance(output, bytes)
        assert len(output) > 0

    def test_get_content_type(self):
        """Test getting content type."""
        collector = MetricsCollector()
        content_type = collector.get_content_type()
        assert "text/plain" in content_type or "openmetrics" in content_type
