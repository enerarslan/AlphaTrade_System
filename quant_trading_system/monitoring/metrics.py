"""
Prometheus metrics collection for the trading system.

Provides comprehensive metrics for:
- System health (CPU, memory, uptime)
- Trading performance (P&L, positions, execution)
- Model performance (accuracy, latency, predictions)
- Risk monitoring (VaR, drawdown, exposure)
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


# Create a custom registry to avoid conflicts
REGISTRY = CollectorRegistry(auto_describe=True)


# =============================================================================
# System Metrics
# =============================================================================

SYSTEM_INFO = Info(
    "trading_system",
    "Trading system information",
    registry=REGISTRY,
)

SYSTEM_UPTIME = Gauge(
    "system_uptime_seconds",
    "System uptime in seconds",
    registry=REGISTRY,
)

MEMORY_USAGE = Gauge(
    "memory_usage_bytes",
    "Memory usage in bytes",
    ["type"],  # heap, rss, etc.
    registry=REGISTRY,
)

CPU_USAGE = Gauge(
    "cpu_usage_percent",
    "CPU usage percentage",
    registry=REGISTRY,
)

OPEN_CONNECTIONS = Gauge(
    "open_connections",
    "Number of open connections",
    ["type"],  # database, redis, broker
    registry=REGISTRY,
)

QUEUE_DEPTH = Gauge(
    "queue_depth",
    "Queue depth",
    ["queue_name"],
    registry=REGISTRY,
)


# =============================================================================
# Request Metrics
# =============================================================================

REQUESTS_TOTAL = Counter(
    "requests_total",
    "Total number of requests",
    ["endpoint", "method", "status"],
    registry=REGISTRY,
)

ERRORS_TOTAL = Counter(
    "errors_total",
    "Total number of errors",
    ["error_type", "component"],
    registry=REGISTRY,
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)


# =============================================================================
# Trading Metrics
# =============================================================================

# Orders
ORDERS_SUBMITTED = Counter(
    "orders_submitted_total",
    "Total orders submitted",
    ["symbol", "side", "order_type"],
    registry=REGISTRY,
)

ORDERS_FILLED = Counter(
    "orders_filled_total",
    "Total orders filled",
    ["symbol", "side"],
    registry=REGISTRY,
)

ORDERS_REJECTED = Counter(
    "orders_rejected_total",
    "Total orders rejected",
    ["symbol", "reason"],
    registry=REGISTRY,
)

ORDER_LATENCY = Histogram(
    "order_latency_seconds",
    "Order execution latency in seconds",
    ["symbol"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)

# Portfolio
PORTFOLIO_EQUITY = Gauge(
    "portfolio_equity",
    "Total portfolio equity",
    registry=REGISTRY,
)

PORTFOLIO_CASH = Gauge(
    "portfolio_cash",
    "Available cash balance",
    registry=REGISTRY,
)

PORTFOLIO_BUYING_POWER = Gauge(
    "portfolio_buying_power",
    "Available buying power",
    registry=REGISTRY,
)

PORTFOLIO_POSITIONS_COUNT = Gauge(
    "portfolio_positions_count",
    "Number of open positions",
    registry=REGISTRY,
)

PORTFOLIO_EXPOSURE = Gauge(
    "portfolio_exposure",
    "Portfolio exposure",
    ["type"],  # long, short, net, gross
    registry=REGISTRY,
)

# Performance
DAILY_PNL = Gauge(
    "daily_pnl",
    "Daily P&L",
    registry=REGISTRY,
)

TOTAL_PNL = Gauge(
    "total_pnl",
    "Total P&L",
    registry=REGISTRY,
)

SHARPE_RATIO_30D = Gauge(
    "sharpe_ratio_30d",
    "30-day rolling Sharpe ratio",
    registry=REGISTRY,
)

CURRENT_DRAWDOWN = Gauge(
    "current_drawdown",
    "Current drawdown percentage",
    registry=REGISTRY,
)

MAX_DRAWDOWN_30D = Gauge(
    "max_drawdown_30d",
    "30-day maximum drawdown",
    registry=REGISTRY,
)

WIN_RATE_30D = Gauge(
    "win_rate_30d",
    "30-day win rate",
    registry=REGISTRY,
)

# Execution
ORDER_FILL_RATE = Gauge(
    "order_fill_rate",
    "Order fill rate percentage",
    registry=REGISTRY,
)

AVG_SLIPPAGE_BPS = Gauge(
    "avg_slippage_bps",
    "Average slippage in basis points",
    registry=REGISTRY,
)

EXECUTION_LATENCY = Histogram(
    "execution_latency_ms",
    "Execution latency in milliseconds",
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
    registry=REGISTRY,
)

PARTIAL_FILL_RATE = Gauge(
    "partial_fill_rate",
    "Partial fill rate percentage",
    registry=REGISTRY,
)


# =============================================================================
# Risk Metrics
# =============================================================================

PORTFOLIO_VAR_95 = Gauge(
    "portfolio_var_95",
    "95% Value at Risk",
    registry=REGISTRY,
)

SECTOR_CONCENTRATION = Gauge(
    "sector_concentration",
    "Sector concentration",
    ["sector"],
    registry=REGISTRY,
)

LARGEST_POSITION_PCT = Gauge(
    "largest_position_pct",
    "Largest position as percentage of equity",
    registry=REGISTRY,
)

BETA_EXPOSURE = Gauge(
    "beta_exposure",
    "Portfolio beta exposure",
    registry=REGISTRY,
)


# =============================================================================
# Model Metrics
# =============================================================================

MODEL_PREDICTIONS = Counter(
    "model_predictions_total",
    "Total model predictions",
    ["model_name"],
    registry=REGISTRY,
)

MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Model accuracy",
    ["model_name"],
    registry=REGISTRY,
)

MODEL_AUC = Gauge(
    "model_auc",
    "Model AUC score",
    ["model_name"],
    registry=REGISTRY,
)

MODEL_SHARPE = Gauge(
    "model_sharpe",
    "Model Sharpe ratio",
    ["model_name"],
    registry=REGISTRY,
)

MODEL_ERROR_COUNT = Counter(
    "model_error_count",
    "Model error count",
    ["model_name", "error_type"],
    registry=REGISTRY,
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Model prediction latency in seconds",
    ["model_name"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=REGISTRY,
)

ENSEMBLE_AGREEMENT = Gauge(
    "ensemble_agreement",
    "Ensemble model agreement score",
    registry=REGISTRY,
)

TOP_SIGNAL_STRENGTH = Gauge(
    "top_signal_strength",
    "Top signal strength",
    registry=REGISTRY,
)

SIGNAL_DISPERSION = Gauge(
    "signal_dispersion",
    "Signal dispersion across models",
    registry=REGISTRY,
)


# =============================================================================
# Granular Latency Metrics
# =============================================================================

# Market data latency breakdown
MARKET_DATA_LATENCY = Histogram(
    "market_data_latency_ms",
    "Market data latency in milliseconds",
    ["operation"],  # websocket_receive, parse, validate, publish
    buckets=(0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500),
    registry=REGISTRY,
)

# Feature calculation latency by category
FEATURE_CALC_LATENCY = Histogram(
    "feature_calc_latency_ms",
    "Feature calculation latency in milliseconds",
    ["category"],  # trend, momentum, volatility, volume, composite
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500),
    registry=REGISTRY,
)

# Model inference latency breakdown
MODEL_INFERENCE_LATENCY = Histogram(
    "model_inference_latency_ms",
    "Model inference latency in milliseconds",
    ["model_type", "stage"],  # preprocess, forward_pass, postprocess
    buckets=(0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000),
    registry=REGISTRY,
)

# Order lifecycle latency stages
ORDER_LIFECYCLE_LATENCY = Histogram(
    "order_lifecycle_latency_ms",
    "Order lifecycle stage latency in milliseconds",
    ["stage"],  # signal_to_order, order_to_submit, submit_to_ack, ack_to_fill
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
    registry=REGISTRY,
)

# End-to-end signal latency
SIGNAL_E2E_LATENCY = Histogram(
    "signal_e2e_latency_ms",
    "End-to-end signal latency in milliseconds",
    buckets=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000),
    registry=REGISTRY,
)

# Database operation latency
DB_OPERATION_LATENCY = Histogram(
    "db_operation_latency_ms",
    "Database operation latency in milliseconds",
    ["operation", "table"],  # select, insert, update, delete
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
    registry=REGISTRY,
)

# Redis operation latency
REDIS_OPERATION_LATENCY = Histogram(
    "redis_operation_latency_ms",
    "Redis operation latency in milliseconds",
    ["operation"],  # get, set, delete, pipeline
    buckets=(0.1, 0.5, 1, 2, 5, 10, 25, 50, 100),
    registry=REGISTRY,
)


# =============================================================================
# Data Pipeline Metrics
# =============================================================================

DATA_PROCESSING = Histogram(
    "data_processing_seconds",
    "Data processing time in seconds",
    ["operation"],  # load, preprocess, feature_calc
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    registry=REGISTRY,
)

DATA_STALENESS = Gauge(
    "data_staleness_seconds",
    "Data staleness in seconds",
    ["data_type"],  # bars, quotes, features
    registry=REGISTRY,
)

CACHE_HITS = Counter(
    "cache_hits_total",
    "Cache hits",
    ["cache_name"],
    registry=REGISTRY,
)

CACHE_MISSES = Counter(
    "cache_misses_total",
    "Cache misses",
    ["cache_name"],
    registry=REGISTRY,
)


# =============================================================================
# Metrics Collector Class
# =============================================================================

class MetricsCollector:
    """Central metrics collector for the trading system.

    Provides convenient methods for updating metrics and generating
    Prometheus-compatible output.
    """

    _instance: "MetricsCollector | None" = None
    _start_time: float = 0.0

    def __new__(cls) -> "MetricsCollector":
        """Singleton pattern for metrics collector."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._start_time = time.time()
        return cls._instance

    def __init__(self) -> None:
        """Initialize the metrics collector."""
        pass  # All initialization done in __new__

    def set_system_info(
        self,
        version: str,
        environment: str,
        hostname: str | None = None,
    ) -> None:
        """Set system information.

        Args:
            version: Application version.
            environment: Environment name (development, staging, production).
            hostname: Optional hostname.
        """
        info = {
            "version": version,
            "environment": environment,
        }
        if hostname:
            info["hostname"] = hostname
        SYSTEM_INFO.info(info)

    def update_system_metrics(
        self,
        memory_bytes: int | None = None,
        cpu_percent: float | None = None,
        connections: dict[str, int] | None = None,
        queues: dict[str, int] | None = None,
    ) -> None:
        """Update system metrics.

        Args:
            memory_bytes: Memory usage in bytes.
            cpu_percent: CPU usage percentage.
            connections: Dict of connection type to count.
            queues: Dict of queue name to depth.
        """
        SYSTEM_UPTIME.set(time.time() - self._start_time)

        if memory_bytes is not None:
            MEMORY_USAGE.labels(type="rss").set(memory_bytes)

        if cpu_percent is not None:
            CPU_USAGE.set(cpu_percent)

        if connections:
            for conn_type, count in connections.items():
                OPEN_CONNECTIONS.labels(type=conn_type).set(count)

        if queues:
            for queue_name, depth in queues.items():
                QUEUE_DEPTH.labels(queue_name=queue_name).set(depth)

    def record_request(
        self,
        endpoint: str,
        method: str,
        status: int,
        latency: float,
    ) -> None:
        """Record an API request.

        Args:
            endpoint: Request endpoint.
            method: HTTP method.
            status: Response status code.
            latency: Request latency in seconds.
        """
        REQUESTS_TOTAL.labels(
            endpoint=endpoint,
            method=method,
            status=str(status),
        ).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)

    def record_error(self, error_type: str, component: str) -> None:
        """Record an error.

        Args:
            error_type: Type of error.
            component: Component where error occurred.
        """
        ERRORS_TOTAL.labels(error_type=error_type, component=component).inc()

    def record_order_submitted(
        self,
        symbol: str,
        side: str,
        order_type: str,
    ) -> None:
        """Record an order submission.

        Args:
            symbol: Trading symbol.
            side: Order side (BUY/SELL).
            order_type: Type of order.
        """
        ORDERS_SUBMITTED.labels(
            symbol=symbol,
            side=side,
            order_type=order_type,
        ).inc()

    def record_order_filled(self, symbol: str, side: str) -> None:
        """Record a filled order.

        Args:
            symbol: Trading symbol.
            side: Order side.
        """
        ORDERS_FILLED.labels(symbol=symbol, side=side).inc()

    def record_order_rejected(self, symbol: str, reason: str) -> None:
        """Record a rejected order.

        Args:
            symbol: Trading symbol.
            reason: Rejection reason.
        """
        ORDERS_REJECTED.labels(symbol=symbol, reason=reason).inc()

    def record_order_latency(self, symbol: str, latency: float) -> None:
        """Record order execution latency.

        Args:
            symbol: Trading symbol.
            latency: Latency in seconds.
        """
        ORDER_LATENCY.labels(symbol=symbol).observe(latency)

    def update_portfolio_metrics(
        self,
        equity: float,
        cash: float,
        buying_power: float,
        positions_count: int,
        long_exposure: float,
        short_exposure: float,
    ) -> None:
        """Update portfolio metrics.

        Args:
            equity: Total equity.
            cash: Cash balance.
            buying_power: Available buying power.
            positions_count: Number of positions.
            long_exposure: Long exposure value.
            short_exposure: Short exposure value.
        """
        PORTFOLIO_EQUITY.set(equity)
        PORTFOLIO_CASH.set(cash)
        PORTFOLIO_BUYING_POWER.set(buying_power)
        PORTFOLIO_POSITIONS_COUNT.set(positions_count)
        PORTFOLIO_EXPOSURE.labels(type="long").set(long_exposure)
        PORTFOLIO_EXPOSURE.labels(type="short").set(short_exposure)
        PORTFOLIO_EXPOSURE.labels(type="net").set(long_exposure - short_exposure)
        PORTFOLIO_EXPOSURE.labels(type="gross").set(long_exposure + short_exposure)

    def update_performance_metrics(
        self,
        daily_pnl: float,
        total_pnl: float,
        sharpe_ratio: float | None = None,
        current_drawdown: float | None = None,
        max_drawdown: float | None = None,
        win_rate: float | None = None,
    ) -> None:
        """Update performance metrics.

        Args:
            daily_pnl: Daily P&L.
            total_pnl: Total P&L.
            sharpe_ratio: 30-day Sharpe ratio.
            current_drawdown: Current drawdown.
            max_drawdown: 30-day max drawdown.
            win_rate: 30-day win rate.
        """
        DAILY_PNL.set(daily_pnl)
        TOTAL_PNL.set(total_pnl)

        if sharpe_ratio is not None:
            SHARPE_RATIO_30D.set(sharpe_ratio)
        if current_drawdown is not None:
            CURRENT_DRAWDOWN.set(current_drawdown)
        if max_drawdown is not None:
            MAX_DRAWDOWN_30D.set(max_drawdown)
        if win_rate is not None:
            WIN_RATE_30D.set(win_rate)

    def update_execution_metrics(
        self,
        fill_rate: float | None = None,
        avg_slippage_bps: float | None = None,
        partial_fill_rate: float | None = None,
    ) -> None:
        """Update execution quality metrics.

        Args:
            fill_rate: Order fill rate percentage.
            avg_slippage_bps: Average slippage in basis points.
            partial_fill_rate: Partial fill rate percentage.
        """
        if fill_rate is not None:
            ORDER_FILL_RATE.set(fill_rate)
        if avg_slippage_bps is not None:
            AVG_SLIPPAGE_BPS.set(avg_slippage_bps)
        if partial_fill_rate is not None:
            PARTIAL_FILL_RATE.set(partial_fill_rate)

    def record_execution_latency(self, latency_ms: float) -> None:
        """Record execution latency.

        Args:
            latency_ms: Latency in milliseconds.
        """
        EXECUTION_LATENCY.observe(latency_ms)

    def update_risk_metrics(
        self,
        var_95: float,
        sector_concentrations: dict[str, float] | None = None,
        largest_position_pct: float | None = None,
        beta: float | None = None,
    ) -> None:
        """Update risk metrics.

        Args:
            var_95: 95% Value at Risk.
            sector_concentrations: Dict of sector to concentration.
            largest_position_pct: Largest position as % of equity.
            beta: Portfolio beta.
        """
        PORTFOLIO_VAR_95.set(var_95)

        if sector_concentrations:
            for sector, concentration in sector_concentrations.items():
                SECTOR_CONCENTRATION.labels(sector=sector).set(concentration)

        if largest_position_pct is not None:
            LARGEST_POSITION_PCT.set(largest_position_pct)

        if beta is not None:
            BETA_EXPOSURE.set(beta)

    def record_model_prediction(
        self,
        model_name: str,
        latency: float | None = None,
    ) -> None:
        """Record a model prediction.

        Args:
            model_name: Name of the model.
            latency: Prediction latency in seconds.
        """
        MODEL_PREDICTIONS.labels(model_name=model_name).inc()
        if latency is not None:
            PREDICTION_LATENCY.labels(model_name=model_name).observe(latency)

    def update_model_metrics(
        self,
        model_name: str,
        accuracy: float | None = None,
        auc: float | None = None,
        sharpe: float | None = None,
    ) -> None:
        """Update model performance metrics.

        Args:
            model_name: Name of the model.
            accuracy: Model accuracy.
            auc: Model AUC score.
            sharpe: Model Sharpe ratio.
        """
        if accuracy is not None:
            MODEL_ACCURACY.labels(model_name=model_name).set(accuracy)
        if auc is not None:
            MODEL_AUC.labels(model_name=model_name).set(auc)
        if sharpe is not None:
            MODEL_SHARPE.labels(model_name=model_name).set(sharpe)

    def record_model_error(self, model_name: str, error_type: str) -> None:
        """Record a model error.

        Args:
            model_name: Name of the model.
            error_type: Type of error.
        """
        MODEL_ERROR_COUNT.labels(
            model_name=model_name,
            error_type=error_type,
        ).inc()

    def update_ensemble_metrics(
        self,
        agreement: float,
        top_signal_strength: float,
        dispersion: float,
    ) -> None:
        """Update ensemble model metrics.

        Args:
            agreement: Ensemble agreement score.
            top_signal_strength: Strength of top signal.
            dispersion: Signal dispersion across models.
        """
        ENSEMBLE_AGREEMENT.set(agreement)
        TOP_SIGNAL_STRENGTH.set(top_signal_strength)
        SIGNAL_DISPERSION.set(dispersion)

    @contextmanager
    def time_data_processing(
        self,
        operation: str,
    ) -> Generator[None, None, None]:
        """Context manager to time data processing operations.

        Args:
            operation: Name of the operation being timed.

        Yields:
            None
        """
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            DATA_PROCESSING.labels(operation=operation).observe(duration)

    def update_data_staleness(self, data_type: str, staleness_seconds: float) -> None:
        """Update data staleness metric.

        Args:
            data_type: Type of data.
            staleness_seconds: Staleness in seconds.
        """
        DATA_STALENESS.labels(data_type=data_type).set(staleness_seconds)

    def record_cache_hit(self, cache_name: str) -> None:
        """Record a cache hit.

        Args:
            cache_name: Name of the cache.
        """
        CACHE_HITS.labels(cache_name=cache_name).inc()

    def record_cache_miss(self, cache_name: str) -> None:
        """Record a cache miss.

        Args:
            cache_name: Name of the cache.
        """
        CACHE_MISSES.labels(cache_name=cache_name).inc()

    def get_metrics(self) -> bytes:
        """Get Prometheus metrics output.

        Returns:
            Prometheus metrics in text format.
        """
        return generate_latest(REGISTRY)

    def get_content_type(self) -> str:
        """Get the content type for metrics output.

        Returns:
            Content type string.
        """
        return CONTENT_TYPE_LATEST

    # =========================================================================
    # Granular Latency Recording Methods
    # =========================================================================

    def record_market_data_latency(self, operation: str, latency_ms: float) -> None:
        """Record market data processing latency.

        Args:
            operation: Operation type (websocket_receive, parse, validate, publish).
            latency_ms: Latency in milliseconds.
        """
        MARKET_DATA_LATENCY.labels(operation=operation).observe(latency_ms)

    def record_feature_calc_latency(self, category: str, latency_ms: float) -> None:
        """Record feature calculation latency by category.

        Args:
            category: Feature category (trend, momentum, volatility, volume, composite).
            latency_ms: Latency in milliseconds.
        """
        FEATURE_CALC_LATENCY.labels(category=category).observe(latency_ms)

    def record_model_inference_latency(
        self,
        model_type: str,
        stage: str,
        latency_ms: float,
    ) -> None:
        """Record model inference latency by stage.

        Args:
            model_type: Type of model (lstm, transformer, xgboost, etc.).
            stage: Processing stage (preprocess, forward_pass, postprocess).
            latency_ms: Latency in milliseconds.
        """
        MODEL_INFERENCE_LATENCY.labels(
            model_type=model_type,
            stage=stage,
        ).observe(latency_ms)

    def record_order_lifecycle_latency(self, stage: str, latency_ms: float) -> None:
        """Record order lifecycle stage latency.

        Args:
            stage: Lifecycle stage (signal_to_order, order_to_submit,
                   submit_to_ack, ack_to_fill).
            latency_ms: Latency in milliseconds.
        """
        ORDER_LIFECYCLE_LATENCY.labels(stage=stage).observe(latency_ms)

    def record_signal_e2e_latency(self, latency_ms: float) -> None:
        """Record end-to-end signal latency.

        Args:
            latency_ms: Total latency from bar receipt to order submission in ms.
        """
        SIGNAL_E2E_LATENCY.observe(latency_ms)

    def record_db_latency(
        self,
        operation: str,
        table: str,
        latency_ms: float,
    ) -> None:
        """Record database operation latency.

        Args:
            operation: DB operation (select, insert, update, delete).
            table: Table name.
            latency_ms: Latency in milliseconds.
        """
        DB_OPERATION_LATENCY.labels(operation=operation, table=table).observe(latency_ms)

    def record_redis_latency(self, operation: str, latency_ms: float) -> None:
        """Record Redis operation latency.

        Args:
            operation: Redis operation (get, set, delete, pipeline).
            latency_ms: Latency in milliseconds.
        """
        REDIS_OPERATION_LATENCY.labels(operation=operation).observe(latency_ms)

    @contextmanager
    def time_market_data(
        self,
        operation: str,
    ) -> Generator[None, None, None]:
        """Context manager to time market data operations.

        Args:
            operation: Operation name.

        Yields:
            None
        """
        start = time.time()
        try:
            yield
        finally:
            latency_ms = (time.time() - start) * 1000
            self.record_market_data_latency(operation, latency_ms)

    @contextmanager
    def time_feature_calc(
        self,
        category: str,
    ) -> Generator[None, None, None]:
        """Context manager to time feature calculations.

        Args:
            category: Feature category.

        Yields:
            None
        """
        start = time.time()
        try:
            yield
        finally:
            latency_ms = (time.time() - start) * 1000
            self.record_feature_calc_latency(category, latency_ms)

    @contextmanager
    def time_model_inference(
        self,
        model_type: str,
        stage: str,
    ) -> Generator[None, None, None]:
        """Context manager to time model inference stages.

        Args:
            model_type: Type of model.
            stage: Processing stage.

        Yields:
            None
        """
        start = time.time()
        try:
            yield
        finally:
            latency_ms = (time.time() - start) * 1000
            self.record_model_inference_latency(model_type, stage, latency_ms)


def get_metrics_collector() -> MetricsCollector:
    """Get the singleton metrics collector instance.

    Returns:
        MetricsCollector instance.
    """
    return MetricsCollector()
