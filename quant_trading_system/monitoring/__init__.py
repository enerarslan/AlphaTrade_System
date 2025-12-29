"""
Monitoring and alerting module.

Provides Prometheus metrics, structured logging, alert management,
and dashboard endpoints.
"""

from .alerting import (
    Alert,
    AlertChannel,
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    AlertType,
    get_alert_manager,
    alert_critical,
    alert_info,
    alert_warning,
)
from .dashboard import (
    app as dashboard_app,
    broadcast_alert,
    broadcast_order_update,
    broadcast_portfolio_update,
    broadcast_signal,
    create_dashboard_app,
    get_connection_manager,
    get_dashboard_app,
    get_dashboard_state,
)
from .logger import (
    ContextLogger,
    JsonFormatter,
    LogCategory,
    LogFormat,
    StructuredLogRecord,
    TextFormatter,
    TradeLogEntry,
    TradeLogger,
    get_logger,
    get_trade_logger,
    log_alert,
    log_data,
    log_model,
    log_order,
    log_risk,
    log_system,
    log_trading,
    setup_logging,
)
from .metrics import (
    MetricsCollector,
    REGISTRY,
    get_metrics_collector,
)


__all__ = [
    # Alerting
    "Alert",
    "AlertChannel",
    "AlertManager",
    "AlertRule",
    "AlertSeverity",
    "AlertStatus",
    "AlertType",
    "get_alert_manager",
    "alert_critical",
    "alert_info",
    "alert_warning",
    # Dashboard
    "dashboard_app",
    "broadcast_alert",
    "broadcast_order_update",
    "broadcast_portfolio_update",
    "broadcast_signal",
    "create_dashboard_app",
    "get_connection_manager",
    "get_dashboard_app",
    "get_dashboard_state",
    # Logger
    "ContextLogger",
    "JsonFormatter",
    "LogCategory",
    "LogFormat",
    "StructuredLogRecord",
    "TextFormatter",
    "TradeLogEntry",
    "TradeLogger",
    "get_logger",
    "get_trade_logger",
    "log_alert",
    "log_data",
    "log_model",
    "log_order",
    "log_risk",
    "log_system",
    "log_trading",
    "setup_logging",
    # Metrics
    "MetricsCollector",
    "REGISTRY",
    "get_metrics_collector",
]
