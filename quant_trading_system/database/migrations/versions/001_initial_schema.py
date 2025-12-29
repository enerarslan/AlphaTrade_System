"""Initial schema creation

Revision ID: 001_initial
Revises: None
Create Date: 2024-01-15

Creates all initial tables for the trading system:
- Market data (ohlcv_bars, features)
- Trading (orders, trades, positions, position_history)
- Signals and predictions
- Performance tracking
- System logs and alerts
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # OHLCV Bars table
    op.create_table(
        "ohlcv_bars",
        sa.Column("symbol", sa.String(10), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(12, 4), nullable=False),
        sa.Column("high", sa.Numeric(12, 4), nullable=False),
        sa.Column("low", sa.Numeric(12, 4), nullable=False),
        sa.Column("close", sa.Numeric(12, 4), nullable=False),
        sa.Column("volume", sa.BigInteger(), nullable=False),
        sa.Column("vwap", sa.Numeric(12, 4), nullable=True),
        sa.Column("trade_count", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("symbol", "timestamp"),
    )
    op.create_index("ix_ohlcv_bars_timestamp", "ohlcv_bars", ["timestamp"])
    op.create_index("ix_ohlcv_bars_symbol", "ohlcv_bars", ["symbol"])

    # Features table
    op.create_table(
        "features",
        sa.Column("symbol", sa.String(10), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature_name", sa.String(100), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("symbol", "timestamp", "feature_name"),
    )
    op.create_index("ix_features_symbol_timestamp", "features", ["symbol", "timestamp"])
    op.create_index("ix_features_feature_name", "features", ["feature_name"])

    # Orders table
    op.create_table(
        "orders",
        sa.Column("order_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("client_order_id", sa.String(50), nullable=False),
        sa.Column("broker_order_id", sa.String(50), nullable=True),
        sa.Column("symbol", sa.String(10), nullable=False),
        sa.Column("side", sa.String(4), nullable=False),
        sa.Column("order_type", sa.String(20), nullable=False),
        sa.Column("quantity", sa.Numeric(12, 4), nullable=False),
        sa.Column("limit_price", sa.Numeric(12, 4), nullable=True),
        sa.Column("stop_price", sa.Numeric(12, 4), nullable=True),
        sa.Column("time_in_force", sa.String(10), nullable=False, server_default="DAY"),
        sa.Column("status", sa.String(20), nullable=False, server_default="PENDING"),
        sa.Column("filled_qty", sa.Numeric(12, 4), server_default="0"),
        sa.Column("filled_avg_price", sa.Numeric(12, 4), nullable=True),
        sa.Column("filled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("strategy_name", sa.String(50), nullable=True),
        sa.Column("signal_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("order_id"),
        sa.UniqueConstraint("client_order_id"),
    )
    op.create_index("ix_orders_symbol", "orders", ["symbol"])
    op.create_index("ix_orders_status", "orders", ["status"])
    op.create_index("ix_orders_symbol_created", "orders", ["symbol", "created_at"])
    op.create_index("ix_orders_strategy_created", "orders", ["strategy_name", "created_at"])

    # Trades table
    op.create_table(
        "trades",
        sa.Column("trade_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("order_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("symbol", sa.String(10), nullable=False),
        sa.Column("side", sa.String(4), nullable=False),
        sa.Column("quantity", sa.Numeric(12, 4), nullable=False),
        sa.Column("price", sa.Numeric(12, 4), nullable=False),
        sa.Column("commission", sa.Numeric(10, 4), server_default="0"),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("trade_id"),
        sa.ForeignKeyConstraint(["order_id"], ["orders.order_id"]),
    )
    op.create_index("ix_trades_symbol", "trades", ["symbol"])
    op.create_index("ix_trades_executed_at", "trades", ["executed_at"])
    op.create_index("ix_trades_symbol_executed", "trades", ["symbol", "executed_at"])

    # Positions table
    op.create_table(
        "positions",
        sa.Column("symbol", sa.String(10), nullable=False),
        sa.Column("quantity", sa.Numeric(12, 4), nullable=False),
        sa.Column("avg_entry_price", sa.Numeric(12, 4), nullable=False),
        sa.Column("cost_basis", sa.Numeric(14, 4), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("symbol"),
    )

    # Position History table
    op.create_table(
        "position_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(10), nullable=False),
        sa.Column("quantity", sa.Numeric(12, 4), nullable=False),
        sa.Column("avg_entry_price", sa.Numeric(12, 4), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_position_history_symbol", "position_history", ["symbol"])
    op.create_index("ix_position_history_timestamp", "position_history", ["timestamp"])

    # Signals table
    op.create_table(
        "signals",
        sa.Column("signal_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(10), nullable=False),
        sa.Column("direction", sa.String(10), nullable=False),
        sa.Column("strength", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("horizon", sa.Integer(), nullable=False),
        sa.Column("model_source", sa.String(50), nullable=False),
        sa.Column("features_snapshot", postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("signal_id"),
    )
    op.create_index("ix_signals_timestamp", "signals", ["timestamp"])
    op.create_index("ix_signals_symbol", "signals", ["symbol"])
    op.create_index("ix_signals_model_source", "signals", ["model_source"])
    op.create_index("ix_signals_symbol_timestamp", "signals", ["symbol", "timestamp"])

    # Model Predictions table
    op.create_table(
        "model_predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_name", sa.String(50), nullable=False),
        sa.Column("model_version", sa.String(20), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(10), nullable=False),
        sa.Column("prediction", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("actual", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_predictions_model_name", "model_predictions", ["model_name"])
    op.create_index("ix_model_predictions_timestamp", "model_predictions", ["timestamp"])
    op.create_index("ix_model_predictions_symbol", "model_predictions", ["symbol"])

    # Daily Performance table
    op.create_table(
        "daily_performance",
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("starting_equity", sa.Numeric(14, 4), nullable=False),
        sa.Column("ending_equity", sa.Numeric(14, 4), nullable=False),
        sa.Column("pnl", sa.Numeric(14, 4), nullable=False),
        sa.Column("pnl_percent", sa.Float(), nullable=False),
        sa.Column("trades_count", sa.Integer(), server_default="0"),
        sa.Column("win_count", sa.Integer(), server_default="0"),
        sa.Column("loss_count", sa.Integer(), server_default="0"),
        sa.Column("max_drawdown", sa.Float(), server_default="0"),
        sa.Column("sharpe_estimate", sa.Float(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("date"),
    )

    # Trade Log table
    op.create_table(
        "trade_log",
        sa.Column("trade_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("symbol", sa.String(10), nullable=False),
        sa.Column("entry_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("exit_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("side", sa.String(4), nullable=False),
        sa.Column("entry_quantity", sa.Numeric(12, 4), nullable=False),
        sa.Column("exit_quantity", sa.Numeric(12, 4), nullable=True),
        sa.Column("entry_price", sa.Numeric(12, 4), nullable=False),
        sa.Column("exit_price", sa.Numeric(12, 4), nullable=True),
        sa.Column("pnl", sa.Numeric(14, 4), nullable=True),
        sa.Column("pnl_percent", sa.Float(), nullable=True),
        sa.Column("commission_total", sa.Numeric(10, 4), server_default="0"),
        sa.Column("slippage", sa.Float(), nullable=True),
        sa.Column("signals", postgresql.JSONB(), nullable=True),
        sa.Column("strategy", sa.String(50), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("trade_id"),
    )
    op.create_index("ix_trade_log_symbol", "trade_log", ["symbol"])
    op.create_index("ix_trade_log_strategy", "trade_log", ["strategy"])

    # System Logs table
    op.create_table(
        "system_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("level", sa.String(10), nullable=False),
        sa.Column("category", sa.String(20), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("correlation_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("context", postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_system_logs_timestamp", "system_logs", ["timestamp"])
    op.create_index("ix_system_logs_level", "system_logs", ["level"])
    op.create_index("ix_system_logs_category", "system_logs", ["category"])

    # Alerts table
    op.create_table(
        "alerts",
        sa.Column("alert_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("severity", sa.String(10), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("context", postgresql.JSONB(), nullable=True),
        sa.Column("acknowledged", sa.Boolean(), server_default="false"),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("acknowledged_by", sa.String(50), nullable=True),
        sa.Column("resolved", sa.Boolean(), server_default="false"),
        sa.PrimaryKeyConstraint("alert_id"),
    )
    op.create_index("ix_alerts_timestamp", "alerts", ["timestamp"])
    op.create_index("ix_alerts_severity", "alerts", ["severity"])


def downgrade() -> None:
    op.drop_table("alerts")
    op.drop_table("system_logs")
    op.drop_table("trade_log")
    op.drop_table("daily_performance")
    op.drop_table("model_predictions")
    op.drop_table("signals")
    op.drop_table("position_history")
    op.drop_table("positions")
    op.drop_table("trades")
    op.drop_table("orders")
    op.drop_table("features")
    op.drop_table("ohlcv_bars")
