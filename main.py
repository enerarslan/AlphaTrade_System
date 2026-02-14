#!/usr/bin/env python3
"""
================================================================================
ALPHATRADE SYSTEM - INSTITUTIONAL-GRADE QUANTITATIVE TRADING PLATFORM
================================================================================

JPMorgan-Level Unified Command Line Interface

This is the single entry point for all AlphaTrade operations:
  - trade     : Live/Paper trading with full risk management
  - backtest  : Historical strategy backtesting with realistic simulation
  - train     : ML model training with purged cross-validation
  - data      : Data loading, preprocessing, and management
  - features  : Feature engineering and pipeline management
  - health    : System diagnostics and monitoring
  - deploy    : Deployment, setup, and configuration

ARCHITECTURE:
  - Event-driven with pub/sub pattern (core/events.py)
  - Singleton services (Settings, EventBus, MetricsCollector)
  - Abstract base classes for extensibility
  - GPU acceleration support (RAPIDS cuDF, PyTorch CUDA)
  - Multi-region deployment ready

SAFETY FEATURES:
  - Kill Switch with 30-minute cooldown
  - Pre-trade risk checks (position limits, sector exposure)
  - Circuit breaker for external APIs
  - Credential masking in logs
  - Audit trail for regulatory compliance

USAGE:
  python main.py trade --mode paper --symbols AAPL MSFT GOOGL
  python main.py backtest --start 2024-01-01 --end 2024-06-30
  python main.py train --model xgboost --symbols AAPL MSFT
  python main.py data load --source alpaca --symbols AAPL
  python main.py features compute --symbols AAPL --gpu
  python main.py health check --full
  python main.py deploy setup --env production

Author: AlphaTrade System
Version: 1.3.0
License: Proprietary
================================================================================
"""

from __future__ import annotations

# Load environment variables FIRST, before any other imports
# This ensures DATABASE_URL and other settings are correct
from dotenv import load_dotenv
load_dotenv(override=True)

import argparse
import asyncio
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

# ==============================================================================
# PATH SETUP
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ==============================================================================
# LAZY IMPORTS - Fast startup, load modules only when needed
# ==============================================================================

def get_settings():
    """Lazy import settings."""
    from quant_trading_system.config.settings import get_settings as _get_settings
    return _get_settings()


def get_logger(name: str, category: str = "SYSTEM"):
    """Lazy import logger."""
    from quant_trading_system.monitoring.logger import get_logger as _get_logger, LogCategory
    cat = getattr(LogCategory, category.upper(), LogCategory.SYSTEM)
    return _get_logger(name, cat)


def setup_logging(log_level: str = "INFO", log_format: str = "CONSOLE"):
    """Lazy import and setup logging."""
    from quant_trading_system.monitoring.logger import (
        setup_logging as _setup_logging,
        LogFormat,
    )
    fmt = getattr(LogFormat, log_format.upper(), LogFormat.TEXT)
    _setup_logging(level=log_level, log_format=fmt)


# ==============================================================================
# VERSION AND BANNER
# ==============================================================================

VERSION = "1.3.0"
BUILD_DATE = "2026-01-04"

BANNER = f"""
================================================================================
    _    _       _         _____              _
   / \\  | |_ __ | |__   __|_   _| __ __ _  __| | ___
  / _ \\ | | '_ \\| '_ \\ / _` || || '__/ _` |/ _` |/ _ \\
 / ___ \\| | |_) | | | | (_| || || | | (_| | (_| |  __/
/_/   \\_\\_| .__/|_| |_|\\__,_||_||_|  \\__,_|\\__,_|\\___|
          |_|

  Institutional-Grade Quantitative Trading System
  Version: {VERSION} | Build: {BUILD_DATE}
================================================================================
"""


# ==============================================================================
# COMMAND HANDLERS
# ==============================================================================

def cmd_trade(args: argparse.Namespace) -> int:
    """
    Execute live or paper trading.

    @trader CRITICAL: This command handles real money operations.
    All orders pass through PreTradeRiskChecker and KillSwitch.

    Modes:
      - live   : Real money trading (requires explicit confirmation)
      - paper  : Simulated trading with real market data
      - dry-run: Signal generation only, no order submission

    Features:
      - VIX-based regime detection for adaptive strategies
      - Real-time position and P&L tracking
      - Intraday drawdown monitoring with alerts
      - Automatic kill switch on limit breach
      - Transaction cost analysis (TCA)
      - A/B testing for strategy variants
    """
    if getattr(args, "dry_run", False):
        args.mode = "dry-run"

    from scripts.trade import run_trading
    return run_trading(args)


def cmd_backtest(args: argparse.Namespace) -> int:
    """
    Run historical backtesting.

    @mlquant + @trader: Combines model predictions with realistic execution.

    Execution Modes:
      - realistic   : Includes slippage, spread, partial fills, market impact
      - optimistic  : Best-case execution (for upper bound)
      - pessimistic : Worst-case execution (for stress testing)

    Features:
      - Event-driven or vectorized engine
      - Walk-forward optimization
      - Out-of-sample validation
      - Regime-specific analysis
      - Performance attribution (Brinson-Fachler)
      - Monte Carlo simulation for confidence intervals
    """
    # Backward-compatible aliases used by older scripts/tests.
    if hasattr(args, "start_date"):
        args.start = args.start_date
    if hasattr(args, "end_date"):
        args.end = args.end_date
    if hasattr(args, "initial_capital"):
        args.capital = args.initial_capital

    from scripts.backtest import run_backtest
    return run_backtest(args)


def cmd_train(args: argparse.Namespace) -> int:
    """
    Train ML models.

    @mlquant: Machine learning pipeline with institutional-grade validation.

    Model Types:
      - xgboost    : Gradient boosting (fast, interpretable)
      - lightgbm   : Light gradient boosting (efficient)
      - random_forest: Bagged tree ensemble baseline
      - lstm       : Long short-term memory (sequential patterns)
      - transformer: Attention-based (complex dependencies)
      - tcn        : Temporal convolutional network
      - ensemble   : IC-weighted model combination
      - all        : Train full model suite

    Validation:
      - Purged K-Fold with embargo (no look-ahead bias)
      - Combinatorial purged CV for robust estimates
      - Walk-forward validation for production simulation
      - Model validation gates (min Sharpe, max drawdown, etc.)

    Features:
      - Meta-labeling for signal filtering
      - Multiple testing correction (Bonferroni, BH, Deflated Sharpe)
      - Feature importance and SHAP explanations
      - Model staleness detection
      - Mandatory Optuna hyperparameter optimization
      - Mandatory PostgreSQL + Redis + leak-validation institutional stack
    """
    from scripts.train import run_training
    return run_training(args)


def cmd_data(args: argparse.Namespace) -> int:
    """
    Data management operations.

    @data: Data loading, preprocessing, and quality control.

    Operations:
      - load           : Load historical data from sources
      - validate       : Validate data quality (gaps, outliers, OHLCV rules)
      - preprocess     : Clean and normalize data
      - export         : Export data to various formats
      - migrate        : Migrate CSV data to PostgreSQL + TimescaleDB
      - export-training: Export PostgreSQL data to Parquet for ML training
      - db-status      : Check database connection and status

    Sources:
      - alpaca    : Alpaca Markets API
      - csv       : Local CSV files
      - database  : PostgreSQL/TimescaleDB

    Features:
      - PostgreSQL + TimescaleDB for live operations
      - Parquet export for ML training (GPU-accelerated)
      - Intrinsic time bars (volume, dollar, imbalance, run bars)
      - Alternative data integration (news, social, satellite)
      - Data lineage tracking for regulatory compliance
      - Automatic gap filling and outlier handling
    """
    from scripts.data import run_data_command
    return run_data_command(args)


def cmd_features(args: argparse.Namespace) -> int:
    """
    Feature engineering pipeline.

    @mlquant: Compute and manage trading features.

    Feature Categories:
      - technical    : 200+ technical indicators
      - statistical  : Returns, volatility, VaR, Hurst exponent
      - microstructure: VPIN, Kyle's Lambda, order flow toxicity
      - cross-sectional: Relative strength, sector rankings
      - alternative  : News sentiment, social signals

    Modes:
      - compute : Calculate features for symbols
      - cache   : Pre-compute and cache features
      - validate: Check for look-ahead bias
      - export  : Export feature matrix

    GPU Acceleration:
      - RAPIDS cuDF for GPU-accelerated computation
      - Numba JIT for CPU fallback
      - Automatic device selection
    """
    from scripts.features import run_features_command
    return run_features_command(args)


def cmd_health(args: argparse.Namespace) -> int:
    """
    System health and diagnostics.

    @infra: Monitor and diagnose system health.

    Checks:
      - connectivity: API keys, broker connection, database
      - models      : Model staleness, validation status
      - data        : Data freshness, gaps, quality
      - risk        : Kill switch status, limit utilization
      - performance : Latency, throughput, resource usage

    Features:
      - Prometheus metrics exposure
      - Grafana dashboard generation
      - Alert configuration validation
      - Audit log verification
    """
    from scripts.health import run_health_command
    return run_health_command(args)


def cmd_deploy(args: argparse.Namespace) -> int:
    """
    Deployment and setup operations.

    @infra: Infrastructure management and deployment.

    Operations:
      - setup   : Initialize system configuration
      - migrate : Run database migrations
      - docker  : Docker container management
      - k8s     : Kubernetes deployment (multi-region)

    Environments:
      - development : Local development setup
      - staging     : Pre-production testing
      - production  : Live trading environment

    Features:
      - Regional configuration (US_EAST, US_WEST, EU_WEST, ASIA_PACIFIC)
      - Secrets management (environment variables)
      - Health check endpoints
      - Graceful shutdown handling
    """
    from scripts.deploy import run_deployment
    return run_deployment(args)


def cmd_dashboard(args: argparse.Namespace) -> int:
    """
    Run the monitoring dashboard.
    
    @infra: Starts the FastAPI dashboard and WebSocket server.
    """
    import uvicorn
    from quant_trading_system.config.settings import get_settings
    
    settings = get_settings()
    
    # Allow command line override of port/host
    host = args.host or "0.0.0.0"
    port = args.port or 8000
    
    # Configure logging
    setup_logging(log_level=args.log_level)
    get_logger("dashboard").info(f"Starting dashboard on http://{host}:{port}")
    
    uvicorn.run(
        "quant_trading_system.monitoring.dashboard:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level=args.log_level.lower(),
    )
    return 0


# ==============================================================================
# CLI ARGUMENT PARSER
# ==============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""

    parser = argparse.ArgumentParser(
        prog="alphatrade",
        description="AlphaTrade - Institutional-Grade Quantitative Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Paper trading with default symbols
  python main.py trade --mode paper

  # Backtest with custom date range
  python main.py backtest --start 2024-01-01 --end 2024-06-30 --symbols AAPL MSFT

  # Train XGBoost model with GPU
  python main.py train --model xgboost --gpu

  # Load historical data from Alpaca
  python main.py data load --source alpaca --symbols AAPL MSFT GOOGL

  # Compute features with caching
  python main.py features compute --symbols AAPL --cache

  # Run full system health check
  python main.py health check --full

  # Run full system health check
  python main.py health check --full

  # Start dashboard
  python main.py dashboard --port 8000

  # Deploy to production
  python main.py deploy setup --env production

For more information, visit: https://github.com/alphatrade/docs
        """,
    )

    # Global arguments
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"AlphaTrade {VERSION}",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-format",
        type=str.lower,
        choices=["console", "json", "structured", "text"],
        default="console",
        help="Log output format (default: CONSOLE)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress banner and non-essential output",
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        title="Commands",
        dest="command",
        metavar="<command>",
    )

    # --------------------------------------------------------------------------
    # TRADE COMMAND
    # --------------------------------------------------------------------------
    trade_parser = subparsers.add_parser(
        "trade",
        help="Execute live or paper trading",
        description="Run the trading engine in live, paper, or dry-run mode.",
    )
    trade_parser.add_argument(
        "--mode", "-m",
        choices=["live", "paper", "dry-run"],
        default="paper",
        help="Trading mode (default: paper)",
    )
    trade_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Alias for --mode dry-run",
    )
    trade_parser.add_argument(
        "--symbols", "-s",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        help="Symbols to trade",
    )
    trade_parser.add_argument(
        "--strategy",
        default="momentum",
        help="Trading strategy to use (default: momentum)",
    )
    trade_parser.add_argument(
        "--capital",
        type=float,
        default=100000.0,
        help="Initial capital (default: 100000)",
    )
    trade_parser.add_argument(
        "--max-positions",
        type=int,
        default=10,
        help="Maximum concurrent positions (default: 10)",
    )
    trade_parser.add_argument(
        "--kill-switch-drawdown",
        type=float,
        default=0.10,
        help="Kill switch drawdown threshold (default: 0.10)",
    )
    trade_parser.add_argument(
        "--vix-enabled",
        action="store_true",
        default=True,
        help="Enable VIX-based regime detection",
    )
    trade_parser.add_argument(
        "--ab-test",
        type=str,
        help="A/B test experiment name",
    )
    trade_parser.add_argument(
        "--duration",
        type=int,
        help="Trading duration in minutes (default: until market close)",
    )
    trade_parser.set_defaults(func=cmd_trade)

    # --------------------------------------------------------------------------
    # BACKTEST COMMAND
    # --------------------------------------------------------------------------
    backtest_parser = subparsers.add_parser(
        "backtest",
        help="Run historical backtesting",
        description="Backtest trading strategies on historical data.",
    )
    backtest_parser.add_argument(
        "--start", "--start-date", "-s",
        dest="start_date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    backtest_parser.add_argument(
        "--end", "--end-date", "-e",
        dest="end_date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    backtest_parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        help="Symbols to backtest",
    )
    backtest_parser.add_argument(
        "--capital", "--initial-capital",
        dest="initial_capital",
        type=float,
        default=100000.0,
        help="Initial capital (default: 100000)",
    )
    backtest_parser.add_argument(
        "--strategy",
        default="momentum",
        help="Strategy to backtest (default: momentum)",
    )
    backtest_parser.add_argument(
        "--execution-mode",
        choices=["realistic", "optimistic", "pessimistic"],
        default="realistic",
        help="Execution simulation mode (default: realistic)",
    )
    backtest_parser.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
        help="Slippage in basis points (default: 5)",
    )
    backtest_parser.add_argument(
        "--commission-bps",
        type=float,
        default=1.0,
        help="Commission in basis points (default: 1)",
    )
    backtest_parser.add_argument(
        "--monte-carlo",
        type=int,
        default=0,
        help="Number of Monte Carlo simulations (default: 0 = disabled)",
    )
    backtest_parser.add_argument(
        "--output",
        type=Path,
        help="Output file for results (JSON/HTML)",
    )
    backtest_parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration for feature computation",
    )
    backtest_parser.add_argument(
        "--use-database",
        action="store_true",
        default=True,
        help="Load data from PostgreSQL + TimescaleDB (default: True)",
    )
    backtest_parser.add_argument(
        "--no-database",
        action="store_true",
        help="Load data from CSV files instead of database",
    )
    backtest_parser.set_defaults(func=cmd_backtest)

    # --------------------------------------------------------------------------
    # TRAIN COMMAND
    # --------------------------------------------------------------------------
    train_parser = subparsers.add_parser(
        "train",
        help="Train ML models",
        description="Train and validate machine learning models.",
    )
    train_parser.add_argument(
        "--model", "-m",
        choices=[
            "xgboost",
            "lightgbm",
            "random_forest",
            "elastic_net",
            "lstm",
            "transformer",
            "tcn",
            "ensemble",
            "all",
        ],
        default="xgboost",
        help="Model type to train (default: xgboost)",
    )
    train_parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Optional model name/version suffix",
    )
    train_parser.add_argument(
        "--symbols",
        nargs="+",
        default=[],
        help="Symbols to train on (default: all available symbols)",
    )
    train_parser.add_argument(
        "--start",
        type=str,
        help="Training start date (YYYY-MM-DD)",
    )
    train_parser.add_argument(
        "--end",
        type=str,
        help="Training end date (YYYY-MM-DD)",
    )
    train_parser.add_argument(
        "--cv-method",
        choices=["purged_kfold", "combinatorial", "walk_forward"],
        default="purged_kfold",
        help="Cross-validation method (default: purged_kfold)",
    )
    train_parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV splits (default: 5)",
    )
    train_parser.add_argument(
        "--embargo-pct",
        type=float,
        default=0.01,
        help="Embargo percentage for purged CV (default: 0.01)",
    )
    train_parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Optuna hyperparameter optimization trials (mandatory, default: 100)",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global deterministic seed for reproducible training (default: 42)",
    )
    train_parser.add_argument(
        "--nested-outer-splits",
        type=int,
        default=4,
        help="Outer splits for mandatory nested walk-forward optimization (default: 4)",
    )
    train_parser.add_argument(
        "--nested-inner-splits",
        type=int,
        default=3,
        help="Inner splits for mandatory nested walk-forward optimization (default: 3)",
    )
    train_parser.add_argument(
        "--disable-nested-walk-forward",
        action="store_true",
        help="Forbidden in institutional mode; retained for explicit fail-fast validation",
    )
    train_parser.add_argument(
        "--objective-weight-sharpe",
        type=float,
        default=1.0,
        help="Multi-objective Sharpe weight (default: 1.0)",
    )
    train_parser.add_argument(
        "--objective-weight-drawdown",
        type=float,
        default=0.5,
        help="Multi-objective drawdown penalty weight (default: 0.5)",
    )
    train_parser.add_argument(
        "--objective-weight-turnover",
        type=float,
        default=0.1,
        help="Multi-objective turnover penalty weight (default: 0.1)",
    )
    train_parser.add_argument(
        "--objective-weight-calibration",
        type=float,
        default=0.25,
        help="Multi-objective calibration penalty weight (default: 0.25)",
    )
    train_parser.add_argument(
        "--objective-weight-trade-activity",
        type=float,
        default=1.0,
        help="Penalty weight for low-trade activity in risk objective (default: 1.0)",
    )
    train_parser.add_argument(
        "--objective-weight-cvar",
        type=float,
        default=0.4,
        help="CVaR/expected shortfall penalty weight in objective (default: 0.4)",
    )
    train_parser.add_argument(
        "--objective-weight-skew",
        type=float,
        default=0.1,
        help="Negative return skew penalty weight in objective (default: 0.1)",
    )
    train_parser.add_argument(
        "--replay-manifest",
        type=Path,
        default=None,
        help="Replay a prior training run from replay/promotion/artifact manifest JSON",
    )
    train_parser.add_argument(
        "--gpu",
        action="store_true",
        help="Deprecated: institutional mode enforces GPU automatically",
    )
    train_parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Deprecated: institutional mode enforces GPU automatically",
    )
    train_parser.add_argument(
        "--no-database",
        action="store_true",
        help="Forbidden in institutional mode; kept for explicit fail-fast validation",
    )
    train_parser.add_argument(
        "--no-redis-cache",
        action="store_true",
        help="Forbidden in institutional mode; kept for explicit fail-fast validation",
    )
    train_parser.add_argument(
        "--no-shap",
        action="store_true",
        help="Forbidden in institutional mode; kept for explicit fail-fast validation",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs for deep learning models (default: 100)",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for deep learning models (default: 64)",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for deep learning models (default: 0.001)",
    )
    train_parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.45,
        help="Validation gate threshold for mean accuracy (default: 0.45)",
    )
    train_parser.add_argument(
        "--min-trades",
        type=int,
        default=100,
        help="Validation gate threshold for minimum mean trade count per fold (default: 100)",
    )
    train_parser.add_argument(
        "--holdout-pct",
        type=float,
        default=0.15,
        help="Terminal untouched holdout split ratio by timestamp (default: 0.15)",
    )
    train_parser.add_argument(
        "--min-holdout-sharpe",
        type=float,
        default=0.0,
        help="Validation gate threshold for holdout Sharpe ratio (default: 0.0)",
    )
    train_parser.add_argument(
        "--max-holdout-drawdown",
        type=float,
        default=0.35,
        help="Validation gate threshold for holdout max drawdown (default: 0.35)",
    )
    train_parser.add_argument(
        "--max-regime-shift",
        type=float,
        default=0.35,
        help="Maximum allowed regime distribution shift across train/test/holdout (default: 0.35)",
    )
    train_parser.add_argument(
        "--label-horizons",
        nargs="+",
        type=int,
        default=[1, 5, 20],
        help="Target horizons in bars (default: 1 5 20)",
    )
    train_parser.add_argument(
        "--primary-horizon",
        type=int,
        default=5,
        help="Primary horizon used for cost-aware labeling (default: 5)",
    )
    train_parser.add_argument(
        "--profit-taking",
        type=float,
        default=0.015,
        help="Triple-barrier profit-taking threshold (default: 0.015)",
    )
    train_parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.010,
        help="Triple-barrier stop-loss threshold (default: 0.010)",
    )
    train_parser.add_argument(
        "--max-holding",
        type=int,
        default=20,
        help="Triple-barrier max holding period in bars (default: 20)",
    )
    train_parser.add_argument(
        "--spread-bps",
        type=float,
        default=1.0,
        help="Cost-aware labeling spread assumption in bps (default: 1.0)",
    )
    train_parser.add_argument(
        "--slippage-bps",
        type=float,
        default=3.0,
        help="Cost-aware labeling slippage assumption in bps (default: 3.0)",
    )
    train_parser.add_argument(
        "--impact-bps",
        type=float,
        default=2.0,
        help="Cost-aware labeling impact assumption in bps (default: 2.0)",
    )
    train_parser.add_argument(
        "--label-min-signal-abs-return-bps",
        type=float,
        default=8.0,
        help="Minimum absolute forward return for signal candidacy in bps (default: 8.0)",
    )
    train_parser.add_argument(
        "--label-neutral-buffer-bps",
        type=float,
        default=4.0,
        help="Neutral edge buffer around zero net return in bps (default: 4.0)",
    )
    train_parser.add_argument(
        "--label-edge-cost-buffer-bps",
        type=float,
        default=2.0,
        help="Additional edge-over-cost buffer in bps for durable labels (default: 2.0)",
    )
    train_parser.add_argument(
        "--label-max-abs-forward-return",
        type=float,
        default=0.35,
        help="Absolute forward-return outlier cap for labeling robustness (default: 0.35)",
    )
    train_parser.add_argument(
        "--label-signal-volatility-floor-mult",
        type=float,
        default=0.50,
        help="Signal floor volatility multiplier for adaptive label filtering (default: 0.50)",
    )
    train_parser.add_argument(
        "--label-volatility-lookback",
        type=int,
        default=20,
        help="Lookback for volatility-adaptive barriers (default: 20)",
    )
    train_parser.add_argument(
        "--label-regime-lookback",
        type=int,
        default=30,
        help="Lookback for regime-aware labels (default: 30)",
    )
    train_parser.add_argument(
        "--label-temporal-weight-decay",
        type=float,
        default=0.999,
        help="Temporal sample weight decay (default: 0.999)",
    )
    train_parser.add_argument(
        "--feature-groups",
        nargs="+",
        default=["technical", "statistical", "microstructure", "cross_sectional"],
        help=(
            "Feature groups to compute (default: technical statistical "
            "microstructure cross_sectional)"
        ),
    )
    train_parser.add_argument(
        "--disable-cross-sectional",
        action="store_true",
        help="Disable cross-sectional features explicitly",
    )
    train_parser.add_argument(
        "--strict-feature-groups",
        action="store_true",
        help="Fail training if any requested feature group cannot be materialized",
    )
    train_parser.add_argument(
        "--max-cross-sectional-symbols",
        type=int,
        default=20,
        help="Adaptive guardrail for cross-sectional features by symbol count",
    )
    train_parser.add_argument(
        "--max-cross-sectional-rows",
        type=int,
        default=250000,
        help="Adaptive guardrail for cross-sectional features by row count",
    )
    train_parser.add_argument(
        "--feature-materialization-batch-rows",
        type=int,
        default=5000,
        help="Source-row chunk size while writing features to PostgreSQL",
    )
    train_parser.add_argument(
        "--feature-reuse-min-coverage",
        type=float,
        default=0.20,
        help="Minimum usable feature-row ratio to reuse PostgreSQL feature cache",
    )
    train_parser.add_argument(
        "--skip-feature-persist",
        action="store_true",
        help="Skip writing computed features back to PostgreSQL for this training run",
    )
    train_parser.add_argument(
        "--disable-dynamic-no-trade-band",
        action="store_true",
        help="Disable adaptive no-trade thresholding in execution-aware evaluation",
    )
    train_parser.add_argument(
        "--execution-vol-target-daily",
        type=float,
        default=0.012,
        help="Daily volatility target used for position scaling during evaluation (default: 0.012)",
    )
    train_parser.add_argument(
        "--execution-turnover-cap",
        type=float,
        default=0.90,
        help="Maximum average turnover cap used in execution-aware evaluation (default: 0.90)",
    )
    train_parser.add_argument(
        "--execution-cooldown-bars",
        type=int,
        default=2,
        help="Cooldown bars before direction flips in execution-aware evaluation (default: 2)",
    )
    train_parser.add_argument(
        "--min-deflated-sharpe",
        type=float,
        default=0.10,
        help="Hard promotion gate for deflated Sharpe (default: 0.10)",
    )
    train_parser.add_argument(
        "--max-deflated-sharpe-pvalue",
        type=float,
        default=0.10,
        help="Hard promotion gate for deflated Sharpe p-value (default: 0.10)",
    )
    train_parser.add_argument(
        "--max-pbo",
        type=float,
        default=0.45,
        help="Hard promotion gate for probability of backtest overfitting (default: 0.45)",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Output directory for trained models",
    )
    train_parser.set_defaults(func=cmd_train)

    # --------------------------------------------------------------------------
    # DATA COMMAND
    # --------------------------------------------------------------------------
    data_parser = subparsers.add_parser(
        "data",
        help="Data management operations",
        description="Load, validate, and manage trading data.",
    )
    data_subparsers = data_parser.add_subparsers(
        title="Operations",
        dest="data_command",
        metavar="<operation>",
    )

    # data load
    data_load = data_subparsers.add_parser("load", help="Load historical data")
    data_load.add_argument(
        "--source",
        choices=["alpaca", "csv", "database"],
        default="alpaca",
        help="Data source (default: alpaca)",
    )
    data_load.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Symbols to load",
    )
    data_load.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)",
    )
    data_load.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD)",
    )
    data_load.add_argument(
        "--timeframe",
        default="15Min",
        help="Bar timeframe (default: 15Min)",
    )
    data_load.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory (default: data/raw)",
    )
    data_load.add_argument(
        "--sync-db",
        action="store_true",
        help="Upsert downloaded bars into PostgreSQL/TimescaleDB",
    )
    data_load.add_argument(
        "--incremental",
        action="store_true",
        help="If --sync-db, fetch each symbol from latest DB bar to now",
    )
    data_load.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Batch size for database upserts (default: 5000)",
    )

    # data download (alias of load)
    data_download = data_subparsers.add_parser(
        "download",
        help="Download historical data (alias for load)",
    )
    data_download.add_argument(
        "--source",
        choices=["alpaca", "csv", "database"],
        default="alpaca",
        help="Data source (default: alpaca)",
    )
    data_download.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Symbols to load",
    )
    data_download.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)",
    )
    data_download.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD)",
    )
    data_download.add_argument(
        "--timeframe",
        default="15Min",
        help="Bar timeframe (default: 15Min)",
    )
    data_download.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory (default: data/raw)",
    )
    data_download.add_argument(
        "--sync-db",
        action="store_true",
        help="Upsert downloaded bars into PostgreSQL/TimescaleDB",
    )
    data_download.add_argument(
        "--incremental",
        action="store_true",
        help="If --sync-db, fetch each symbol from latest DB bar to now",
    )
    data_download.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Batch size for database upserts (default: 5000)",
    )

    # data validate
    data_validate = data_subparsers.add_parser("validate", help="Validate data quality")
    data_validate.add_argument(
        "--path",
        type=Path,
        default=Path("data/raw"),
        help="Data directory to validate",
    )
    data_validate.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any validation error",
    )

    # data export
    data_export = data_subparsers.add_parser("export", help="Export data")
    data_export.add_argument(
        "--format",
        choices=["csv", "parquet", "hdf5"],
        default="parquet",
        help="Export format (default: parquet)",
    )
    data_export.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to export (default: all)",
    )
    data_export.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path",
    )

    # data migrate - Migrate CSV to PostgreSQL
    data_migrate = data_subparsers.add_parser(
        "migrate",
        help="Migrate CSV data to PostgreSQL + TimescaleDB",
    )
    data_migrate.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw"),
        help="Source directory containing CSV files (default: data/raw)",
    )
    data_migrate.add_argument(
        "--symbols",
        nargs="+",
        help="Specific symbols to migrate (default: all)",
    )
    data_migrate.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Batch size for bulk inserts (default: 50000)",
    )
    data_migrate.add_argument(
        "--verify",
        action="store_true",
        help="Verify migration after each symbol",
    )
    data_migrate.add_argument(
        "--source-timezone",
        type=str,
        default="America/New_York",
        help="Timezone for naive source timestamps (default: America/New_York)",
    )
    data_migrate.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resumable chunk checkpoints during migration",
    )
    data_migrate.add_argument(
        "--export-after",
        action="store_true",
        help="Export to Parquet after migration",
    )

    # data export-training - Export PostgreSQL to Parquet for ML
    data_export_training = data_subparsers.add_parser(
        "export-training",
        help="Export PostgreSQL data to Parquet for ML training",
    )
    data_export_training.add_argument(
        "--output",
        type=Path,
        default=Path("data/training"),
        help="Output directory for Parquet files (default: data/training)",
    )
    data_export_training.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to export (default: all)",
    )
    data_export_training.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)",
    )
    data_export_training.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD)",
    )
    data_export_training.add_argument(
        "--ohlcv-only",
        action="store_true",
        help="Export only OHLCV data (no features)",
    )
    data_export_training.add_argument(
        "--features-only",
        action="store_true",
        help="Export only feature data (no OHLCV)",
    )

    # data db-status - Database status check
    data_db_status = data_subparsers.add_parser(
        "db-status",
        help="Check PostgreSQL + TimescaleDB database status",
    )
    data_db_status.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed database information",
    )

    data_parser.set_defaults(func=cmd_data)

    # --------------------------------------------------------------------------
    # FEATURES COMMAND
    # --------------------------------------------------------------------------
    features_parser = subparsers.add_parser(
        "features",
        help="Feature engineering pipeline",
        description="Compute and manage trading features.",
    )
    features_subparsers = features_parser.add_subparsers(
        title="Operations",
        dest="operation",
        metavar="<operation>",
    )

    # features compute
    features_compute = features_subparsers.add_parser("compute", help="Compute features")
    features_compute.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Symbols to compute features for",
    )
    features_compute.add_argument(
        "--groups",
        nargs="+",
        default=["technical", "statistical", "microstructure"],
        help="Feature groups to compute",
    )
    features_compute.add_argument(
        "--cache",
        action="store_true",
        help="Cache computed features",
    )
    features_compute.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration",
    )
    features_compute.add_argument(
        "--output",
        type=Path,
        help="Output file for feature matrix",
    )

    # features validate
    features_validate = features_subparsers.add_parser("validate", help="Validate features")
    features_validate.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Path to feature file",
    )
    features_validate.add_argument(
        "--check-leakage",
        action="store_true",
        default=True,
        help="Check for look-ahead bias",
    )

    features_parser.set_defaults(func=cmd_features)

    # --------------------------------------------------------------------------
    # HEALTH COMMAND
    # --------------------------------------------------------------------------
    health_parser = subparsers.add_parser(
        "health",
        help="System health and diagnostics",
        description="Check system health and run diagnostics.",
    )
    health_subparsers = health_parser.add_subparsers(
        title="Operations",
        dest="operation",
        metavar="<operation>",
    )

    # health check
    health_check = health_subparsers.add_parser("check", help="Run health checks")
    health_check.add_argument(
        "--full",
        action="store_true",
        help="Run full diagnostic suite",
    )
    health_check.add_argument(
        "--component",
        choices=["connectivity", "models", "data", "risk", "performance", "all"],
        default="all",
        help="Component to check (default: all)",
    )
    health_check.add_argument(
        "--output",
        type=Path,
        help="Output file for health report",
    )

    # health metrics
    health_metrics = health_subparsers.add_parser("metrics", help="Expose Prometheus metrics")
    health_metrics.add_argument(
        "--port",
        type=int,
        default=9090,
        help="Metrics server port (default: 9090)",
    )

    health_parser.set_defaults(func=cmd_health)

    # --------------------------------------------------------------------------
    # DEPLOY COMMAND
    # --------------------------------------------------------------------------
    deploy_parser = subparsers.add_parser(
        "deploy",
        help="Deployment and setup operations",
        description="Deploy and configure the trading system.",
    )
    deploy_subparsers = deploy_parser.add_subparsers(
        title="Operations",
        dest="operation",
        metavar="<operation>",
    )

    # deploy setup
    deploy_setup = deploy_subparsers.add_parser("setup", help="Initialize system")
    deploy_setup.add_argument(
        "--env",
        choices=["development", "staging", "production"],
        default="development",
        help="Environment (default: development)",
    )
    deploy_setup.add_argument(
        "--region",
        choices=["US_EAST", "US_WEST", "EU_WEST", "ASIA_PACIFIC"],
        default="US_EAST",
        help="Deployment region (default: US_EAST)",
    )
    deploy_setup.add_argument(
        "--force",
        action="store_true",
        help="Force re-initialization",
    )

    # deploy migrate
    deploy_migrate = deploy_subparsers.add_parser("migrate", help="Run database migrations")
    deploy_migrate.add_argument(
        "--revision",
        type=str,
        help="Target revision (default: head)",
    )
    deploy_migrate.add_argument(
        "--dry-run",
        action="store_true",
        help="Show migration plan without executing",
    )

    # deploy docker
    deploy_docker = deploy_subparsers.add_parser("docker", help="Docker operations")
    deploy_docker.add_argument(
        "action",
        choices=["build", "up", "down", "logs", "status"],
        help="Docker action",
    )
    deploy_docker.add_argument(
        "--profile",
        choices=["default", "production"],
        default="default",
        help="Compose profile (default: default)",
    )
    deploy_docker.add_argument(
        "--compose-file",
        default="docker-compose.yml",
        help="Compose file name, or comma-separated list for custom stacking",
    )
    deploy_docker.add_argument(
        "--services",
        nargs="+",
        help="Optional list of services for up/build",
    )
    deploy_docker.add_argument(
        "--service",
        help="Single service for logs",
    )
    deploy_docker.add_argument(
        "--tail",
        type=int,
        default=200,
        help="Number of log lines for deploy docker logs (default: 200)",
    )
    deploy_docker.add_argument(
        "--volumes",
        action="store_true",
        help="Remove volumes when running deploy docker down",
    )
    deploy_docker.add_argument(
        "--no-cache",
        action="store_true",
        help="Use no-cache when running deploy docker build",
    )
    deploy_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate deployment actions",
    )
    deploy_parser.set_defaults(func=cmd_deploy)

    # --------------------------------------------------------------------------
    # DASHBOARD COMMAND
    # --------------------------------------------------------------------------
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Run monitoring dashboard",
        description="Start the real-time monitoring dashboard.",
    )
    dashboard_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    dashboard_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    dashboard_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (dev mode)",
    )
    dashboard_parser.set_defaults(func=cmd_dashboard)

    return parser


# ==============================================================================
# SIGNAL HANDLERS
# ==============================================================================

def setup_signal_handlers() -> None:
    """Setup graceful shutdown signal handlers."""

    def signal_handler(signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        logger = get_logger("main", "SYSTEM")
        logger.warning(f"Received signal {signum}, initiating graceful shutdown...")

        # Publish shutdown event
        try:
            from quant_trading_system.core.events import EventBus, EventType, create_system_event
            event_bus = EventBus()
            event_bus.publish(create_system_event(
                EventType.SYSTEM_STOP,
                {"reason": f"Signal {signum}", "graceful": True},
            ))
        except Exception:
            pass

        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


# ==============================================================================
# COMPATIBILITY API
# ==============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments (compatibility wrapper used by tests)."""
    parser = create_parser()
    return parser.parse_args(argv)


class TradingSystemApp:
    """Lightweight async app wrapper used by unit tests and orchestration."""

    def __init__(self, settings: Any):
        self.settings = settings
        self._running = False
        self._metrics: dict[str, Any] = {}
        self._logger = get_logger("trading_app", "SYSTEM")

    async def start(self, mode: str = "paper") -> None:
        """Start app loop until stop() is called."""
        self._running = True
        self._logger.info(f"TradingSystemApp started in {mode} mode")

        while self._running:
            self._update_system_metrics()
            await asyncio.sleep(1.0)

    async def stop(self) -> None:
        """Stop app loop."""
        self._running = False
        self._logger.info("TradingSystemApp stopped")

    def _update_system_metrics(self) -> None:
        """Best-effort system metric update (never raises)."""
        try:
            import psutil

            process = psutil.Process()
            self._metrics = {
                "memory_mb": process.memory_info().rss / (1024 * 1024),
                "cpu_percent": process.cpu_percent(interval=None),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception:
            self._metrics = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main() -> int:
    """Main entry point for the AlphaTrade CLI."""

    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(
        log_level=args.log_level,
        log_format=args.log_format,
    )

    # Print banner unless quiet mode
    if not args.quiet:
        print(BANNER)

    # Setup signal handlers
    setup_signal_handlers()

    # Check if command was provided
    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    logger = get_logger("main", "SYSTEM")

    try:
        logger.info(f"Executing command: {args.command}")
        start_time = datetime.now(timezone.utc)

        # Run the command handler
        exit_code = args.func(args)

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(f"Command '{args.command}' completed in {elapsed:.2f}s with exit code {exit_code}")

        return exit_code

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130

    except Exception as e:
        logger.exception(f"Command '{args.command}' failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
