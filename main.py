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
    fmt = getattr(LogFormat, log_format.upper(), LogFormat.CONSOLE)
    _setup_logging(log_level=log_level, log_format=fmt)


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
    from scripts.backtest import run_backtest
    return run_backtest(args)


def cmd_train(args: argparse.Namespace) -> int:
    """
    Train ML models.

    @mlquant: Machine learning pipeline with institutional-grade validation.

    Model Types:
      - xgboost    : Gradient boosting (fast, interpretable)
      - lightgbm   : Light gradient boosting (efficient)
      - lstm       : Long short-term memory (sequential patterns)
      - transformer: Attention-based (complex dependencies)
      - ensemble   : IC-weighted model combination

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
      - Hyperparameter optimization with Optuna
    """
    from scripts.train import run_training
    return run_training(args)


def cmd_data(args: argparse.Namespace) -> int:
    """
    Data management operations.

    @data: Data loading, preprocessing, and quality control.

    Operations:
      - load      : Load historical data from sources
      - validate  : Validate data quality (gaps, outliers, OHLCV rules)
      - preprocess: Clean and normalize data
      - export    : Export data to various formats

    Sources:
      - alpaca    : Alpaca Markets API
      - csv       : Local CSV files
      - database  : PostgreSQL/TimescaleDB

    Features:
      - Intrinsic time bars (volume, dollar, imbalance, run bars)
      - Alternative data integration (news, social, satellite)
      - Data lineage tracking for regulatory compliance
      - Automatic gap filling and outlier handling
    """
    from scripts.data import run_data_management
    return run_data_management(args)


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
    from scripts.features import run_features
    return run_features(args)


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
    from scripts.health import run_health_check
    return run_health_check(args)


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
        choices=["CONSOLE", "JSON", "STRUCTURED"],
        default="CONSOLE",
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
        "--start", "-s",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    backtest_parser.add_argument(
        "--end", "-e",
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
        "--capital",
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
        choices=["xgboost", "lightgbm", "lstm", "transformer", "ensemble", "all"],
        default="xgboost",
        help="Model type to train (default: xgboost)",
    )
    train_parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        help="Symbols to train on",
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
        "--optimize",
        action="store_true",
        help="Run hyperparameter optimization",
    )
    train_parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration",
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
        dest="operation",
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
        "--service",
        type=str,
        help="Specific service (default: all)",
    )

    deploy_parser.set_defaults(func=cmd_deploy)

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
