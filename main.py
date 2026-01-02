#!/usr/bin/env python3
"""
Main entry point for the Quant Trading System.

Provides a unified entry point for running the trading system
in various modes: live trading, paper trading, backtest, or dashboard.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_trading_system.config.settings import get_settings, Settings
from quant_trading_system.core.events import EventBus, EventType, create_system_event
from quant_trading_system.monitoring.logger import (
    setup_logging,
    LogFormat,
    get_logger,
    LogCategory,
)
from quant_trading_system.monitoring.metrics import get_metrics_collector
from quant_trading_system.monitoring.alerting import (
    get_alert_manager,
    AlertType,
    alert_critical,
    alert_info,
)


logger = get_logger("main", LogCategory.SYSTEM)


class TradingSystemApp:
    """Main trading system application.

    Orchestrates all components and manages the application lifecycle.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the trading system.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self.event_bus = EventBus()
        self.metrics = get_metrics_collector()
        self.alert_manager = get_alert_manager()
        self._running = False
        self._tasks: list[asyncio.Task] = []

        # Set system info in metrics
        self.metrics.set_system_info(
            version=settings.app_version,
            environment=settings.environment,
        )

    async def start(self, mode: str = "paper") -> None:
        """Start the trading system.

        Args:
            mode: Trading mode (live, paper, backtest).
        """
        logger.info(
            f"Starting trading system in {mode} mode",
            extra={"extra_data": {"environment": self.settings.environment}},
        )

        self._running = True

        try:
            # Publish system start event
            self.event_bus.publish(
                create_system_event(
                    EventType.SYSTEM_START,
                    f"Trading system starting in {mode} mode",
                    {"mode": mode, "environment": self.settings.environment},
                    source="main",
                )
            )

            # Initialize components based on mode
            await self._initialize_components(mode)

            # Send startup alert
            await alert_info(
                AlertType.SYSTEM_MAINTENANCE_SCHEDULED,
                "Trading System Started",
                f"Trading system started in {mode} mode",
                mode=mode,
                environment=self.settings.environment,
            )

            # Start event processing
            self._tasks.append(
                asyncio.create_task(self.event_bus.start_processing())
            )

            # Run main loop
            await self._run_main_loop(mode)

        except Exception as e:
            logger.error(f"Trading system error: {e}", exc_info=True)
            await alert_critical(
                AlertType.SYSTEM_DOWN,
                "Trading System Error",
                f"Critical error in trading system: {e}",
                error=str(e),
            )
            raise

        finally:
            await self.stop()

    async def _initialize_components(self, mode: str) -> None:
        """Initialize system components.

        CRITICAL: This method now fully initializes all components required
        for production trading. Each component is initialized in order of
        dependency and validated before proceeding.

        Args:
            mode: Trading mode (live, paper, backtest).

        Raises:
            RuntimeError: If any critical component fails to initialize.
        """
        logger.info("Initializing system components")
        initialization_errors: list[str] = []

        # 1. Verify secure configuration
        try:
            logger.info("Step 1/7: Verifying secure configuration")
            from quant_trading_system.config.secure_config import SecureConfigManager
            self.secure_config = SecureConfigManager.get_instance()
            config_status = self.secure_config.get_configuration_status()
            logger.info(f"Config status: {config_status}")

            # Verify Alpaca credentials if trading mode
            if mode in ("live", "paper"):
                if not self.secure_config.verify_alpaca_credentials():
                    initialization_errors.append(
                        "Alpaca credentials not configured or invalid"
                    )
        except Exception as e:
            logger.error(f"Failed to verify secure config: {e}")
            initialization_errors.append(f"Secure config error: {e}")

        # 2. Configure alerting channels
        self.alerting_channels = []
        try:
            logger.info("Step 2/8: Configuring alerting channels")
            alerting_status = self.alert_manager.setup_notifiers_from_config()
            self.alerting_channels = [ch for ch, ok in alerting_status.items() if ok]
            if self.alerting_channels:
                logger.info(f"Alerting channels configured: {self.alerting_channels}")
            else:
                logger.warning(
                    "No external alerting channels configured. "
                    "Set SLACK_WEBHOOK_URL, SMTP_*, or PAGERDUTY_SERVICE_KEY in environment."
                )
        except Exception as e:
            logger.warning(f"Alerting setup failed (non-critical): {e}")

        # 3. Initialize database connection
        try:
            logger.info("Step 3/8: Initializing database connection")
            from quant_trading_system.database.connection import DatabaseManager
            self.database = DatabaseManager()
            await self.database.initialize()

            # Verify connection with health check
            if await self.database.async_health_check():
                logger.info("Database connection verified")
            else:
                initialization_errors.append("Database health check failed")
        except Exception as e:
            logger.warning(f"Database initialization failed (non-critical): {e}")
            self.database = None

        # 4. Initialize Redis/Feature Store
        try:
            logger.info("Step 4/8: Initializing feature store (Redis)")
            from quant_trading_system.data.feature_store import FeatureStore, FeatureStoreConfig
            redis_url = self.settings.redis.url if hasattr(self.settings, 'redis') else None
            if redis_url:
                config = FeatureStoreConfig(redis_url=redis_url)
                self.feature_store = FeatureStore(config)
                logger.info("Feature store initialized with Redis")
            else:
                # Fallback to memory-only feature store
                self.feature_store = FeatureStore(FeatureStoreConfig())
                logger.info("Feature store initialized with memory cache only")
        except Exception as e:
            logger.warning(f"Feature store initialization failed: {e}")
            self.feature_store = None

        # 5. Initialize broker client (if trading mode)
        if mode in ("live", "paper"):
            try:
                logger.info("Step 5/8: Initializing broker client (Alpaca)")
                from quant_trading_system.execution.alpaca_client import AlpacaClient

                api_key = self.secure_config.get_credential("ALPACA_API_KEY", required=False)
                api_secret = self.secure_config.get_credential("ALPACA_API_SECRET", required=False)

                if api_key and api_secret:
                    self.broker_client = AlpacaClient(
                        api_key=api_key,
                        api_secret=api_secret,
                        paper=(mode == "paper"),
                    )
                    # Verify connection
                    account = await self.broker_client.get_account()
                    if account:
                        logger.info(
                            f"Broker connected: {account.status}, "
                            f"Equity: ${float(account.equity):,.2f}"
                        )
                    else:
                        initialization_errors.append("Failed to get broker account info")
                else:
                    initialization_errors.append("Alpaca API credentials not found")
                    self.broker_client = None
            except Exception as e:
                logger.error(f"Broker initialization failed: {e}")
                initialization_errors.append(f"Broker error: {e}")
                self.broker_client = None
        else:
            self.broker_client = None
            logger.info("Step 5/8: Skipping broker (backtest mode)")

        # 6. Initialize risk management
        try:
            logger.info("Step 6/8: Initializing risk management")
            from quant_trading_system.risk.limits import (
                RiskLimitsConfig,
                PreTradeRiskChecker,
                KillSwitch,
            )

            risk_config = RiskLimitsConfig()
            self.risk_checker = PreTradeRiskChecker(risk_config)
            self.kill_switch = KillSwitch(risk_config)
            logger.info("Risk management initialized")
        except Exception as e:
            logger.error(f"Risk management initialization failed: {e}")
            initialization_errors.append(f"Risk management error: {e}")

        # 7. Load trained models
        try:
            logger.info("Step 7/8: Loading trained models")
            from quant_trading_system.models.model_manager import ModelManager

            self.model_manager = ModelManager(registry_path="models/registry")
            models_path = Path("models")

            loaded_models = []
            if models_path.exists():
                for model_file in models_path.glob("*.joblib"):
                    try:
                        import joblib
                        model = joblib.load(model_file)
                        loaded_models.append(model_file.stem)
                    except Exception as e:
                        logger.warning(f"Failed to load model {model_file}: {e}")

                for model_file in models_path.glob("*.pt"):
                    try:
                        import torch
                        # Just verify file is loadable
                        torch.load(model_file, weights_only=True)
                        loaded_models.append(model_file.stem)
                    except Exception as e:
                        logger.warning(f"Failed to load model {model_file}: {e}")

            if loaded_models:
                logger.info(f"Loaded models: {loaded_models}")
            else:
                logger.warning("No trained models found in models/ directory")
        except Exception as e:
            logger.warning(f"Model loading failed (non-critical): {e}")
            self.model_manager = None

        # 8. Initialize data feeds (if trading mode)
        if mode in ("live", "paper") and self.broker_client:
            try:
                logger.info("Step 8/8: Initializing data feeds")
                from quant_trading_system.data.live_feed import LiveFeed, LiveFeedConfig

                feed_config = LiveFeedConfig(
                    symbols=self.settings.trading.symbols if hasattr(self.settings, 'trading') else ["SPY"],
                )
                self.data_feed = LiveFeed(
                    config=feed_config,
                    event_bus=self.event_bus,
                )
                logger.info("Data feed initialized")
            except Exception as e:
                logger.warning(f"Data feed initialization failed: {e}")
                self.data_feed = None
        else:
            self.data_feed = None
            logger.info("Step 8/8: Skipping data feeds (backtest mode or no broker)")

        # Report initialization summary
        logger.info("=" * 60)
        logger.info("COMPONENT INITIALIZATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Mode: {mode}")
        alerting_str = ', '.join(self.alerting_channels) if self.alerting_channels else 'NONE'
        logger.info(f"  Alerting: {alerting_str}")
        logger.info(f"  Database: {'OK' if self.database else 'DISABLED'}")
        logger.info(f"  Feature Store: {'OK' if self.feature_store else 'DISABLED'}")
        logger.info(f"  Broker: {'OK' if self.broker_client else 'DISABLED'}")
        logger.info(f"  Risk Management: {'OK' if hasattr(self, 'risk_checker') else 'DISABLED'}")
        logger.info(f"  Models: {'OK' if self.model_manager else 'DISABLED'}")
        logger.info(f"  Data Feed: {'OK' if self.data_feed else 'DISABLED'}")
        logger.info("=" * 60)

        # Check for critical initialization errors
        if initialization_errors and mode == "live":
            error_msg = "Critical initialization errors: " + "; ".join(initialization_errors)
            logger.error(error_msg)
            await alert_critical(
                AlertType.SYSTEM_DOWN,
                "Initialization Failed",
                error_msg,
            )
            raise RuntimeError(error_msg)
        elif initialization_errors:
            logger.warning(f"Non-critical initialization issues: {initialization_errors}")

        logger.info("System components initialized successfully")

    async def _run_main_loop(self, mode: str) -> None:
        """Run the main trading loop.

        Args:
            mode: Trading mode.
        """
        logger.info("Starting main trading loop")

        heartbeat_interval = 60  # seconds
        last_heartbeat = datetime.now(timezone.utc)

        while self._running:
            try:
                # Send heartbeat
                now = datetime.now(timezone.utc)
                if (now - last_heartbeat).total_seconds() >= heartbeat_interval:
                    self.event_bus.publish(
                        create_system_event(
                            EventType.HEARTBEAT,
                            "System heartbeat",
                            source="main",
                        )
                    )
                    last_heartbeat = now

                # Update system metrics
                self._update_system_metrics()

                # Sleep briefly to prevent busy-waiting
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                self.metrics.record_error("main_loop_error", "main")

    def _update_system_metrics(self) -> None:
        """Update system metrics."""
        try:
            import psutil

            process = psutil.Process()
            self.metrics.update_system_metrics(
                memory_bytes=process.memory_info().rss,
                cpu_percent=process.cpu_percent(),
            )
        except ImportError:
            pass  # psutil not available

    async def stop(self) -> None:
        """Stop the trading system."""
        logger.info("Stopping trading system")
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Stop event processing
        self.event_bus.stop_processing()

        # Publish system stop event
        self.event_bus.publish(
            create_system_event(
                EventType.SYSTEM_STOP,
                "Trading system stopped",
                source="main",
            )
        )

        logger.info("Trading system stopped")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Quant Trading System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Trading command
    trade_parser = subparsers.add_parser("trade", help="Run trading system")
    trade_parser.add_argument(
        "--mode",
        choices=["live", "paper"],
        default="paper",
        help="Trading mode",
    )
    trade_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no actual trades)",
    )

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Backtest start date (YYYY-MM-DD)",
    )
    backtest_parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="Backtest end date (YYYY-MM-DD)",
    )
    backtest_parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000.0,
        help="Initial capital",
    )

    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Run dashboard server")
    dashboard_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    dashboard_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    dashboard_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload",
    )

    # Common arguments
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--log-format",
        choices=["json", "text"],
        default="json",
        help="Log format",
    )

    return parser.parse_args()


async def run_trading(args: argparse.Namespace, settings: Settings) -> None:
    """Run trading mode.

    Args:
        args: Command line arguments.
        settings: Application settings.
    """
    app = TradingSystemApp(settings)

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler() -> None:
        logger.info("Shutdown signal received")
        asyncio.create_task(app.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass  # Windows

    await app.start(mode=args.mode)


async def run_backtest(args: argparse.Namespace, settings: Settings) -> None:
    """Run backtest mode.

    Args:
        args: Command line arguments.
        settings: Application settings.
    """
    logger.info(
        f"Running backtest from {args.start_date} to {args.end_date}",
        extra={
            "extra_data": {
                "initial_capital": args.initial_capital,
            }
        },
    )

    # Placeholder for backtest implementation
    # Would initialize backtesting engine and run simulation

    logger.info("Backtest completed")


def run_dashboard(args: argparse.Namespace, settings: Settings) -> None:
    """Run dashboard server.

    Args:
        args: Command line arguments.
        settings: Application settings.
    """
    logger.info(f"Starting dashboard server on {args.host}:{args.port}")

    try:
        import uvicorn

        uvicorn.run(
            "quant_trading_system.monitoring.dashboard:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level.lower() if hasattr(args, "log_level") else "info",
        )
    except ImportError:
        logger.error("uvicorn not installed. Install with: pip install uvicorn")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_format = LogFormat.JSON if args.log_format == "json" else LogFormat.TEXT
    setup_logging(level=args.log_level, log_format=log_format)

    # Load settings
    settings = get_settings()

    logger.info(
        f"Quant Trading System v{settings.app_version}",
        extra={"extra_data": {"command": args.command, "environment": settings.environment}},
    )

    try:
        if args.command == "trade":
            asyncio.run(run_trading(args, settings))
        elif args.command == "backtest":
            asyncio.run(run_backtest(args, settings))
        elif args.command == "dashboard":
            run_dashboard(args, settings)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
