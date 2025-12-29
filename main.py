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
from datetime import datetime
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

        Args:
            mode: Trading mode.
        """
        logger.info("Initializing system components")

        # Placeholder for actual component initialization
        # In a real implementation, this would:
        # 1. Connect to database
        # 2. Connect to Redis
        # 3. Connect to broker (if live/paper)
        # 4. Load models
        # 5. Initialize data feeds
        # 6. Start monitoring

        logger.info("System components initialized")

    async def _run_main_loop(self, mode: str) -> None:
        """Run the main trading loop.

        Args:
            mode: Trading mode.
        """
        logger.info("Starting main trading loop")

        heartbeat_interval = 60  # seconds
        last_heartbeat = datetime.utcnow()

        while self._running:
            try:
                # Send heartbeat
                now = datetime.utcnow()
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
