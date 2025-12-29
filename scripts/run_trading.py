#!/usr/bin/env python3
"""
Main trading script.

Runs the trading system in live or paper trading mode.
"""

import argparse
import asyncio
import signal
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant_trading_system.config.settings import get_settings
from quant_trading_system.monitoring.logger import setup_logging, LogFormat, get_logger, LogCategory


logger = get_logger("run_trading", LogCategory.SYSTEM)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the quant trading system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["live", "paper"],
        default="paper",
        help="Trading mode (live or paper)",
    )
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no actual trades)",
    )
    return parser.parse_args()


async def run_trading_system(args: argparse.Namespace) -> None:
    """Run the trading system.

    Args:
        args: Command line arguments.
    """
    settings = get_settings()

    logger.info(
        f"Starting trading system in {args.mode} mode",
        extra={"extra_data": {"dry_run": args.dry_run}},
    )

    # Placeholder for actual trading system initialization
    # In a real implementation, this would:
    # 1. Initialize database connections
    # 2. Connect to broker
    # 3. Load models
    # 4. Start data feeds
    # 5. Run trading loop

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logger.info("Trading system shutdown requested")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_format = LogFormat.JSON if args.log_format == "json" else LogFormat.TEXT
    setup_logging(level=args.log_level, log_format=log_format)

    # Handle shutdown signals
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def signal_handler() -> None:
        logger.info("Received shutdown signal")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    try:
        loop.run_until_complete(run_trading_system(args))
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        loop.close()
        logger.info("Trading system stopped")


if __name__ == "__main__":
    main()
