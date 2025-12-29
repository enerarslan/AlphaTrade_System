#!/usr/bin/env python3
"""
Dashboard server script.

Runs the FastAPI dashboard server for monitoring and control.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant_trading_system.config.settings import get_settings
from quant_trading_system.monitoring.logger import setup_logging, LogFormat, get_logger, LogCategory


logger = get_logger("run_dashboard", LogCategory.SYSTEM)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the monitoring dashboard server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes",
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
        default="text",
        help="Log format",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_format = LogFormat.JSON if args.log_format == "json" else LogFormat.TEXT
    setup_logging(level=args.log_level, log_format=log_format)

    logger.info(
        f"Starting dashboard server on {args.host}:{args.port}",
        extra={
            "extra_data": {
                "reload": args.reload,
                "workers": args.workers,
            }
        },
    )

    try:
        import uvicorn

        uvicorn.run(
            "quant_trading_system.monitoring.dashboard:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level=args.log_level.lower(),
        )
    except ImportError:
        logger.error("uvicorn not installed. Install with: pip install uvicorn")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Dashboard server failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
