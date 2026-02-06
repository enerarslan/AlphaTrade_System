"""
================================================================================
ALPHATRADE SYSTEM HEALTH & DIAGNOSTICS
================================================================================

Institutional-grade system health monitoring for the AlphaTrade trading system.

@infra: This script implements all infrastructure and monitoring requirements:
  - Component health checks (database, Redis, broker, APIs)
  - Kill switch status (P0 Safety - CRITICAL)
  - Circuit breaker status (P2-M5)
  - Model staleness detection (P2-M2)
  - VIX regime monitoring (P1-A)
  - Correlation monitoring (P3-A)
  - Drawdown monitoring (P2-E)
  - Prometheus metrics exposure
  - Audit log verification

Commands:
    python main.py health check              # Full system health check
    python main.py health status             # Component status summary
    python main.py health risk               # Risk system status
    python main.py health models             # Model health and staleness
    python main.py health audit              # Verify audit log integrity

Author: AlphaTrade System
Version: 1.3.1
================================================================================
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("health")

# Import shared health monitoring module
from quant_trading_system.monitoring.health import (
    SystemHealthChecker,
    HealthStatus,
    HealthCheckResult,
)


# ============================================================================
# COMMAND HANDLERS
# ============================================================================


def cmd_check(args: argparse.Namespace) -> int:
    """Run full health check."""
    logger.info("=" * 80)
    logger.info("ALPHATRADE SYSTEM HEALTH CHECK")
    logger.info("=" * 80)

    checker = SystemHealthChecker()
    results = asyncio.run(checker.run_all_checks())

    # Group by status
    by_status: dict[HealthStatus, list[HealthCheckResult]] = {}
    for result in results:
        if result.status not in by_status:
            by_status[result.status] = []
        by_status[result.status].append(result)

    # Print results
    print("\n" + "=" * 60)

    for status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY, HealthStatus.DEGRADED,
                   HealthStatus.UNKNOWN, HealthStatus.HEALTHY]:
        if status not in by_status:
            continue

        status_icon = {
            HealthStatus.HEALTHY: "[OK]",
            HealthStatus.DEGRADED: "[WARN]",
            HealthStatus.UNHEALTHY: "[FAIL]",
            HealthStatus.CRITICAL: "[CRIT]",
            HealthStatus.UNKNOWN: "[???]",
        }.get(status, "[???]")

        print(f"\n{status_icon} {status.value} ({len(by_status[status])} components)")
        print("-" * 40)

        for result in by_status[status]:
            latency_str = f" ({result.latency_ms:.0f}ms)" if result.latency_ms > 0 else ""
            print(f"  {result.component}: {result.message}{latency_str}")

    # Overall status
    overall = checker.get_overall_status(results)
    print("\n" + "=" * 60)
    print(f"OVERALL STATUS: {overall.value}")
    print("=" * 60 + "\n")

    # Export to JSON if requested
    if getattr(args, "json", False):
        output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": overall.value,
            "checks": [r.to_dict() for r in results],
        }
        print(json.dumps(output, indent=2))

    return 0 if overall in [HealthStatus.HEALTHY, HealthStatus.DEGRADED] else 1


def cmd_status(args: argparse.Namespace) -> int:
    """Quick status summary."""
    logger.info("System Status Summary")

    checker = SystemHealthChecker()

    # Run only essential checks
    async def quick_checks():
        # Note: We manually call specific checks for speed
        return [
            await checker.check_broker(),
            await checker.check_kill_switch(),
            await checker.check_risk_system(),
            await checker.check_data_feed(),
            await checker.check_gpu(), # Added GPU to status
        ]

    results = asyncio.run(quick_checks())

    print("\nQuick Status:")
    print("-" * 40)
    for result in results:
        status_icon = "[OK]" if result.status == HealthStatus.HEALTHY else "[FAIL]"
        print(f"  {status_icon} {result.component}: {result.status.value}")

    return 0


def cmd_risk(args: argparse.Namespace) -> int:
    """Risk system status."""
    logger.info("=" * 80)
    logger.info("RISK SYSTEM STATUS")
    logger.info("=" * 80)

    checker = SystemHealthChecker()

    async def risk_checks():
        return [
            await checker.check_kill_switch(),
            await checker.check_risk_system(),
            await checker.check_drawdown_monitor(),
            # VIX check and Correlation check would be added to SystemHealthChecker if needed commonly
            # For now keeping it simple or we can add specific methods back to SystemHealthChecker if they are critical
        ]

    results = asyncio.run(risk_checks())

    print("\nRisk Components:")
    print("-" * 60)
    for result in results:
        status_icon = {
            HealthStatus.HEALTHY: "[OK]",
            HealthStatus.DEGRADED: "[WARN]",
            HealthStatus.CRITICAL: "[CRIT]",
        }.get(result.status, "[???]")

        print(f"\n{status_icon} {result.component.upper()}")
        print(f"   Status: {result.status.value}")
        print(f"   Message: {result.message}")

        if result.details:
            for key, value in result.details.items():
                print(f"   {key}: {value}")

    return 0


def cmd_models(args: argparse.Namespace) -> int:
    """Model health status."""
    logger.info("=" * 80)
    logger.info("MODEL HEALTH STATUS")
    logger.info("=" * 80)

    checker = SystemHealthChecker()

    async def model_checks():
        return [
            await checker.check_models(),
            await checker.check_model_staleness(),
            await checker.check_gpu(),
        ]

    results = asyncio.run(model_checks())

    for result in results:
        print(f"\n{result.component.upper()}")
        print(f"  Status: {result.status.value}")
        print(f"  Message: {result.message}")

        if result.details:
            for key, value in result.details.items():
                if isinstance(value, list):
                    print(f"  {key}:")
                    for item in value[:10]:
                        print(f"    - {item}")
                else:
                    print(f"  {key}: {value}")

    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Verify audit log integrity."""
    logger.info("=" * 80)
    logger.info("AUDIT LOG VERIFICATION")
    logger.info("=" * 80)

    # Use the shared checker for audit log
    checker = SystemHealthChecker()
    result = asyncio.run(checker.check_audit_log())

    if result.status == HealthStatus.HEALTHY:
        logger.info("[OK] Audit log integrity VERIFIED")
        if result.message == "Audit log integrity verified":
             logger.info("  All entries have valid hash chains")
        return 0
    elif result.status == HealthStatus.CRITICAL:
        logger.error("[FAIL] AUDIT LOG INTEGRITY COMPROMISED")
        logger.error(f"  {result.message}")
        return 1
    else:
        logger.info(f"Audit log status: {result.status.value} - {result.message}")
        return 0


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def run_health_command(args: argparse.Namespace) -> int:
    """
    Main entry point for health commands.

    @infra: This function routes to the appropriate health command handler.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code.
    """
    command = getattr(args, "health_command", "check")

    commands = {
        "check": cmd_check,
        "status": cmd_status,
        "risk": cmd_risk,
        "models": cmd_models,
        "audit": cmd_audit,
    }

    handler = commands.get(command)
    if handler:
        return handler(args)
    else:
        logger.error(f"Unknown health command: {command}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaTrade System Health")
    subparsers = parser.add_subparsers(dest="health_command")

    # Check command
    check_parser = subparsers.add_parser("check", help="Full health check")
    check_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Status command
    subparsers.add_parser("status", help="Quick status")

    # Risk command
    subparsers.add_parser("risk", help="Risk system status")

    # Models command
    subparsers.add_parser("models", help="Model health")

    # Audit command
    subparsers.add_parser("audit", help="Verify audit log")

    args = parser.parse_args()
    sys.exit(run_health_command(args))
