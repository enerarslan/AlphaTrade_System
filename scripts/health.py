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
Version: 1.3.0
================================================================================
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

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


# ============================================================================
# HEALTH STATUS TYPES
# ============================================================================


class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"
    CRITICAL = "CRITICAL"


class ComponentType(str, Enum):
    """System component types."""
    DATABASE = "database"
    REDIS = "redis"
    BROKER = "broker"
    DATA_FEED = "data_feed"
    MODEL = "model"
    RISK = "risk"
    EXECUTION = "execution"
    MONITORING = "monitoring"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    status: HealthStatus
    message: str
    latency_ms: float = 0.0
    details: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# HEALTH CHECKER
# ============================================================================


class SystemHealthChecker:
    """
    Comprehensive system health checker.

    @infra: This class performs health checks on all system components
    including databases, caches, brokers, and trading subsystems.
    """

    def __init__(self):
        self.logger = logging.getLogger("SystemHealthChecker")
        self.check_results: list[HealthCheckResult] = []

    async def run_all_checks(self) -> list[HealthCheckResult]:
        """Run all health checks."""
        self.logger.info("Running comprehensive health checks...")
        self.check_results = []

        # Core infrastructure
        checks = [
            self.check_database(),
            self.check_redis(),
            self.check_broker(),
            self.check_data_feed(),
        ]

        # Trading systems
        checks.extend([
            self.check_kill_switch(),
            self.check_circuit_breakers(),
            self.check_risk_system(),
            self.check_drawdown_monitor(),
        ])

        # Models
        checks.extend([
            self.check_models(),
            self.check_model_staleness(),
        ])

        # Monitoring
        checks.extend([
            self.check_metrics_collector(),
            self.check_audit_log(),
        ])

        # Market conditions
        checks.extend([
            self.check_vix_regime(),
            self.check_correlation_monitor(),
        ])

        # Run all checks
        for check in checks:
            try:
                result = await check
                self.check_results.append(result)
            except Exception as e:
                self.check_results.append(HealthCheckResult(
                    component="unknown",
                    status=HealthStatus.UNKNOWN,
                    message=f"Check failed: {e}",
                ))

        return self.check_results

    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.check_results:
            return HealthStatus.UNKNOWN

        statuses = [r.status for r in self.check_results]

        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY

        return HealthStatus.UNKNOWN

    # ========================================================================
    # INFRASTRUCTURE CHECKS
    # ========================================================================

    async def check_database(self) -> HealthCheckResult:
        """Check database connectivity and health."""
        start = time.time()

        try:
            from quant_trading_system.database.connection import get_connection

            conn = await get_connection()
            await conn.execute("SELECT 1")

            latency = (time.time() - start) * 1000

            return HealthCheckResult(
                component="database",
                status=HealthStatus.HEALTHY,
                message="PostgreSQL connection OK",
                latency_ms=latency,
                details={"type": "postgresql"},
            )

        except ImportError:
            return HealthCheckResult(
                component="database",
                status=HealthStatus.UNKNOWN,
                message="Database module not available",
                latency_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                component="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database error: {e}",
                latency_ms=(time.time() - start) * 1000,
            )

    async def check_redis(self) -> HealthCheckResult:
        """Check Redis connectivity."""
        start = time.time()

        try:
            import redis

            client = redis.Redis(
                host=os.getenv("REDIS__HOST", "localhost"),
                port=int(os.getenv("REDIS__PORT", 6379)),
                socket_timeout=5,
            )

            client.ping()
            info = client.info()

            latency = (time.time() - start) * 1000

            return HealthCheckResult(
                component="redis",
                status=HealthStatus.HEALTHY,
                message="Redis connection OK",
                latency_ms=latency,
                details={
                    "version": info.get("redis_version", "unknown"),
                    "used_memory": info.get("used_memory_human", "unknown"),
                    "connected_clients": info.get("connected_clients", 0),
                },
            )

        except ImportError:
            return HealthCheckResult(
                component="redis",
                status=HealthStatus.UNKNOWN,
                message="Redis client not installed",
                latency_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                component="redis",
                status=HealthStatus.DEGRADED,
                message=f"Redis unavailable: {e}",
                latency_ms=(time.time() - start) * 1000,
            )

    async def check_broker(self) -> HealthCheckResult:
        """Check Alpaca broker connectivity."""
        start = time.time()

        try:
            from quant_trading_system.execution.alpaca_client import AlpacaClient

            client = AlpacaClient()
            account = await client.get_account()

            latency = (time.time() - start) * 1000

            if account:
                return HealthCheckResult(
                    component="broker",
                    status=HealthStatus.HEALTHY,
                    message="Alpaca connection OK",
                    latency_ms=latency,
                    details={
                        "account_status": getattr(account, "status", "unknown"),
                        "buying_power": str(getattr(account, "buying_power", 0)),
                        "equity": str(getattr(account, "equity", 0)),
                    },
                )
            else:
                return HealthCheckResult(
                    component="broker",
                    status=HealthStatus.UNHEALTHY,
                    message="Failed to get account info",
                    latency_ms=latency,
                )

        except ImportError:
            return HealthCheckResult(
                component="broker",
                status=HealthStatus.UNKNOWN,
                message="Alpaca client not available",
                latency_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                component="broker",
                status=HealthStatus.UNHEALTHY,
                message=f"Broker error: {e}",
                latency_ms=(time.time() - start) * 1000,
            )

    async def check_data_feed(self) -> HealthCheckResult:
        """Check data feed availability."""
        start = time.time()

        try:
            from quant_trading_system.data.loader import DataLoader

            loader = DataLoader()
            symbols = loader.get_available_symbols()

            latency = (time.time() - start) * 1000

            if symbols:
                return HealthCheckResult(
                    component="data_feed",
                    status=HealthStatus.HEALTHY,
                    message=f"Data available for {len(symbols)} symbols",
                    latency_ms=latency,
                    details={"symbol_count": len(symbols)},
                )
            else:
                return HealthCheckResult(
                    component="data_feed",
                    status=HealthStatus.DEGRADED,
                    message="No symbols available",
                    latency_ms=latency,
                )

        except ImportError:
            # Fallback: Check data directory
            data_dir = PROJECT_ROOT / "data" / "raw"
            if data_dir.exists():
                csv_files = list(data_dir.glob("*.csv"))
                return HealthCheckResult(
                    component="data_feed",
                    status=HealthStatus.HEALTHY if csv_files else HealthStatus.DEGRADED,
                    message=f"Found {len(csv_files)} data files",
                    latency_ms=(time.time() - start) * 1000,
                    details={"file_count": len(csv_files)},
                )

            return HealthCheckResult(
                component="data_feed",
                status=HealthStatus.UNHEALTHY,
                message="No data directory found",
                latency_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                component="data_feed",
                status=HealthStatus.UNHEALTHY,
                message=f"Data feed error: {e}",
                latency_ms=(time.time() - start) * 1000,
            )

    # ========================================================================
    # SAFETY CHECKS (P0 CRITICAL)
    # ========================================================================

    async def check_kill_switch(self) -> HealthCheckResult:
        """
        Check kill switch status.

        @trader P0 CRITICAL: Kill switch is the primary safety mechanism.
        If active, all trading is halted.
        """
        start = time.time()

        try:
            from quant_trading_system.risk.limits import KillSwitch

            kill_switch = KillSwitch()

            if kill_switch.is_active():
                return HealthCheckResult(
                    component="kill_switch",
                    status=HealthStatus.CRITICAL,
                    message="KILL SWITCH ACTIVE - Trading halted",
                    latency_ms=(time.time() - start) * 1000,
                    details={
                        "reason": str(getattr(kill_switch, "reason", "unknown")),
                        "activated_at": str(getattr(kill_switch, "activated_at", "unknown")),
                        "cooldown_remaining": getattr(kill_switch, "cooldown_remaining_seconds", 0),
                    },
                )
            else:
                return HealthCheckResult(
                    component="kill_switch",
                    status=HealthStatus.HEALTHY,
                    message="Kill switch inactive - Trading enabled",
                    latency_ms=(time.time() - start) * 1000,
                )

        except ImportError:
            return HealthCheckResult(
                component="kill_switch",
                status=HealthStatus.UNKNOWN,
                message="Kill switch module not available",
                latency_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                component="kill_switch",
                status=HealthStatus.UNKNOWN,
                message=f"Kill switch check failed: {e}",
                latency_ms=(time.time() - start) * 1000,
            )

    async def check_circuit_breakers(self) -> HealthCheckResult:
        """
        Check circuit breaker status.

        @infra P2-M5: Circuit breakers protect against external API failures.
        """
        start = time.time()

        try:
            from quant_trading_system.core.circuit_breaker import (
                CircuitBreakerRegistry,
                CircuitState,
            )

            registry = CircuitBreakerRegistry()
            breakers = registry.get_all_breakers()

            open_breakers = []
            half_open_breakers = []

            for name, breaker in breakers.items():
                state = breaker.get_state()
                if state == CircuitState.OPEN:
                    open_breakers.append(name)
                elif state == CircuitState.HALF_OPEN:
                    half_open_breakers.append(name)

            latency = (time.time() - start) * 1000

            if open_breakers:
                return HealthCheckResult(
                    component="circuit_breakers",
                    status=HealthStatus.DEGRADED,
                    message=f"{len(open_breakers)} circuit breakers OPEN",
                    latency_ms=latency,
                    details={
                        "open": open_breakers,
                        "half_open": half_open_breakers,
                        "total": len(breakers),
                    },
                )
            elif half_open_breakers:
                return HealthCheckResult(
                    component="circuit_breakers",
                    status=HealthStatus.DEGRADED,
                    message=f"{len(half_open_breakers)} circuit breakers recovering",
                    latency_ms=latency,
                    details={
                        "half_open": half_open_breakers,
                        "total": len(breakers),
                    },
                )
            else:
                return HealthCheckResult(
                    component="circuit_breakers",
                    status=HealthStatus.HEALTHY,
                    message=f"All {len(breakers)} circuit breakers closed",
                    latency_ms=latency,
                    details={"total": len(breakers)},
                )

        except ImportError:
            return HealthCheckResult(
                component="circuit_breakers",
                status=HealthStatus.UNKNOWN,
                message="Circuit breaker module not available",
                latency_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                component="circuit_breakers",
                status=HealthStatus.UNKNOWN,
                message=f"Circuit breaker check failed: {e}",
                latency_ms=(time.time() - start) * 1000,
            )

    # ========================================================================
    # RISK SYSTEM CHECKS
    # ========================================================================

    async def check_risk_system(self) -> HealthCheckResult:
        """Check risk management system status."""
        start = time.time()

        try:
            from quant_trading_system.risk.limits import PreTradeRiskChecker

            checker = PreTradeRiskChecker()

            # Verify risk checker is operational
            limits = checker.get_current_limits() if hasattr(checker, "get_current_limits") else {}

            latency = (time.time() - start) * 1000

            return HealthCheckResult(
                component="risk_system",
                status=HealthStatus.HEALTHY,
                message="Risk system operational",
                latency_ms=latency,
                details={
                    "max_position_pct": limits.get("max_position_pct", 0.10),
                    "max_daily_loss_pct": limits.get("max_daily_loss_pct", 0.05),
                    "max_drawdown_pct": limits.get("max_drawdown_pct", 0.15),
                },
            )

        except ImportError:
            return HealthCheckResult(
                component="risk_system",
                status=HealthStatus.UNKNOWN,
                message="Risk system module not available",
                latency_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                component="risk_system",
                status=HealthStatus.UNHEALTHY,
                message=f"Risk system error: {e}",
                latency_ms=(time.time() - start) * 1000,
            )

    async def check_drawdown_monitor(self) -> HealthCheckResult:
        """
        Check drawdown monitoring status.

        @infra P2-E: Intraday drawdown alerts for risk management.
        """
        start = time.time()

        try:
            from quant_trading_system.risk.drawdown_monitor import IntradayDrawdownMonitor

            monitor = IntradayDrawdownMonitor()
            status = monitor.get_status() if hasattr(monitor, "get_status") else {}

            latency = (time.time() - start) * 1000

            current_drawdown = status.get("current_drawdown_pct", 0)

            if current_drawdown > 0.10:  # > 10%
                return HealthCheckResult(
                    component="drawdown_monitor",
                    status=HealthStatus.CRITICAL,
                    message=f"Drawdown CRITICAL: {current_drawdown:.1%}",
                    latency_ms=latency,
                    details=status,
                )
            elif current_drawdown > 0.05:  # > 5%
                return HealthCheckResult(
                    component="drawdown_monitor",
                    status=HealthStatus.DEGRADED,
                    message=f"Drawdown elevated: {current_drawdown:.1%}",
                    latency_ms=latency,
                    details=status,
                )
            else:
                return HealthCheckResult(
                    component="drawdown_monitor",
                    status=HealthStatus.HEALTHY,
                    message=f"Drawdown normal: {current_drawdown:.1%}",
                    latency_ms=latency,
                    details=status,
                )

        except ImportError:
            return HealthCheckResult(
                component="drawdown_monitor",
                status=HealthStatus.UNKNOWN,
                message="Drawdown monitor not available",
                latency_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                component="drawdown_monitor",
                status=HealthStatus.UNKNOWN,
                message=f"Drawdown monitor error: {e}",
                latency_ms=(time.time() - start) * 1000,
            )

    # ========================================================================
    # MODEL CHECKS
    # ========================================================================

    async def check_models(self) -> HealthCheckResult:
        """Check model availability."""
        start = time.time()

        models_dir = PROJECT_ROOT / "models"

        if not models_dir.exists():
            return HealthCheckResult(
                component="models",
                status=HealthStatus.DEGRADED,
                message="Models directory not found",
                latency_ms=(time.time() - start) * 1000,
            )

        model_files = list(models_dir.glob("*.pkl"))

        if model_files:
            return HealthCheckResult(
                component="models",
                status=HealthStatus.HEALTHY,
                message=f"Found {len(model_files)} trained models",
                latency_ms=(time.time() - start) * 1000,
                details={
                    "model_count": len(model_files),
                    "models": [f.stem for f in model_files[:10]],
                },
            )
        else:
            return HealthCheckResult(
                component="models",
                status=HealthStatus.DEGRADED,
                message="No trained models found",
                latency_ms=(time.time() - start) * 1000,
            )

    async def check_model_staleness(self) -> HealthCheckResult:
        """
        Check model staleness.

        @mlquant P2-M2: Model staleness detection with auto-quarantine.
        """
        start = time.time()

        try:
            from quant_trading_system.models.staleness_detector import ModelStalenessDetector

            detector = ModelStalenessDetector()
            stale_models = detector.get_stale_models() if hasattr(detector, "get_stale_models") else []
            quarantined = detector.get_quarantined_models() if hasattr(detector, "get_quarantined_models") else []

            latency = (time.time() - start) * 1000

            if quarantined:
                return HealthCheckResult(
                    component="model_staleness",
                    status=HealthStatus.DEGRADED,
                    message=f"{len(quarantined)} models quarantined",
                    latency_ms=latency,
                    details={
                        "stale": stale_models,
                        "quarantined": quarantined,
                    },
                )
            elif stale_models:
                return HealthCheckResult(
                    component="model_staleness",
                    status=HealthStatus.DEGRADED,
                    message=f"{len(stale_models)} models need retraining",
                    latency_ms=latency,
                    details={"stale": stale_models},
                )
            else:
                return HealthCheckResult(
                    component="model_staleness",
                    status=HealthStatus.HEALTHY,
                    message="All models are fresh",
                    latency_ms=latency,
                )

        except ImportError:
            return HealthCheckResult(
                component="model_staleness",
                status=HealthStatus.UNKNOWN,
                message="Staleness detector not available",
                latency_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                component="model_staleness",
                status=HealthStatus.UNKNOWN,
                message=f"Staleness check failed: {e}",
                latency_ms=(time.time() - start) * 1000,
            )

    # ========================================================================
    # MONITORING CHECKS
    # ========================================================================

    async def check_metrics_collector(self) -> HealthCheckResult:
        """Check Prometheus metrics collector."""
        start = time.time()

        try:
            from quant_trading_system.monitoring.metrics import MetricsCollector

            collector = MetricsCollector()
            metrics_count = len(collector.get_all_metrics()) if hasattr(collector, "get_all_metrics") else 0

            latency = (time.time() - start) * 1000

            return HealthCheckResult(
                component="metrics_collector",
                status=HealthStatus.HEALTHY,
                message=f"Metrics collector active ({metrics_count} metrics)",
                latency_ms=latency,
                details={"metrics_count": metrics_count},
            )

        except ImportError:
            return HealthCheckResult(
                component="metrics_collector",
                status=HealthStatus.UNKNOWN,
                message="Metrics collector not available",
                latency_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                component="metrics_collector",
                status=HealthStatus.DEGRADED,
                message=f"Metrics collector error: {e}",
                latency_ms=(time.time() - start) * 1000,
            )

    async def check_audit_log(self) -> HealthCheckResult:
        """Check audit log integrity."""
        start = time.time()

        try:
            from quant_trading_system.monitoring.audit import AuditLogger

            audit_logger = AuditLogger()

            # Verify chain integrity
            if hasattr(audit_logger, "verify_chain"):
                is_valid = audit_logger.verify_chain()

                if is_valid:
                    return HealthCheckResult(
                        component="audit_log",
                        status=HealthStatus.HEALTHY,
                        message="Audit log integrity verified",
                        latency_ms=(time.time() - start) * 1000,
                    )
                else:
                    return HealthCheckResult(
                        component="audit_log",
                        status=HealthStatus.CRITICAL,
                        message="AUDIT LOG INTEGRITY COMPROMISED",
                        latency_ms=(time.time() - start) * 1000,
                    )
            else:
                return HealthCheckResult(
                    component="audit_log",
                    status=HealthStatus.HEALTHY,
                    message="Audit logger active",
                    latency_ms=(time.time() - start) * 1000,
                )

        except ImportError:
            return HealthCheckResult(
                component="audit_log",
                status=HealthStatus.UNKNOWN,
                message="Audit logger not available",
                latency_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                component="audit_log",
                status=HealthStatus.DEGRADED,
                message=f"Audit log error: {e}",
                latency_ms=(time.time() - start) * 1000,
            )

    # ========================================================================
    # MARKET CONDITION CHECKS
    # ========================================================================

    async def check_vix_regime(self) -> HealthCheckResult:
        """
        Check VIX regime status.

        @data P1-A: VIX-based regime detection for risk adjustment.
        """
        start = time.time()

        try:
            from quant_trading_system.data.vix_feed import VIXFeed
            from quant_trading_system.alpha.vix_integration import VIXRegime

            vix_feed = VIXFeed()
            current_vix = vix_feed.get_current() if hasattr(vix_feed, "get_current") else None

            latency = (time.time() - start) * 1000

            if current_vix is None:
                return HealthCheckResult(
                    component="vix_regime",
                    status=HealthStatus.UNKNOWN,
                    message="VIX data unavailable",
                    latency_ms=latency,
                )

            vix_value = float(current_vix.get("close", 20))

            # Determine regime
            if vix_value >= 35:
                regime = "CRISIS"
                status = HealthStatus.CRITICAL
            elif vix_value >= 25:
                regime = "HIGH"
                status = HealthStatus.DEGRADED
            elif vix_value >= 18:
                regime = "ELEVATED"
                status = HealthStatus.HEALTHY
            else:
                regime = "NORMAL"
                status = HealthStatus.HEALTHY

            return HealthCheckResult(
                component="vix_regime",
                status=status,
                message=f"VIX: {vix_value:.1f} ({regime})",
                latency_ms=latency,
                details={
                    "vix": vix_value,
                    "regime": regime,
                },
            )

        except ImportError:
            return HealthCheckResult(
                component="vix_regime",
                status=HealthStatus.UNKNOWN,
                message="VIX module not available",
                latency_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                component="vix_regime",
                status=HealthStatus.UNKNOWN,
                message=f"VIX check failed: {e}",
                latency_ms=(time.time() - start) * 1000,
            )

    async def check_correlation_monitor(self) -> HealthCheckResult:
        """
        Check correlation regime.

        @infra P3-A: Correlation monitoring for risk concentration.
        """
        start = time.time()

        try:
            from quant_trading_system.risk.correlation_monitor import CorrelationMonitor

            monitor = CorrelationMonitor()
            regime = monitor.get_regime() if hasattr(monitor, "get_regime") else "NORMAL"

            latency = (time.time() - start) * 1000

            if regime == "BREAKDOWN":
                status = HealthStatus.CRITICAL
            elif regime == "HIGH":
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            return HealthCheckResult(
                component="correlation_monitor",
                status=status,
                message=f"Correlation regime: {regime}",
                latency_ms=latency,
                details={"regime": regime},
            )

        except ImportError:
            return HealthCheckResult(
                component="correlation_monitor",
                status=HealthStatus.UNKNOWN,
                message="Correlation monitor not available",
                latency_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                component="correlation_monitor",
                status=HealthStatus.UNKNOWN,
                message=f"Correlation check failed: {e}",
                latency_ms=(time.time() - start) * 1000,
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
            HealthStatus.HEALTHY: "✓",
            HealthStatus.DEGRADED: "⚠",
            HealthStatus.UNHEALTHY: "✗",
            HealthStatus.CRITICAL: "🔴",
            HealthStatus.UNKNOWN: "?",
        }.get(status, "?")

        print(f"\n{status_icon} {status.value} ({len(by_status[status])} components)")
        print("-" * 40)

        for result in by_status[status]:
            latency_str = f" ({result.latency_ms:.0f}ms)" if result.latency_ms > 0 else ""
            print(f"  {result.component}: {result.message}{latency_str}")

    # Overall status
    overall = checker.get_overall_status()
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
        return [
            await checker.check_broker(),
            await checker.check_kill_switch(),
            await checker.check_risk_system(),
            await checker.check_data_feed(),
        ]

    results = asyncio.run(quick_checks())

    print("\nQuick Status:")
    print("-" * 40)
    for result in results:
        status_icon = "✓" if result.status == HealthStatus.HEALTHY else "✗"
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
            await checker.check_vix_regime(),
            await checker.check_correlation_monitor(),
        ]

    results = asyncio.run(risk_checks())

    print("\nRisk Components:")
    print("-" * 60)
    for result in results:
        status_icon = {
            HealthStatus.HEALTHY: "✓",
            HealthStatus.DEGRADED: "⚠",
            HealthStatus.CRITICAL: "🔴",
        }.get(result.status, "?")

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

    try:
        from quant_trading_system.monitoring.audit import AuditLogger

        audit_logger = AuditLogger()

        if hasattr(audit_logger, "verify_chain"):
            is_valid = audit_logger.verify_chain()

            if is_valid:
                logger.info("✓ Audit log integrity VERIFIED")
                logger.info("  All entries have valid hash chains")
                return 0
            else:
                logger.error("✗ AUDIT LOG INTEGRITY COMPROMISED")
                logger.error("  Hash chain verification failed!")
                return 1
        else:
            logger.info("Audit log verification not available")
            return 0

    except ImportError:
        logger.warning("Audit logger module not available")
        return 0

    except Exception as e:
        logger.error(f"Audit verification failed: {e}")
        return 1


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
