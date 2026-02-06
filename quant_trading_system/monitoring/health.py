"""
System health monitoring and diagnostics module.

This module provides the core SystemHealthChecker class used by both the
health check script and the dashboard to monitor system components,
including:
- Infrastructure (Database, Redis, GPU)
- Trading Components (Broker, Data Feed)
- Safety Systems (Kill Switch, Circuit Breakers)
- Models & Risk
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from quant_trading_system.config.settings import get_settings

logger = logging.getLogger(__name__)

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
    INFRASTRUCTURE = "infrastructure"


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

    Performs health checks on all system components including:
    - Infrastructure (DB, Redis, GPU)
    - Trading Systems (Alpaca, Data)
    - Safety (Kill Switch, Risk)
    """

    def __init__(self):
        self.logger = logging.getLogger("SystemHealthChecker")
        self.settings = get_settings()
        # Resolve project root assuming this file is in quant_trading_system/monitoring/
        self.project_root = Path(__file__).parent.parent.parent

    async def run_all_checks(self) -> list[HealthCheckResult]:
        """Run all health checks."""
        self.logger.info("Running comprehensive health checks...")
        results = []

        # Core infrastructure
        checks = [
            self.check_database(),
            self.check_redis(),
            self.check_gpu(),  # New GPU check
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

        # Run all checks
        for check in checks:
            try:
                result = await check
                results.append(result)
            except Exception as e:
                self.logger.error(f"Health check failed unexpectedly: {e}", exc_info=True)
                results.append(HealthCheckResult(
                    component="unknown",
                    status=HealthStatus.UNKNOWN,
                    message=f"Check failed: {e}",
                ))

        return results

    def get_overall_status(self, results: list[HealthCheckResult]) -> HealthStatus:
        """Get overall system health status from results."""
        if not results:
            return HealthStatus.UNKNOWN

        statuses = [r.status for r in results]

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
            from sqlalchemy import text
            from quant_trading_system.database.connection import get_db_manager

            db_manager = get_db_manager()
            with db_manager.session() as session:
                session.execute(text("SELECT 1"))

            latency = (time.time() - start) * 1000

            return HealthCheckResult(
                component="database",
                status=HealthStatus.HEALTHY,
                message="PostgreSQL connection OK",
                latency_ms=latency,
                details={"type": "postgresql"},
            )

        except ImportError as e:
            return HealthCheckResult(
                component="database",
                status=HealthStatus.UNKNOWN,
                message=f"Database module import error: {e}",
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
            import redis.asyncio as redis

            client = redis.Redis(
                host=self.settings.redis.host,
                port=self.settings.redis.port,
                db=self.settings.redis.db,
                socket_timeout=5,
            )

            await client.ping()
            info = await client.info()
            await client.close()

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

    async def check_gpu(self) -> HealthCheckResult:
        """Check GPU availability and status."""
        start = time.time()

        try:
            import torch

            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                
                # Get memory stats for the first GPU
                mem_alloc = torch.cuda.memory_allocated(0) / 1024**3  # GB
                mem_reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3 # GB

                latency = (time.time() - start) * 1000

                return HealthCheckResult(
                    component="gpu",
                    status=HealthStatus.HEALTHY,
                    message=f"GPU Active: {device_name}",
                    latency_ms=latency,
                    details={
                        "available": True,
                        "count": device_count,
                        "name": device_name,
                        "memory_allocated_gb": round(mem_alloc, 2),
                        "memory_reserved_gb": round(mem_reserved, 2),
                        "memory_total_gb": round(mem_total, 2),
                        "cuda_version": torch.version.cuda,
                    },
                )
            else:
                latency = (time.time() - start) * 1000
                return HealthCheckResult(
                    component="gpu",
                    status=HealthStatus.DEGRADED, # Degraded because we WANT GPU but can run on CPU
                    message="GPU Not Available (Running on CPU)",
                    latency_ms=latency,
                    details={"available": False},
                )

        except ImportError:
            return HealthCheckResult(
                component="gpu",
                status=HealthStatus.UNKNOWN,
                message="PyTorch not installed",
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
             return HealthCheckResult(
                component="gpu",
                status=HealthStatus.UNKNOWN,
                message=f"GPU check failed: {e}",
                latency_ms=(time.time() - start) * 1000,
            )


    async def check_broker(self) -> HealthCheckResult:
        """Check Alpaca broker connectivity."""
        start = time.time()

        try:
            from quant_trading_system.execution.alpaca_client import AlpacaClient

            # Assuming AlpacaClient handles its own auth via settings
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

            data_dir = self.project_root / "data" / "raw"
            loader = DataLoader(data_dir=data_dir, use_database=True)
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
            data_dir = self.project_root / "data" / "raw"
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
        """Check kill switch status."""
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
        """Check circuit breaker status."""
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
        """Check drawdown monitoring status."""
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

        models_dir = self.project_root / "models"

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
        """Check model staleness."""
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
            from quant_trading_system.monitoring.audit import AuditLogger, FileAuditStorage

            audit_dir = self.project_root / "logs" / "audit"
            audit_dir.mkdir(parents=True, exist_ok=True)
            storage = FileAuditStorage(storage_dir=audit_dir)
            audit_logger = AuditLogger(storage=storage)

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
                message="Audit logger import failed",
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return HealthCheckResult(
                component="audit_log",
                status=HealthStatus.UNKNOWN,
                message=f"Audit check failed: {e}",
                latency_ms=(time.time() - start) * 1000,
            )
