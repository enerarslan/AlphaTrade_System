"""Unit tests for monitoring/health.py."""

import asyncio

import pytest

from quant_trading_system.monitoring.health import HealthStatus, SystemHealthChecker


class TestSystemHealthChecker:
    """Tests for bounded health-check execution."""

    @pytest.mark.asyncio
    async def test_run_check_with_timeout_returns_degraded_result(self):
        """Slow health checks should degrade instead of hanging forever."""
        checker = SystemHealthChecker()

        async def slow_check():
            await asyncio.sleep(0.05)

        result = await checker._run_check_with_timeout(
            "broker",
            slow_check(),
            timeout_seconds=0.01,
        )

        assert result.component == "broker"
        assert result.status == HealthStatus.DEGRADED
        assert "timed out" in result.message
