"""
Multi-Region Deployment Configuration.

P3-F Enhancement: Infrastructure for deploying across multiple geographic regions:
- Regional latency optimization
- Disaster recovery and failover
- Exchange proximity settings
- Cross-region data replication

Expected Impact: +5-10 bps from reduced latency and improved reliability.

Author: AlphaTrade System
Version: 1.0.0
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RegionType(str, Enum):
    """Region deployment type."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    DR = "disaster_recovery"
    EDGE = "edge"


class ExchangeProximity(str, Enum):
    """Proximity to major exchanges."""

    HIGH = "high"  # < 10ms latency (colocation)
    MEDIUM = "medium"  # 10-50ms latency
    LOW = "low"  # > 50ms latency


@dataclass
class ExchangeInfo:
    """Information about a stock exchange."""

    name: str
    code: str
    region: str
    optimal_latency_ms: float
    colocation_available: bool = True


# Major US exchanges
EXCHANGES = {
    "NYSE": ExchangeInfo("New York Stock Exchange", "NYSE", "us-east-1", 5.0),
    "NASDAQ": ExchangeInfo("NASDAQ", "NASDAQ", "us-east-1", 5.0),
    "BATS": ExchangeInfo("BATS Exchange", "BATS", "us-east-1", 5.0),
    "IEX": ExchangeInfo("Investors Exchange", "IEX", "us-east-1", 10.0),
    "CME": ExchangeInfo("Chicago Mercantile Exchange", "CME", "us-central-1", 5.0),
    "CBOE": ExchangeInfo("Chicago Board Options Exchange", "CBOE", "us-central-1", 5.0),
}


@dataclass
class RegionConfig:
    """Configuration for a deployment region."""

    region_id: str
    region_name: str
    region_type: RegionType
    exchange_proximity: ExchangeProximity
    target_latency_ms: float
    enable_colocation: bool = False
    data_center: str = ""

    # Networking
    api_endpoint: str = ""
    market_data_endpoint: str = ""

    # Failover
    primary_region: str | None = None
    failover_priority: int = 0
    enable_failover: bool = False

    # Features
    enable_microstructure: bool = True
    order_routing: str = "smart"

    # Resource limits
    max_cpu: int = 4
    max_memory_gb: int = 8

    # Replication
    enable_replication: bool = True
    replication_role: str = "primary"

    def to_env_dict(self) -> dict[str, str]:
        """Convert to environment variable dictionary."""
        return {
            "REGION": self.region_id,
            "REGION_NAME": self.region_name,
            "REGION_TYPE": self.region_type.value,
            "EXCHANGE_PROXIMITY": self.exchange_proximity.value,
            "TARGET_LATENCY_MS": str(self.target_latency_ms),
            "ENABLE_COLOCATION": str(self.enable_colocation).lower(),
            "DATA_CENTER": self.data_center,
            "ENABLE_MARKET_MICROSTRUCTURE": str(self.enable_microstructure).lower(),
            "ORDER_ROUTING": self.order_routing,
            "ENABLE_FAILOVER_MODE": str(self.enable_failover).lower(),
            "PRIMARY_REGION": self.primary_region or "",
            "FAILOVER_PRIORITY": str(self.failover_priority),
        }


# Pre-defined region configurations
REGION_CONFIGS: dict[str, RegionConfig] = {
    "us-east-1": RegionConfig(
        region_id="us-east-1",
        region_name="US East (N. Virginia)",
        region_type=RegionType.PRIMARY,
        exchange_proximity=ExchangeProximity.HIGH,
        target_latency_ms=5.0,
        enable_colocation=True,
        data_center="equinix-ny5",
        api_endpoint="https://paper-api.alpaca.markets",
        market_data_endpoint="wss://stream.data.alpaca.markets/v2/iex",
        enable_microstructure=True,
        order_routing="smart",
        max_cpu=4,
        max_memory_gb=8,
        replication_role="primary",
    ),
    "us-west-2": RegionConfig(
        region_id="us-west-2",
        region_name="US West (Oregon)",
        region_type=RegionType.SECONDARY,
        exchange_proximity=ExchangeProximity.MEDIUM,
        target_latency_ms=30.0,
        enable_colocation=False,
        data_center="aws-us-west-2",
        api_endpoint="https://paper-api.alpaca.markets",
        market_data_endpoint="wss://stream.data.alpaca.markets/v2/iex",
        primary_region="us-east-1",
        failover_priority=1,
        enable_failover=True,
        enable_microstructure=False,
        order_routing="standard",
        max_cpu=2,
        max_memory_gb=4,
        replication_role="replica",
    ),
    "eu-west-1": RegionConfig(
        region_id="eu-west-1",
        region_name="EU West (Ireland)",
        region_type=RegionType.SECONDARY,
        exchange_proximity=ExchangeProximity.LOW,
        target_latency_ms=80.0,
        enable_colocation=False,
        data_center="aws-eu-west-1",
        api_endpoint="https://paper-api.alpaca.markets",
        market_data_endpoint="wss://stream.data.alpaca.markets/v2/iex",
        primary_region="us-east-1",
        failover_priority=2,
        enable_failover=True,
        enable_microstructure=False,
        order_routing="standard",
        max_cpu=2,
        max_memory_gb=4,
        replication_role="replica",
    ),
    "ap-northeast-1": RegionConfig(
        region_id="ap-northeast-1",
        region_name="Asia Pacific (Tokyo)",
        region_type=RegionType.EDGE,
        exchange_proximity=ExchangeProximity.LOW,
        target_latency_ms=150.0,
        enable_colocation=False,
        data_center="aws-ap-northeast-1",
        api_endpoint="https://paper-api.alpaca.markets",
        market_data_endpoint="wss://stream.data.alpaca.markets/v2/iex",
        primary_region="us-west-2",
        failover_priority=3,
        enable_failover=True,
        enable_microstructure=False,
        order_routing="standard",
        max_cpu=2,
        max_memory_gb=4,
        replication_role="replica",
    ),
}


class RegionalSettings(BaseModel):
    """Regional deployment settings model."""

    current_region: str = Field(default="us-east-1")
    enable_multi_region: bool = Field(default=False)
    active_regions: list[str] = Field(default_factory=lambda: ["us-east-1"])
    cross_region_replication: bool = Field(default=False)
    latency_threshold_ms: float = Field(default=50.0)
    auto_failover: bool = Field(default=True)
    failover_timeout_seconds: int = Field(default=30)

    def get_current_config(self) -> RegionConfig:
        """Get configuration for current region."""
        return get_region_config(self.current_region)

    def get_primary_region(self) -> str:
        """Get the primary region ID."""
        for region_id, config in REGION_CONFIGS.items():
            if config.region_type == RegionType.PRIMARY:
                return region_id
        return "us-east-1"


def get_region_config(region_id: str | None = None) -> RegionConfig:
    """Get configuration for a region.

    Args:
        region_id: Region identifier. If None, uses REGION env var or default.

    Returns:
        RegionConfig for the specified region.
    """
    if region_id is None:
        region_id = os.environ.get("REGION", "us-east-1")

    if region_id in REGION_CONFIGS:
        return REGION_CONFIGS[region_id]

    logger.warning(f"Unknown region {region_id}, using default us-east-1")
    return REGION_CONFIGS["us-east-1"]


def get_optimal_region_for_exchange(exchange_code: str) -> str:
    """Get the optimal region for trading on a specific exchange.

    Args:
        exchange_code: Exchange code (e.g., NYSE, NASDAQ)

    Returns:
        Region ID with lowest latency to the exchange.
    """
    exchange = EXCHANGES.get(exchange_code)
    if exchange:
        return exchange.region
    return "us-east-1"  # Default to primary


def calculate_latency_score(region_id: str, target_exchanges: list[str]) -> float:
    """Calculate a latency score for a region based on target exchanges.

    Lower score is better.

    Args:
        region_id: Region identifier
        target_exchanges: List of exchange codes to trade on

    Returns:
        Weighted latency score
    """
    config = get_region_config(region_id)

    total_latency = 0.0
    for exchange_code in target_exchanges:
        exchange = EXCHANGES.get(exchange_code)
        if exchange:
            if exchange.region == region_id:
                # Same region - use target latency
                total_latency += config.target_latency_ms
            else:
                # Different region - add cross-region latency (estimated)
                total_latency += config.target_latency_ms + 50.0  # Base cross-region penalty

    return total_latency / len(target_exchanges) if target_exchanges else config.target_latency_ms


def select_optimal_region(
    target_exchanges: list[str],
    available_regions: list[str] | None = None,
) -> str:
    """Select the optimal region for trading given target exchanges.

    Args:
        target_exchanges: List of exchange codes to trade on
        available_regions: List of available region IDs (default: all)

    Returns:
        Optimal region ID
    """
    available_regions = available_regions or list(REGION_CONFIGS.keys())

    best_region = available_regions[0]
    best_score = float("inf")

    for region_id in available_regions:
        score = calculate_latency_score(region_id, target_exchanges)
        if score < best_score:
            best_score = score
            best_region = region_id

    logger.info(f"Selected optimal region {best_region} with latency score {best_score:.2f}ms")
    return best_region


class RegionalHealthMonitor:
    """Monitor health and latency of regional deployments."""

    def __init__(self, settings: RegionalSettings | None = None):
        """Initialize monitor.

        Args:
            settings: Regional settings
        """
        self.settings = settings or RegionalSettings()
        self._latencies: dict[str, list[float]] = {}
        self._health_status: dict[str, bool] = {}

    def record_latency(self, region_id: str, latency_ms: float) -> None:
        """Record a latency measurement for a region.

        Args:
            region_id: Region identifier
            latency_ms: Measured latency in milliseconds
        """
        if region_id not in self._latencies:
            self._latencies[region_id] = []

        self._latencies[region_id].append(latency_ms)

        # Keep last 100 measurements
        if len(self._latencies[region_id]) > 100:
            self._latencies[region_id] = self._latencies[region_id][-100:]

    def get_average_latency(self, region_id: str) -> float | None:
        """Get average latency for a region.

        Args:
            region_id: Region identifier

        Returns:
            Average latency in ms or None if no data
        """
        latencies = self._latencies.get(region_id)
        if latencies:
            return sum(latencies) / len(latencies)
        return None

    def set_health_status(self, region_id: str, healthy: bool) -> None:
        """Set health status for a region.

        Args:
            region_id: Region identifier
            healthy: Whether the region is healthy
        """
        self._health_status[region_id] = healthy

    def is_healthy(self, region_id: str) -> bool:
        """Check if a region is healthy.

        Args:
            region_id: Region identifier

        Returns:
            True if healthy (or unknown), False if unhealthy
        """
        return self._health_status.get(region_id, True)

    def should_failover(self, current_region: str) -> str | None:
        """Check if failover is needed and return target region.

        Args:
            current_region: Current active region

        Returns:
            Target region for failover, or None if no failover needed
        """
        if not self.settings.auto_failover:
            return None

        # Check current region health
        if self.is_healthy(current_region):
            # Check latency threshold
            avg_latency = self.get_average_latency(current_region)
            if avg_latency is None or avg_latency < self.settings.latency_threshold_ms * 2:
                return None

        # Find best healthy alternative
        config = get_region_config(current_region)
        if config.primary_region:
            # Try primary first
            if self.is_healthy(config.primary_region):
                return config.primary_region

        # Find lowest priority healthy region
        candidates = [
            (rid, rc)
            for rid, rc in REGION_CONFIGS.items()
            if rid != current_region and self.is_healthy(rid)
        ]

        if candidates:
            # Sort by failover priority
            candidates.sort(key=lambda x: x[1].failover_priority)
            return candidates[0][0]

        return None

    def get_status_report(self) -> dict[str, Any]:
        """Get a status report for all regions.

        Returns:
            Dictionary with region status information
        """
        report = {}
        for region_id in REGION_CONFIGS:
            report[region_id] = {
                "healthy": self.is_healthy(region_id),
                "average_latency_ms": self.get_average_latency(region_id),
                "config": REGION_CONFIGS[region_id].region_type.value,
            }
        return report


# Singleton instance
_regional_settings: RegionalSettings | None = None
_health_monitor: RegionalHealthMonitor | None = None


def get_regional_settings() -> RegionalSettings:
    """Get or create regional settings singleton."""
    global _regional_settings
    if _regional_settings is None:
        _regional_settings = RegionalSettings()
    return _regional_settings


def get_health_monitor() -> RegionalHealthMonitor:
    """Get or create health monitor singleton."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = RegionalHealthMonitor(get_regional_settings())
    return _health_monitor
