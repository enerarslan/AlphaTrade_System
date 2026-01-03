"""
Configuration module for the trading system.

Provides centralized configuration management using Pydantic settings
and YAML-based configuration files.
"""

from .settings import Settings, get_settings
from .regional import (
    ExchangeProximity,
    RegionConfig,
    RegionalHealthMonitor,
    RegionalSettings,
    RegionType,
    EXCHANGES,
    REGION_CONFIGS,
    calculate_latency_score,
    get_health_monitor,
    get_optimal_region_for_exchange,
    get_region_config,
    get_regional_settings,
    select_optimal_region,
)

__all__ = [
    "Settings",
    "get_settings",
    # Regional Configuration (P3-F Enhancement)
    "RegionType",
    "ExchangeProximity",
    "RegionConfig",
    "RegionalSettings",
    "RegionalHealthMonitor",
    "EXCHANGES",
    "REGION_CONFIGS",
    "get_region_config",
    "get_regional_settings",
    "get_health_monitor",
    "get_optimal_region_for_exchange",
    "calculate_latency_score",
    "select_optimal_region",
]
