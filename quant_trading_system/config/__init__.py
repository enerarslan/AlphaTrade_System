"""
Configuration module for the trading system.

Provides centralized configuration management using Pydantic settings
and YAML-based configuration files.
"""

from .settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
