"""
Backtest package.

This module intentionally avoids eager imports so engine-only consumers do not
pull in optimizer or training dependencies during package initialization.
Import concrete helpers from their submodules instead.
"""

__all__: list[str] = []
