"""
Model package.

This module intentionally avoids eager submodule imports so classical-ML,
backtest, and lineage code can run without optional deep-learning packages.
Import concrete model classes from their submodules instead.
"""

__all__: list[str] = []
