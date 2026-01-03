"""
Lazy Module Loading for Performance Optimization.

Implements lazy loading to defer importing heavy modules until first use.
This reduces application startup time and memory footprint when not all
functionality is needed.

Features:
- Lazy module proxies that import on first access
- Configurable import groups (core, ml, backtesting, etc.)
- Import timing metrics for optimization
- Thread-safe lazy loading
- Preload API for performance-critical paths

Usage:
    from quant_trading_system.core.lazy_loader import lazy_import, LazyModule

    # Lazy import a module
    pd = lazy_import("pandas")

    # Later, actual import happens on first attribute access
    df = pd.DataFrame()  # pandas imported here

    # Or use LazyModule for attribute-based access
    modules = LazyModule("quant_trading_system.models")
    model = modules.deep_learning.LSTMModel()  # Imported on access

Example for package __init__:
    from quant_trading_system.core.lazy_loader import LazyModuleLoader

    loader = LazyModuleLoader(__name__)
    loader.register_lazy("models.deep_learning", ["LSTMModel", "TransformerModel"])
    loader.register_lazy("models.classical_ml", ["XGBoostModel"])

    # Exports are lazy
    LSTMModel = loader.get_lazy_attr("LSTMModel")
"""

from __future__ import annotations

import importlib
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ImportMetrics:
    """Metrics for module import times.

    Attributes:
        module_name: Fully qualified module name.
        import_time_ms: Time taken to import in milliseconds.
        imported_at: Unix timestamp of import.
        size_estimate_kb: Rough estimate of module memory size.
    """

    module_name: str
    import_time_ms: float
    imported_at: float
    size_estimate_kb: float = 0.0


class ImportMetricsCollector:
    """Collects and reports import timing metrics.

    Thread-safe singleton for tracking lazy import performance.
    """

    _instance: ImportMetricsCollector | None = None
    _lock = threading.Lock()

    def __new__(cls) -> ImportMetricsCollector:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._metrics: list[ImportMetrics] = []
        self._metrics_lock = threading.Lock()

    def record(self, module_name: str, import_time_ms: float) -> None:
        """Record an import timing."""
        with self._metrics_lock:
            self._metrics.append(
                ImportMetrics(
                    module_name=module_name,
                    import_time_ms=import_time_ms,
                    imported_at=time.time(),
                )
            )

    def get_metrics(self) -> list[ImportMetrics]:
        """Get all recorded metrics."""
        with self._metrics_lock:
            return list(self._metrics)

    def get_slow_imports(self, threshold_ms: float = 100.0) -> list[ImportMetrics]:
        """Get imports slower than threshold."""
        with self._metrics_lock:
            return [m for m in self._metrics if m.import_time_ms > threshold_ms]

    def get_total_import_time(self) -> float:
        """Get total time spent importing."""
        with self._metrics_lock:
            return sum(m.import_time_ms for m in self._metrics)

    def report(self) -> dict[str, Any]:
        """Generate import report."""
        metrics = self.get_metrics()
        return {
            "total_imports": len(metrics),
            "total_time_ms": round(self.get_total_import_time(), 2),
            "avg_time_ms": round(
                self.get_total_import_time() / len(metrics), 2
            ) if metrics else 0,
            "slowest_imports": [
                {"module": m.module_name, "time_ms": round(m.import_time_ms, 2)}
                for m in sorted(metrics, key=lambda x: x.import_time_ms, reverse=True)[:5]
            ],
        }

    @classmethod
    def reset(cls) -> None:
        """Reset metrics (for testing)."""
        if cls._instance:
            with cls._instance._metrics_lock:
                cls._instance._metrics.clear()


def get_import_metrics() -> ImportMetricsCollector:
    """Get the global import metrics collector."""
    return ImportMetricsCollector()


class LazyModuleProxy:
    """
    Proxy object that imports a module on first attribute access.

    This is the core lazy loading mechanism. The actual module is not
    imported until an attribute is accessed on the proxy.

    Attributes:
        _module_name: Name of module to import.
        _module: Cached module reference (None until imported).
        _lock: Thread lock for safe importing.
    """

    __slots__ = ("_module_name", "_module", "_lock", "_importing")

    def __init__(self, module_name: str) -> None:
        """
        Initialize lazy proxy.

        Args:
            module_name: Fully qualified module name to import.
        """
        object.__setattr__(self, "_module_name", module_name)
        object.__setattr__(self, "_module", None)
        object.__setattr__(self, "_lock", threading.Lock())
        object.__setattr__(self, "_importing", False)

    def _load_module(self) -> ModuleType:
        """Load the module if not already loaded."""
        module = object.__getattribute__(self, "_module")
        if module is not None:
            return module

        lock = object.__getattribute__(self, "_lock")
        with lock:
            # Double-check after acquiring lock
            module = object.__getattribute__(self, "_module")
            if module is not None:
                return module

            module_name = object.__getattribute__(self, "_module_name")

            # Track import time
            start = time.time()
            try:
                module = importlib.import_module(module_name)
            except ImportError as e:
                logger.error(f"Failed to lazy-import {module_name}: {e}")
                raise

            elapsed_ms = (time.time() - start) * 1000

            # Record metrics
            get_import_metrics().record(module_name, elapsed_ms)

            if elapsed_ms > 100:
                logger.debug(f"Lazy import '{module_name}' took {elapsed_ms:.1f}ms")

            object.__setattr__(self, "_module", module)
            return module

    def __getattr__(self, name: str) -> Any:
        """Get attribute from the lazy-loaded module."""
        module = self._load_module()
        return getattr(module, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute on the lazy-loaded module."""
        module = self._load_module()
        setattr(module, name, value)

    def __dir__(self) -> list[str]:
        """List module attributes."""
        module = self._load_module()
        return dir(module)

    def __repr__(self) -> str:
        module_name = object.__getattribute__(self, "_module_name")
        module = object.__getattribute__(self, "_module")
        status = "loaded" if module else "lazy"
        return f"<LazyModuleProxy({module_name!r}, {status})>"


def lazy_import(module_name: str) -> LazyModuleProxy:
    """
    Lazily import a module.

    The module is not actually imported until an attribute is accessed.

    Args:
        module_name: Fully qualified module name.

    Returns:
        Lazy proxy that imports on first access.

    Example:
        pd = lazy_import("pandas")
        # pandas not imported yet

        df = pd.DataFrame()  # pandas imported here
    """
    return LazyModuleProxy(module_name)


class LazyAttributeProxy:
    """
    Proxy for a specific attribute from a lazy module.

    Used to expose specific classes/functions from lazy modules
    while keeping the module itself lazy.
    """

    __slots__ = ("_module_proxy", "_attr_name", "_cached")

    def __init__(self, module_proxy: LazyModuleProxy, attr_name: str) -> None:
        object.__setattr__(self, "_module_proxy", module_proxy)
        object.__setattr__(self, "_attr_name", attr_name)
        object.__setattr__(self, "_cached", None)

    def _get_attr(self) -> Any:
        """Get the actual attribute."""
        cached = object.__getattribute__(self, "_cached")
        if cached is not None:
            return cached

        module_proxy = object.__getattribute__(self, "_module_proxy")
        attr_name = object.__getattribute__(self, "_attr_name")
        attr = getattr(module_proxy, attr_name)
        object.__setattr__(self, "_cached", attr)
        return attr

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_attr(), name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._get_attr()(*args, **kwargs)

    def __repr__(self) -> str:
        attr_name = object.__getattribute__(self, "_attr_name")
        return f"<LazyAttributeProxy({attr_name!r})>"


class LazyModuleLoader:
    """
    Package-level lazy loading coordinator.

    Helps packages define lazy exports that are only imported when
    accessed. Useful for __init__.py files to reduce import time.

    Example:
        # In package __init__.py
        loader = LazyModuleLoader(__name__)

        # Register lazy submodules
        loader.register_lazy(".models.deep_learning", ["LSTMModel"])
        loader.register_lazy(".models.classical_ml", ["XGBoostModel"])

        # Get lazy attribute (returns proxy)
        LSTMModel = loader.get_lazy_attr("LSTMModel")
    """

    def __init__(self, package_name: str) -> None:
        """
        Initialize loader for a package.

        Args:
            package_name: The package __name__.
        """
        self._package_name = package_name
        self._registry: dict[str, tuple[str, str]] = {}  # attr -> (module, attr)
        self._module_proxies: dict[str, LazyModuleProxy] = {}
        self._lock = threading.Lock()

    def register_lazy(
        self,
        module_path: str,
        attributes: list[str],
    ) -> None:
        """
        Register attributes to be lazily imported.

        Args:
            module_path: Relative or absolute module path.
            attributes: List of attribute names from that module.
        """
        # Make absolute if relative
        if module_path.startswith("."):
            full_path = f"{self._package_name}{module_path}"
        else:
            full_path = module_path

        with self._lock:
            for attr in attributes:
                self._registry[attr] = (full_path, attr)

    def get_lazy_attr(self, name: str) -> LazyAttributeProxy:
        """
        Get a lazy attribute proxy.

        Args:
            name: Attribute name.

        Returns:
            Lazy proxy for the attribute.

        Raises:
            KeyError: If attribute not registered.
        """
        with self._lock:
            if name not in self._registry:
                raise KeyError(f"Attribute '{name}' not registered for lazy loading")

            module_path, attr_name = self._registry[name]

            # Get or create module proxy
            if module_path not in self._module_proxies:
                self._module_proxies[module_path] = LazyModuleProxy(module_path)

            return LazyAttributeProxy(self._module_proxies[module_path], attr_name)

    def preload(self, *names: str) -> None:
        """
        Preload specified attributes (force import).

        Args:
            names: Attribute names to preload.
        """
        for name in names:
            if name in self._registry:
                proxy = self.get_lazy_attr(name)
                # Force load by accessing
                _ = proxy._get_attr()

    def preload_all(self) -> None:
        """Preload all registered attributes."""
        self.preload(*self._registry.keys())

    def get_registered_attrs(self) -> list[str]:
        """Get list of registered attribute names."""
        return list(self._registry.keys())


# =============================================================================
# MODULE GROUPS FOR SELECTIVE LOADING
# =============================================================================


class ModuleGroup:
    """
    Group of related modules for selective loading.

    Allows defining groups of modules that can be loaded together
    based on use case (e.g., 'ml', 'backtesting', 'live_trading').
    """

    def __init__(self, name: str, modules: list[str]) -> None:
        """
        Create module group.

        Args:
            name: Group identifier.
            modules: List of module paths in the group.
        """
        self.name = name
        self.modules = modules
        self._loaded = False

    def load(self) -> dict[str, ModuleType]:
        """
        Load all modules in the group.

        Returns:
            Dict mapping module names to loaded modules.
        """
        if self._loaded:
            return {}

        loaded = {}
        for module_path in self.modules:
            try:
                start = time.time()
                module = importlib.import_module(module_path)
                elapsed_ms = (time.time() - start) * 1000
                get_import_metrics().record(module_path, elapsed_ms)
                loaded[module_path] = module
            except ImportError as e:
                logger.warning(f"Failed to load {module_path} in group '{self.name}': {e}")

        self._loaded = True
        logger.info(f"Loaded module group '{self.name}': {len(loaded)} modules")
        return loaded


# Pre-defined module groups for AlphaTrade
CORE_MODULES = ModuleGroup(
    "core",
    [
        "quant_trading_system.core.data_types",
        "quant_trading_system.core.events",
        "quant_trading_system.core.exceptions",
    ],
)

ML_MODULES = ModuleGroup(
    "ml",
    [
        "quant_trading_system.models.classical_ml",
        "quant_trading_system.models.deep_learning",
        "quant_trading_system.models.ensemble",
    ],
)

TRADING_MODULES = ModuleGroup(
    "trading",
    [
        "quant_trading_system.trading.trading_engine",
        "quant_trading_system.execution.order_manager",
        "quant_trading_system.execution.alpaca_client",
        "quant_trading_system.risk.limits",
    ],
)

BACKTESTING_MODULES = ModuleGroup(
    "backtesting",
    [
        "quant_trading_system.backtest.engine",
        "quant_trading_system.backtest.simulator",
        "quant_trading_system.backtest.analyzer",
    ],
)

DATA_MODULES = ModuleGroup(
    "data",
    [
        "quant_trading_system.data.loader",
        "quant_trading_system.data.preprocessor",
        "quant_trading_system.features.technical",
    ],
)


def preload_modules(*group_names: str) -> None:
    """
    Preload module groups by name.

    Args:
        group_names: Names of groups to load (core, ml, trading, etc.)

    Example:
        preload_modules("core", "trading")  # Load core + trading modules
    """
    groups = {
        "core": CORE_MODULES,
        "ml": ML_MODULES,
        "trading": TRADING_MODULES,
        "backtesting": BACKTESTING_MODULES,
        "data": DATA_MODULES,
    }

    for name in group_names:
        if name in groups:
            groups[name].load()
        else:
            logger.warning(f"Unknown module group: {name}")


# =============================================================================
# CONVENIENT LAZY IMPORTS FOR HEAVY MODULES
# =============================================================================


# These are created at module level but don't import until used
numpy = lazy_import("numpy")
pandas = lazy_import("pandas")
polars = lazy_import("polars")
torch = lazy_import("torch")
sklearn = lazy_import("sklearn")
xgboost = lazy_import("xgboost")
lightgbm = lazy_import("lightgbm")
