"""
Component registration and discovery system.

Provides a centralized registry for:
- Dynamic component registration
- Dependency injection support
- Lazy initialization
- Singleton management
- Hot-reload capability
"""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ComponentType(str, Enum):
    """Types of components that can be registered."""

    MODEL = "model"
    ALPHA = "alpha"
    FEATURE = "feature"
    RISK_MANAGER = "risk_manager"
    EXECUTION_ALGO = "execution_algo"
    DATA_LOADER = "data_loader"
    STRATEGY = "strategy"
    INDICATOR = "indicator"


class ComponentInfo(Generic[T]):
    """Information about a registered component.

    Attributes:
        name: Unique component name.
        component_type: Type of the component.
        factory: Factory function or class to create the component.
        instance: Cached instance if singleton or already created.
        singleton: Whether to cache a single instance.
        metadata: Additional component metadata.
    """

    def __init__(
        self,
        name: str,
        component_type: ComponentType,
        factory: Callable[..., T],
        singleton: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.component_type = component_type
        self.factory = factory
        self.singleton = singleton
        self.metadata = metadata or {}
        self._instance: T | None = None
        self._lock = threading.Lock()

    def get_instance(self, **kwargs: Any) -> T:
        """Get or create component instance.

        Args:
            **kwargs: Arguments to pass to factory if creating new instance.

        Returns:
            Component instance.
        """
        if self.singleton:
            if self._instance is None:
                with self._lock:
                    if self._instance is None:
                        self._instance = self.factory(**kwargs)
                        logger.debug(f"Created singleton instance: {self.name}")
            return self._instance
        return self.factory(**kwargs)

    def reset_instance(self) -> None:
        """Reset the cached instance (for hot-reload)."""
        with self._lock:
            self._instance = None
            logger.debug(f"Reset instance: {self.name}")


class ComponentRegistry:
    """Central registry for all system components.

    Provides registration, discovery, and lifecycle management for
    models, alphas, features, risk managers, and other components.

    Thread-safe implementation supporting concurrent access.
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._components: dict[ComponentType, dict[str, ComponentInfo]] = {
            component_type: {} for component_type in ComponentType
        }
        self._lock = threading.RLock()
        self._aliases: dict[str, tuple[ComponentType, str]] = {}

    def register(
        self,
        name: str,
        component_type: ComponentType,
        factory: Callable[..., T],
        singleton: bool = True,
        metadata: dict[str, Any] | None = None,
        aliases: list[str] | None = None,
    ) -> None:
        """Register a component with the registry.

        Args:
            name: Unique name for the component.
            component_type: Type of component being registered.
            factory: Factory function or class to create instances.
            singleton: Whether to maintain a single cached instance.
            metadata: Optional metadata about the component.
            aliases: Optional list of alias names for the component.

        Raises:
            ConfigurationError: If component with same name already exists.
        """
        with self._lock:
            type_registry = self._components[component_type]
            if name in type_registry:
                raise ConfigurationError(
                    f"Component '{name}' already registered as {component_type.value}",
                    details={"component_name": name, "component_type": component_type.value},
                )

            info = ComponentInfo(
                name=name,
                component_type=component_type,
                factory=factory,
                singleton=singleton,
                metadata=metadata,
            )
            type_registry[name] = info

            # Register aliases
            if aliases:
                for alias in aliases:
                    if alias in self._aliases:
                        logger.warning(f"Alias '{alias}' already exists, overwriting")
                    self._aliases[alias] = (component_type, name)

            logger.info(f"Registered {component_type.value}: {name}")

    def unregister(self, name: str, component_type: ComponentType) -> bool:
        """Unregister a component from the registry.

        Args:
            name: Name of the component to unregister.
            component_type: Type of the component.

        Returns:
            True if component was found and removed, False otherwise.
        """
        with self._lock:
            type_registry = self._components[component_type]
            if name in type_registry:
                del type_registry[name]
                # Remove any aliases
                self._aliases = {
                    k: v for k, v in self._aliases.items()
                    if v != (component_type, name)
                }
                logger.info(f"Unregistered {component_type.value}: {name}")
                return True
            return False

    def get(
        self,
        name: str,
        component_type: ComponentType,
        **kwargs: Any,
    ) -> Any:
        """Get a component instance by name and type.

        P1 FIX: Releases lock before calling get_instance() to prevent deadlock
        when factory function tries to register additional components.

        Args:
            name: Name of the component.
            component_type: Type of the component.
            **kwargs: Arguments to pass to factory if creating new instance.

        Returns:
            Component instance.

        Raises:
            ConfigurationError: If component is not found.
        """
        # P1 FIX: Get the ComponentInfo reference inside the lock, then release
        # the lock before calling get_instance(). This prevents deadlock when
        # the factory tries to register or retrieve other components.
        with self._lock:
            # Check for alias
            if name in self._aliases:
                aliased_type, aliased_name = self._aliases[name]
                if aliased_type == component_type:
                    name = aliased_name

            type_registry = self._components[component_type]
            if name not in type_registry:
                raise ConfigurationError(
                    f"Component '{name}' not found in {component_type.value} registry",
                    details={"component_name": name, "component_type": component_type.value},
                )

            # Get reference while holding lock
            component_info = type_registry[name]

        # P1 FIX: Call get_instance() OUTSIDE the lock
        return component_info.get_instance(**kwargs)

    def get_by_alias(self, alias: str, **kwargs: Any) -> Any:
        """Get a component instance by alias.

        Args:
            alias: Alias name of the component.
            **kwargs: Arguments to pass to factory.

        Returns:
            Component instance.

        Raises:
            ConfigurationError: If alias is not found.
        """
        with self._lock:
            if alias not in self._aliases:
                raise ConfigurationError(
                    f"Alias '{alias}' not found",
                    details={"alias": alias},
                )
            component_type, name = self._aliases[alias]
            return self.get(name, component_type, **kwargs)

    def has(self, name: str, component_type: ComponentType) -> bool:
        """Check if a component is registered.

        Args:
            name: Name of the component.
            component_type: Type of the component.

        Returns:
            True if component is registered.
        """
        with self._lock:
            return name in self._components[component_type]

    def list_components(
        self,
        component_type: ComponentType | None = None,
    ) -> dict[str, list[str]]:
        """List all registered components.

        Args:
            component_type: Optional type to filter by.

        Returns:
            Dictionary mapping component types to list of names.
        """
        with self._lock:
            if component_type:
                return {component_type.value: list(self._components[component_type].keys())}
            return {
                ct.value: list(registry.keys())
                for ct, registry in self._components.items()
                if registry
            }

    def get_metadata(
        self,
        name: str,
        component_type: ComponentType,
    ) -> dict[str, Any]:
        """Get metadata for a component.

        Args:
            name: Name of the component.
            component_type: Type of the component.

        Returns:
            Component metadata dictionary.

        Raises:
            ConfigurationError: If component is not found.
        """
        with self._lock:
            type_registry = self._components[component_type]
            if name not in type_registry:
                raise ConfigurationError(
                    f"Component '{name}' not found in {component_type.value} registry",
                    details={"component_name": name, "component_type": component_type.value},
                )
            return type_registry[name].metadata.copy()

    def reset(self, name: str, component_type: ComponentType) -> bool:
        """Reset a component's cached instance.

        Useful for hot-reloading components with new configurations.

        Args:
            name: Name of the component to reset.
            component_type: Type of the component.

        Returns:
            True if component was found and reset, False otherwise.
        """
        with self._lock:
            type_registry = self._components[component_type]
            if name in type_registry:
                type_registry[name].reset_instance()
                logger.info(f"Reset {component_type.value}: {name}")
                return True
            return False

    def reset_all(self, component_type: ComponentType | None = None) -> int:
        """Reset all cached component instances.

        Args:
            component_type: Optional type to filter by.

        Returns:
            Number of components reset.
        """
        with self._lock:
            count = 0
            types_to_reset = [component_type] if component_type else list(ComponentType)

            for ct in types_to_reset:
                for info in self._components[ct].values():
                    info.reset_instance()
                    count += 1

            logger.info(f"Reset {count} component instances")
            return count

    def clear(self, component_type: ComponentType | None = None) -> int:
        """Clear all registrations.

        Args:
            component_type: Optional type to clear. If None, clears all.

        Returns:
            Number of components cleared.
        """
        with self._lock:
            count = 0
            if component_type:
                count = len(self._components[component_type])
                self._components[component_type].clear()
                # Clear related aliases
                self._aliases = {
                    k: v for k, v in self._aliases.items()
                    if v[0] != component_type
                }
            else:
                for registry in self._components.values():
                    count += len(registry)
                    registry.clear()
                self._aliases.clear()

            logger.info(f"Cleared {count} component registrations")
            return count


# Decorator for easy component registration
def register_component(
    name: str,
    component_type: ComponentType,
    singleton: bool = True,
    metadata: dict[str, Any] | None = None,
    aliases: list[str] | None = None,
) -> Callable[[type[T]], type[T]]:
    """Decorator to register a class as a component.

    Args:
        name: Unique name for the component.
        component_type: Type of component.
        singleton: Whether to cache a single instance.
        metadata: Optional metadata.
        aliases: Optional alias names.

    Returns:
        Decorator function.

    Example:
        @register_component("xgboost", ComponentType.MODEL)
        class XGBoostModel:
            pass
    """
    def decorator(cls: type[T]) -> type[T]:
        registry.register(
            name=name,
            component_type=component_type,
            factory=cls,
            singleton=singleton,
            metadata=metadata,
            aliases=aliases,
        )
        return cls
    return decorator


# Global registry instance
registry = ComponentRegistry()


# Convenience functions for common component types
def register_model(
    name: str,
    factory: Callable[..., T],
    singleton: bool = True,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Register a model component."""
    registry.register(name, ComponentType.MODEL, factory, singleton, metadata)


def register_alpha(
    name: str,
    factory: Callable[..., T],
    singleton: bool = True,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Register an alpha component."""
    registry.register(name, ComponentType.ALPHA, factory, singleton, metadata)


def register_feature(
    name: str,
    factory: Callable[..., T],
    singleton: bool = True,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Register a feature component."""
    registry.register(name, ComponentType.FEATURE, factory, singleton, metadata)


def get_model(name: str, **kwargs: Any) -> Any:
    """Get a model by name."""
    return registry.get(name, ComponentType.MODEL, **kwargs)


def get_alpha(name: str, **kwargs: Any) -> Any:
    """Get an alpha by name."""
    return registry.get(name, ComponentType.ALPHA, **kwargs)


def get_feature(name: str, **kwargs: Any) -> Any:
    """Get a feature by name."""
    return registry.get(name, ComponentType.FEATURE, **kwargs)
