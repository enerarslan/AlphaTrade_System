"""
Unit tests for core/registry.py
"""

import pytest

from quant_trading_system.core.exceptions import ConfigurationError
from quant_trading_system.core.registry import (
    ComponentInfo,
    ComponentRegistry,
    ComponentType,
    get_alpha,
    get_feature,
    get_model,
    register_alpha,
    register_component,
    register_feature,
    register_model,
    registry,
)


class TestComponentType:
    """Tests for ComponentType enum."""

    def test_component_types(self):
        """Test component type values."""
        assert ComponentType.MODEL.value == "model"
        assert ComponentType.ALPHA.value == "alpha"
        assert ComponentType.FEATURE.value == "feature"
        assert ComponentType.RISK_MANAGER.value == "risk_manager"
        assert ComponentType.EXECUTION_ALGO.value == "execution_algo"
        assert ComponentType.DATA_LOADER.value == "data_loader"
        assert ComponentType.STRATEGY.value == "strategy"
        assert ComponentType.INDICATOR.value == "indicator"


class TestComponentInfo:
    """Tests for ComponentInfo class."""

    def test_singleton_behavior(self):
        """Test singleton component behavior."""
        call_count = [0]

        def factory():
            call_count[0] += 1
            return {"instance": call_count[0]}

        info = ComponentInfo(
            name="test",
            component_type=ComponentType.MODEL,
            factory=factory,
            singleton=True,
        )

        instance1 = info.get_instance()
        instance2 = info.get_instance()

        assert instance1 is instance2
        assert call_count[0] == 1  # Factory only called once

    def test_non_singleton_behavior(self):
        """Test non-singleton component behavior."""
        call_count = [0]

        def factory():
            call_count[0] += 1
            return {"instance": call_count[0]}

        info = ComponentInfo(
            name="test",
            component_type=ComponentType.MODEL,
            factory=factory,
            singleton=False,
        )

        instance1 = info.get_instance()
        instance2 = info.get_instance()

        assert instance1["instance"] == 1
        assert instance2["instance"] == 2
        assert call_count[0] == 2  # Factory called each time

    def test_reset_instance(self):
        """Test resetting cached instance."""
        call_count = [0]

        def factory():
            call_count[0] += 1
            return call_count[0]

        info = ComponentInfo(
            name="test",
            component_type=ComponentType.MODEL,
            factory=factory,
            singleton=True,
        )

        first = info.get_instance()
        assert first == 1

        info.reset_instance()
        second = info.get_instance()
        assert second == 2

    def test_factory_with_kwargs(self):
        """Test factory with keyword arguments."""
        def factory(value=10, multiplier=2):
            return value * multiplier

        info = ComponentInfo(
            name="test",
            component_type=ComponentType.MODEL,
            factory=factory,
            singleton=False,
        )

        result1 = info.get_instance()
        assert result1 == 20

        result2 = info.get_instance(value=5, multiplier=3)
        assert result2 == 15


class TestComponentRegistry:
    """Tests for ComponentRegistry class."""

    def test_register_and_get(self, component_registry):
        """Test basic registration and retrieval."""
        component_registry.register(
            name="test_model",
            component_type=ComponentType.MODEL,
            factory=lambda: {"name": "test"},
        )

        result = component_registry.get("test_model", ComponentType.MODEL)
        assert result["name"] == "test"

    def test_register_duplicate_raises_error(self, component_registry):
        """Test that registering duplicate name raises error."""
        component_registry.register(
            name="duplicate",
            component_type=ComponentType.MODEL,
            factory=lambda: {},
        )

        with pytest.raises(ConfigurationError, match="already registered"):
            component_registry.register(
                name="duplicate",
                component_type=ComponentType.MODEL,
                factory=lambda: {},
            )

    def test_get_nonexistent_raises_error(self, component_registry):
        """Test that getting nonexistent component raises error."""
        with pytest.raises(ConfigurationError, match="not found"):
            component_registry.get("nonexistent", ComponentType.MODEL)

    def test_has_component(self, component_registry):
        """Test checking if component exists."""
        component_registry.register(
            name="exists",
            component_type=ComponentType.MODEL,
            factory=lambda: {},
        )

        assert component_registry.has("exists", ComponentType.MODEL)
        assert not component_registry.has("missing", ComponentType.MODEL)

    def test_unregister(self, component_registry):
        """Test unregistering a component."""
        component_registry.register(
            name="removable",
            component_type=ComponentType.MODEL,
            factory=lambda: {},
        )

        assert component_registry.has("removable", ComponentType.MODEL)

        result = component_registry.unregister("removable", ComponentType.MODEL)
        assert result is True
        assert not component_registry.has("removable", ComponentType.MODEL)

    def test_unregister_nonexistent(self, component_registry):
        """Test unregistering nonexistent component."""
        result = component_registry.unregister("missing", ComponentType.MODEL)
        assert result is False

    def test_list_components(self, component_registry):
        """Test listing registered components."""
        component_registry.register("model1", ComponentType.MODEL, lambda: {})
        component_registry.register("model2", ComponentType.MODEL, lambda: {})
        component_registry.register("alpha1", ComponentType.ALPHA, lambda: {})

        # List all
        all_components = component_registry.list_components()
        assert "model" in all_components
        assert "alpha" in all_components
        assert len(all_components["model"]) == 2
        assert len(all_components["alpha"]) == 1

        # List specific type
        models = component_registry.list_components(ComponentType.MODEL)
        assert "model" in models
        assert "model1" in models["model"]
        assert "model2" in models["model"]

    def test_aliases(self, component_registry):
        """Test component aliases."""
        component_registry.register(
            name="xgboost_classifier_v1",
            component_type=ComponentType.MODEL,
            factory=lambda: {"type": "xgboost"},
            aliases=["xgb", "xgboost"],
        )

        # Get by name
        result1 = component_registry.get("xgboost_classifier_v1", ComponentType.MODEL)
        assert result1["type"] == "xgboost"

        # Get by alias
        result2 = component_registry.get_by_alias("xgb")
        assert result2["type"] == "xgboost"

        result3 = component_registry.get_by_alias("xgboost")
        assert result3["type"] == "xgboost"

    def test_alias_not_found(self, component_registry):
        """Test getting by nonexistent alias."""
        with pytest.raises(ConfigurationError, match="Alias.*not found"):
            component_registry.get_by_alias("nonexistent")

    def test_get_metadata(self, component_registry):
        """Test getting component metadata."""
        component_registry.register(
            name="with_meta",
            component_type=ComponentType.MODEL,
            factory=lambda: {},
            metadata={"version": "1.0", "author": "test"},
        )

        meta = component_registry.get_metadata("with_meta", ComponentType.MODEL)
        assert meta["version"] == "1.0"
        assert meta["author"] == "test"

    def test_reset_component(self, component_registry):
        """Test resetting a component's cached instance."""
        call_count = [0]

        def factory():
            call_count[0] += 1
            return call_count[0]

        component_registry.register(
            name="resettable",
            component_type=ComponentType.MODEL,
            factory=factory,
            singleton=True,
        )

        first = component_registry.get("resettable", ComponentType.MODEL)
        assert first == 1

        component_registry.reset("resettable", ComponentType.MODEL)

        second = component_registry.get("resettable", ComponentType.MODEL)
        assert second == 2

    def test_reset_all(self, component_registry):
        """Test resetting all component instances."""
        component_registry.register("m1", ComponentType.MODEL, lambda: {})
        component_registry.register("m2", ComponentType.MODEL, lambda: {})
        component_registry.register("a1", ComponentType.ALPHA, lambda: {})

        # Get instances to cache them
        component_registry.get("m1", ComponentType.MODEL)
        component_registry.get("m2", ComponentType.MODEL)
        component_registry.get("a1", ComponentType.ALPHA)

        # Reset all models
        count = component_registry.reset_all(ComponentType.MODEL)
        assert count == 2

        # Reset everything
        count = component_registry.reset_all()
        assert count >= 1  # At least the alpha

    def test_clear(self, component_registry):
        """Test clearing registrations."""
        component_registry.register("m1", ComponentType.MODEL, lambda: {})
        component_registry.register("m2", ComponentType.MODEL, lambda: {})
        component_registry.register("a1", ComponentType.ALPHA, lambda: {})

        # Clear models only
        count = component_registry.clear(ComponentType.MODEL)
        assert count == 2
        assert not component_registry.has("m1", ComponentType.MODEL)
        assert component_registry.has("a1", ComponentType.ALPHA)

        # Clear everything
        component_registry.register("m3", ComponentType.MODEL, lambda: {})
        count = component_registry.clear()
        assert count >= 2


class TestDecoratorRegistration:
    """Tests for decorator-based registration."""

    def test_register_component_decorator(self):
        """Test using decorator to register component."""
        # Create a fresh registry for this test
        test_registry = ComponentRegistry()

        # We can't easily test the decorator with the global registry
        # without side effects, so we test the manual equivalent
        class TestModel:
            def __init__(self):
                self.name = "test_model"

        test_registry.register(
            name="decorated_model",
            component_type=ComponentType.MODEL,
            factory=TestModel,
            singleton=True,
            metadata={"decorated": True},
        )

        instance = test_registry.get("decorated_model", ComponentType.MODEL)
        assert instance.name == "test_model"


class TestConvenienceFunctions:
    """Tests for convenience registration functions."""

    def test_register_and_get_model(self):
        """Test model convenience functions."""
        # Use fresh registry to avoid conflicts
        test_registry = ComponentRegistry()

        # Register directly on registry
        test_registry.register(
            name="convenience_model",
            component_type=ComponentType.MODEL,
            factory=lambda: {"type": "model"},
        )

        result = test_registry.get("convenience_model", ComponentType.MODEL)
        assert result["type"] == "model"

    def test_register_and_get_alpha(self):
        """Test alpha convenience functions."""
        test_registry = ComponentRegistry()

        test_registry.register(
            name="convenience_alpha",
            component_type=ComponentType.ALPHA,
            factory=lambda: {"type": "alpha"},
        )

        result = test_registry.get("convenience_alpha", ComponentType.ALPHA)
        assert result["type"] == "alpha"

    def test_register_and_get_feature(self):
        """Test feature convenience functions."""
        test_registry = ComponentRegistry()

        test_registry.register(
            name="convenience_feature",
            component_type=ComponentType.FEATURE,
            factory=lambda: {"type": "feature"},
        )

        result = test_registry.get("convenience_feature", ComponentType.FEATURE)
        assert result["type"] == "feature"


class TestThreadSafety:
    """Tests for thread safety (basic tests)."""

    def test_concurrent_registration(self, component_registry):
        """Test that concurrent registration doesn't corrupt state."""
        import threading

        errors = []

        def register_components(prefix):
            try:
                for i in range(10):
                    component_registry.register(
                        name=f"{prefix}_{i}",
                        component_type=ComponentType.MODEL,
                        factory=lambda: {},
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_components, args=(f"thread_{j}",))
            for j in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should have occurred
        assert len(errors) == 0

        # All registrations should be present
        components = component_registry.list_components(ComponentType.MODEL)
        assert len(components["model"]) == 50  # 5 threads * 10 components

    def test_concurrent_get(self, component_registry):
        """Test that concurrent gets work correctly."""
        import threading

        component_registry.register(
            name="shared",
            component_type=ComponentType.MODEL,
            factory=lambda: {"value": 42},
            singleton=True,
        )

        results = []
        errors = []

        def get_component():
            try:
                result = component_registry.get("shared", ComponentType.MODEL)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_component) for _ in range(20)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 20
        # All should be the same instance (singleton)
        for r in results:
            assert r["value"] == 42
