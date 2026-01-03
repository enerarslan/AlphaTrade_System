"""
Feature flags implementation for gradual rollout of new strategies and features.

Provides JPMorgan-level feature flag management with:
- Percentage-based gradual rollout
- Environment-specific flag values
- User/symbol targeting
- A/B testing capabilities
- Thread-safe evaluation
- Persistent storage
- Audit logging
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RolloutStrategy(str, Enum):
    """Strategy for rolling out a feature flag."""

    ALL = "all"  # Enable for all
    NONE = "none"  # Disable for all
    PERCENTAGE = "percentage"  # Enable for percentage of users/symbols
    ALLOWLIST = "allowlist"  # Enable only for specific entities
    BLOCKLIST = "blocklist"  # Enable for all except specific entities
    ENVIRONMENT = "environment"  # Based on environment (dev/staging/prod)
    GRADUAL = "gradual"  # Time-based gradual rollout
    AB_TEST = "ab_test"  # A/B testing with control/variant groups


class Environment(str, Enum):
    """Deployment environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class RolloutConfig:
    """Configuration for feature flag rollout."""

    strategy: RolloutStrategy = RolloutStrategy.NONE
    percentage: float = 0.0  # 0-100 for percentage rollout
    allowlist: list[str] = field(default_factory=list)
    blocklist: list[str] = field(default_factory=list)
    environments: list[Environment] = field(default_factory=list)

    # Gradual rollout settings
    gradual_start_time: datetime | None = None
    gradual_end_time: datetime | None = None
    gradual_start_percentage: float = 0.0
    gradual_end_percentage: float = 100.0

    # A/B test settings
    ab_test_id: str | None = None
    ab_variant_weights: dict[str, float] = field(default_factory=dict)  # variant -> weight

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy": self.strategy.value,
            "percentage": self.percentage,
            "allowlist": self.allowlist,
            "blocklist": self.blocklist,
            "environments": [e.value for e in self.environments],
            "gradual_start_time": self.gradual_start_time.isoformat() if self.gradual_start_time else None,
            "gradual_end_time": self.gradual_end_time.isoformat() if self.gradual_end_time else None,
            "gradual_start_percentage": self.gradual_start_percentage,
            "gradual_end_percentage": self.gradual_end_percentage,
            "ab_test_id": self.ab_test_id,
            "ab_variant_weights": self.ab_variant_weights,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RolloutConfig:
        """Create from dictionary."""
        return cls(
            strategy=RolloutStrategy(data.get("strategy", "none")),
            percentage=data.get("percentage", 0.0),
            allowlist=data.get("allowlist", []),
            blocklist=data.get("blocklist", []),
            environments=[Environment(e) for e in data.get("environments", [])],
            gradual_start_time=datetime.fromisoformat(data["gradual_start_time"]) if data.get("gradual_start_time") else None,
            gradual_end_time=datetime.fromisoformat(data["gradual_end_time"]) if data.get("gradual_end_time") else None,
            gradual_start_percentage=data.get("gradual_start_percentage", 0.0),
            gradual_end_percentage=data.get("gradual_end_percentage", 100.0),
            ab_test_id=data.get("ab_test_id"),
            ab_variant_weights=data.get("ab_variant_weights", {}),
        )


@dataclass
class FeatureFlag:
    """Feature flag definition."""

    name: str
    description: str = ""
    enabled: bool = False  # Master switch
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    default_value: Any = None  # Default value if flag is disabled
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    owner: str = ""  # Team/person responsible
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "rollout": self.rollout.to_dict(),
            "default_value": self.default_value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "owner": self.owner,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureFlag:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            enabled=data.get("enabled", False),
            rollout=RolloutConfig.from_dict(data.get("rollout", {})),
            default_value=data.get("default_value"),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
            owner=data.get("owner", ""),
            tags=data.get("tags", []),
        )


@dataclass
class FlagEvaluation:
    """Result of evaluating a feature flag."""

    flag_name: str
    enabled: bool
    variant: str | None = None  # For A/B tests
    value: Any = None
    reason: str = ""
    entity_id: str | None = None
    evaluated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/audit."""
        return {
            "flag_name": self.flag_name,
            "enabled": self.enabled,
            "variant": self.variant,
            "value": self.value,
            "reason": self.reason,
            "entity_id": self.entity_id,
            "evaluated_at": self.evaluated_at.isoformat(),
        }


class FeatureFlagStorage(ABC):
    """Abstract base class for feature flag storage backends."""

    @abstractmethod
    def get_flag(self, name: str) -> FeatureFlag | None:
        """Get a feature flag by name."""
        pass

    @abstractmethod
    def get_all_flags(self) -> dict[str, FeatureFlag]:
        """Get all feature flags."""
        pass

    @abstractmethod
    def save_flag(self, flag: FeatureFlag) -> None:
        """Save a feature flag."""
        pass

    @abstractmethod
    def delete_flag(self, name: str) -> bool:
        """Delete a feature flag."""
        pass


class InMemoryFlagStorage(FeatureFlagStorage):
    """In-memory feature flag storage."""

    def __init__(self) -> None:
        self._flags: dict[str, FeatureFlag] = {}
        self._lock = threading.RLock()

    def get_flag(self, name: str) -> FeatureFlag | None:
        with self._lock:
            return self._flags.get(name)

    def get_all_flags(self) -> dict[str, FeatureFlag]:
        with self._lock:
            return dict(self._flags)

    def save_flag(self, flag: FeatureFlag) -> None:
        with self._lock:
            flag.updated_at = datetime.utcnow()
            self._flags[flag.name] = flag

    def delete_flag(self, name: str) -> bool:
        with self._lock:
            if name in self._flags:
                del self._flags[name]
                return True
            return False


class FileFlagStorage(FeatureFlagStorage):
    """File-based feature flag storage with JSON format."""

    def __init__(self, file_path: str | Path) -> None:
        self._file_path = Path(file_path)
        self._lock = threading.RLock()
        self._flags: dict[str, FeatureFlag] = {}
        self._load()

    def _load(self) -> None:
        """Load flags from file."""
        if self._file_path.exists():
            try:
                with open(self._file_path, "r") as f:
                    data = json.load(f)
                    self._flags = {
                        name: FeatureFlag.from_dict(flag_data)
                        for name, flag_data in data.get("flags", {}).items()
                    }
            except Exception as e:
                logger.error(f"Failed to load feature flags from {self._file_path}: {e}")
                self._flags = {}

    def _save(self) -> None:
        """Save flags to file."""
        try:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._file_path, "w") as f:
                data = {
                    "flags": {name: flag.to_dict() for name, flag in self._flags.items()},
                    "updated_at": datetime.utcnow().isoformat(),
                }
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save feature flags to {self._file_path}: {e}")

    def get_flag(self, name: str) -> FeatureFlag | None:
        with self._lock:
            return self._flags.get(name)

    def get_all_flags(self) -> dict[str, FeatureFlag]:
        with self._lock:
            return dict(self._flags)

    def save_flag(self, flag: FeatureFlag) -> None:
        with self._lock:
            flag.updated_at = datetime.utcnow()
            self._flags[flag.name] = flag
            self._save()

    def delete_flag(self, name: str) -> bool:
        with self._lock:
            if name in self._flags:
                del self._flags[name]
                self._save()
                return True
            return False

    def reload(self) -> None:
        """Reload flags from file."""
        with self._lock:
            self._load()


class FeatureFlagManager:
    """
    Singleton manager for feature flag evaluation.

    Provides thread-safe flag evaluation with:
    - Percentage-based rollout using consistent hashing
    - Environment-specific flags
    - Gradual time-based rollout
    - A/B testing with variant assignment
    - Audit logging of evaluations
    """

    _instance: FeatureFlagManager | None = None
    _lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> FeatureFlagManager:
        """Thread-safe singleton implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        storage: FeatureFlagStorage | None = None,
        environment: Environment = Environment.DEVELOPMENT,
        enable_audit_log: bool = True,
    ) -> None:
        if self._initialized:
            return

        self._storage = storage or InMemoryFlagStorage()
        self._environment = environment
        self._enable_audit_log = enable_audit_log
        self._evaluation_cache: dict[str, tuple[FlagEvaluation, float]] = {}
        self._cache_ttl = 60.0  # seconds
        self._cache_lock = threading.RLock()
        self._evaluation_callbacks: list[Callable[[FlagEvaluation], None]] = []
        self._initialized = True

        logger.info(f"FeatureFlagManager initialized for environment: {environment.value}")

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def set_storage(self, storage: FeatureFlagStorage) -> None:
        """Set the storage backend."""
        self._storage = storage

    def set_environment(self, environment: Environment) -> None:
        """Set the current environment."""
        self._environment = environment
        self.clear_cache()

    def register_evaluation_callback(self, callback: Callable[[FlagEvaluation], None]) -> None:
        """Register a callback to be invoked on each evaluation (for audit logging)."""
        self._evaluation_callbacks.append(callback)

    def create_flag(
        self,
        name: str,
        description: str = "",
        enabled: bool = False,
        rollout: RolloutConfig | None = None,
        default_value: Any = None,
        owner: str = "",
        tags: list[str] | None = None,
    ) -> FeatureFlag:
        """Create a new feature flag."""
        flag = FeatureFlag(
            name=name,
            description=description,
            enabled=enabled,
            rollout=rollout or RolloutConfig(),
            default_value=default_value,
            owner=owner,
            tags=tags or [],
        )
        self._storage.save_flag(flag)
        logger.info(f"Created feature flag: {name}")
        return flag

    def get_flag(self, name: str) -> FeatureFlag | None:
        """Get a feature flag by name."""
        return self._storage.get_flag(name)

    def update_flag(self, flag: FeatureFlag) -> None:
        """Update a feature flag."""
        self._storage.save_flag(flag)
        self.clear_cache(flag.name)
        logger.info(f"Updated feature flag: {flag.name}")

    def delete_flag(self, name: str) -> bool:
        """Delete a feature flag."""
        result = self._storage.delete_flag(name)
        if result:
            self.clear_cache(name)
            logger.info(f"Deleted feature flag: {name}")
        return result

    def list_flags(self, tag: str | None = None) -> list[FeatureFlag]:
        """List all feature flags, optionally filtered by tag."""
        flags = list(self._storage.get_all_flags().values())
        if tag:
            flags = [f for f in flags if tag in f.tags]
        return flags

    def is_enabled(
        self,
        flag_name: str,
        entity_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """
        Check if a feature flag is enabled for the given entity.

        Args:
            flag_name: Name of the feature flag
            entity_id: Optional entity identifier (user ID, symbol, etc.)
            context: Optional additional context for evaluation

        Returns:
            True if the flag is enabled, False otherwise
        """
        evaluation = self.evaluate(flag_name, entity_id, context)
        return evaluation.enabled

    def evaluate(
        self,
        flag_name: str,
        entity_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> FlagEvaluation:
        """
        Evaluate a feature flag with full result details.

        Args:
            flag_name: Name of the feature flag
            entity_id: Optional entity identifier
            context: Optional additional context

        Returns:
            FlagEvaluation with enabled status, variant, reason, etc.
        """
        # Check cache first
        cache_key = f"{flag_name}:{entity_id}:{self._environment.value}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        # Get flag definition
        flag = self._storage.get_flag(flag_name)
        if flag is None:
            evaluation = FlagEvaluation(
                flag_name=flag_name,
                enabled=False,
                reason="flag_not_found",
                entity_id=entity_id,
            )
            self._notify_evaluation(evaluation)
            return evaluation

        # Master switch check
        if not flag.enabled:
            evaluation = FlagEvaluation(
                flag_name=flag_name,
                enabled=False,
                value=flag.default_value,
                reason="flag_disabled",
                entity_id=entity_id,
            )
            self._cache_evaluation(cache_key, evaluation)
            self._notify_evaluation(evaluation)
            return evaluation

        # Evaluate based on rollout strategy
        evaluation = self._evaluate_rollout(flag, entity_id, context)
        self._cache_evaluation(cache_key, evaluation)
        self._notify_evaluation(evaluation)
        return evaluation

    def _evaluate_rollout(
        self,
        flag: FeatureFlag,
        entity_id: str | None,
        context: dict[str, Any] | None,
    ) -> FlagEvaluation:
        """Evaluate flag based on rollout configuration."""
        rollout = flag.rollout
        context = context or {}

        if rollout.strategy == RolloutStrategy.ALL:
            return FlagEvaluation(
                flag_name=flag.name,
                enabled=True,
                value=True,
                reason="strategy_all",
                entity_id=entity_id,
            )

        elif rollout.strategy == RolloutStrategy.NONE:
            return FlagEvaluation(
                flag_name=flag.name,
                enabled=False,
                value=flag.default_value,
                reason="strategy_none",
                entity_id=entity_id,
            )

        elif rollout.strategy == RolloutStrategy.PERCENTAGE:
            if entity_id is None:
                return FlagEvaluation(
                    flag_name=flag.name,
                    enabled=False,
                    value=flag.default_value,
                    reason="percentage_no_entity_id",
                    entity_id=entity_id,
                )

            bucket = self._get_bucket(flag.name, entity_id)
            enabled = bucket < rollout.percentage
            return FlagEvaluation(
                flag_name=flag.name,
                enabled=enabled,
                value=enabled if enabled else flag.default_value,
                reason=f"percentage_bucket_{bucket:.1f}",
                entity_id=entity_id,
            )

        elif rollout.strategy == RolloutStrategy.ALLOWLIST:
            enabled = entity_id in rollout.allowlist
            return FlagEvaluation(
                flag_name=flag.name,
                enabled=enabled,
                value=enabled if enabled else flag.default_value,
                reason="allowlist" if enabled else "not_in_allowlist",
                entity_id=entity_id,
            )

        elif rollout.strategy == RolloutStrategy.BLOCKLIST:
            enabled = entity_id not in rollout.blocklist
            return FlagEvaluation(
                flag_name=flag.name,
                enabled=enabled,
                value=enabled if enabled else flag.default_value,
                reason="not_in_blocklist" if enabled else "blocklist",
                entity_id=entity_id,
            )

        elif rollout.strategy == RolloutStrategy.ENVIRONMENT:
            enabled = self._environment in rollout.environments
            return FlagEvaluation(
                flag_name=flag.name,
                enabled=enabled,
                value=enabled if enabled else flag.default_value,
                reason=f"environment_{self._environment.value}",
                entity_id=entity_id,
            )

        elif rollout.strategy == RolloutStrategy.GRADUAL:
            percentage = self._calculate_gradual_percentage(rollout)
            if entity_id is None:
                return FlagEvaluation(
                    flag_name=flag.name,
                    enabled=False,
                    value=flag.default_value,
                    reason="gradual_no_entity_id",
                    entity_id=entity_id,
                )

            bucket = self._get_bucket(flag.name, entity_id)
            enabled = bucket < percentage
            return FlagEvaluation(
                flag_name=flag.name,
                enabled=enabled,
                value=enabled if enabled else flag.default_value,
                reason=f"gradual_percentage_{percentage:.1f}_bucket_{bucket:.1f}",
                entity_id=entity_id,
            )

        elif rollout.strategy == RolloutStrategy.AB_TEST:
            if entity_id is None:
                return FlagEvaluation(
                    flag_name=flag.name,
                    enabled=False,
                    variant="control",
                    value=flag.default_value,
                    reason="ab_test_no_entity_id",
                    entity_id=entity_id,
                )

            variant = self._assign_ab_variant(flag.name, entity_id, rollout.ab_variant_weights)
            enabled = variant != "control"
            return FlagEvaluation(
                flag_name=flag.name,
                enabled=enabled,
                variant=variant,
                value=variant,
                reason=f"ab_test_variant_{variant}",
                entity_id=entity_id,
            )

        # Default: disabled
        return FlagEvaluation(
            flag_name=flag.name,
            enabled=False,
            value=flag.default_value,
            reason="unknown_strategy",
            entity_id=entity_id,
        )

    def _get_bucket(self, flag_name: str, entity_id: str) -> float:
        """
        Get consistent bucket (0-100) for an entity.
        Uses SHA-256 for uniform distribution.
        """
        key = f"{flag_name}:{entity_id}"
        hash_bytes = hashlib.sha256(key.encode()).digest()
        # Use first 4 bytes to get a 32-bit integer
        hash_int = int.from_bytes(hash_bytes[:4], byteorder="big")
        # Normalize to 0-100
        return (hash_int / (2**32 - 1)) * 100

    def _calculate_gradual_percentage(self, rollout: RolloutConfig) -> float:
        """Calculate current percentage for gradual rollout."""
        now = datetime.utcnow()

        if rollout.gradual_start_time is None or rollout.gradual_end_time is None:
            return rollout.gradual_start_percentage

        if now < rollout.gradual_start_time:
            return rollout.gradual_start_percentage

        if now >= rollout.gradual_end_time:
            return rollout.gradual_end_percentage

        # Linear interpolation
        total_duration = (rollout.gradual_end_time - rollout.gradual_start_time).total_seconds()
        elapsed = (now - rollout.gradual_start_time).total_seconds()
        progress = elapsed / total_duration

        return rollout.gradual_start_percentage + (
            rollout.gradual_end_percentage - rollout.gradual_start_percentage
        ) * progress

    def _assign_ab_variant(
        self,
        flag_name: str,
        entity_id: str,
        variant_weights: dict[str, float],
    ) -> str:
        """Assign entity to A/B test variant based on weights."""
        if not variant_weights:
            # Default 50/50 split
            variant_weights = {"control": 50.0, "treatment": 50.0}

        bucket = self._get_bucket(f"ab:{flag_name}", entity_id)

        cumulative = 0.0
        for variant, weight in sorted(variant_weights.items()):
            cumulative += weight
            if bucket < cumulative:
                return variant

        return "control"

    def _get_from_cache(self, cache_key: str) -> FlagEvaluation | None:
        """Get evaluation from cache if not expired."""
        with self._cache_lock:
            if cache_key in self._evaluation_cache:
                evaluation, timestamp = self._evaluation_cache[cache_key]
                if time.time() - timestamp < self._cache_ttl:
                    return evaluation
                else:
                    del self._evaluation_cache[cache_key]
        return None

    def _cache_evaluation(self, cache_key: str, evaluation: FlagEvaluation) -> None:
        """Cache an evaluation result."""
        with self._cache_lock:
            self._evaluation_cache[cache_key] = (evaluation, time.time())

    def clear_cache(self, flag_name: str | None = None) -> None:
        """Clear evaluation cache, optionally for specific flag."""
        with self._cache_lock:
            if flag_name:
                keys_to_delete = [k for k in self._evaluation_cache if k.startswith(f"{flag_name}:")]
                for key in keys_to_delete:
                    del self._evaluation_cache[key]
            else:
                self._evaluation_cache.clear()

    def _notify_evaluation(self, evaluation: FlagEvaluation) -> None:
        """Notify evaluation callbacks."""
        if not self._enable_audit_log:
            return

        for callback in self._evaluation_callbacks:
            try:
                callback(evaluation)
            except Exception as e:
                logger.error(f"Error in evaluation callback: {e}")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            return {
                "size": len(self._evaluation_cache),
                "ttl_seconds": self._cache_ttl,
            }


# Convenience functions for common use cases

def create_strategy_flag(
    name: str,
    description: str,
    initial_percentage: float = 0.0,
    owner: str = "",
) -> FeatureFlag:
    """
    Create a feature flag for gradual strategy rollout.

    Args:
        name: Flag name (e.g., "strategy.momentum_v2")
        description: Description of the strategy
        initial_percentage: Starting rollout percentage (0-100)
        owner: Owner team/person

    Returns:
        Created FeatureFlag
    """
    manager = FeatureFlagManager()
    return manager.create_flag(
        name=name,
        description=description,
        enabled=True,
        rollout=RolloutConfig(
            strategy=RolloutStrategy.PERCENTAGE,
            percentage=initial_percentage,
        ),
        owner=owner,
        tags=["strategy"],
    )


def create_gradual_rollout_flag(
    name: str,
    description: str,
    start_time: datetime,
    end_time: datetime,
    start_percentage: float = 0.0,
    end_percentage: float = 100.0,
    owner: str = "",
) -> FeatureFlag:
    """
    Create a feature flag with time-based gradual rollout.

    Args:
        name: Flag name
        description: Flag description
        start_time: When to start rollout
        end_time: When to complete rollout
        start_percentage: Starting percentage
        end_percentage: Ending percentage
        owner: Owner team/person

    Returns:
        Created FeatureFlag
    """
    manager = FeatureFlagManager()
    return manager.create_flag(
        name=name,
        description=description,
        enabled=True,
        rollout=RolloutConfig(
            strategy=RolloutStrategy.GRADUAL,
            gradual_start_time=start_time,
            gradual_end_time=end_time,
            gradual_start_percentage=start_percentage,
            gradual_end_percentage=end_percentage,
        ),
        owner=owner,
        tags=["gradual_rollout"],
    )


def create_ab_test_flag(
    name: str,
    description: str,
    variant_weights: dict[str, float] | None = None,
    owner: str = "",
) -> FeatureFlag:
    """
    Create a feature flag for A/B testing.

    Args:
        name: Flag name
        description: Description of the A/B test
        variant_weights: Dict mapping variant names to weights (should sum to 100)
        owner: Owner team/person

    Returns:
        Created FeatureFlag
    """
    if variant_weights is None:
        variant_weights = {"control": 50.0, "treatment": 50.0}

    manager = FeatureFlagManager()
    return manager.create_flag(
        name=name,
        description=description,
        enabled=True,
        rollout=RolloutConfig(
            strategy=RolloutStrategy.AB_TEST,
            ab_test_id=name,
            ab_variant_weights=variant_weights,
        ),
        owner=owner,
        tags=["ab_test"],
    )


def is_feature_enabled(
    flag_name: str,
    entity_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> bool:
    """
    Quick check if a feature flag is enabled.

    Args:
        flag_name: Name of the feature flag
        entity_id: Optional entity identifier
        context: Optional additional context

    Returns:
        True if enabled, False otherwise
    """
    manager = FeatureFlagManager()
    return manager.is_enabled(flag_name, entity_id, context)


def get_ab_variant(
    flag_name: str,
    entity_id: str,
) -> str | None:
    """
    Get A/B test variant for an entity.

    Args:
        flag_name: Name of the A/B test flag
        entity_id: Entity identifier

    Returns:
        Variant name or None if not in test
    """
    manager = FeatureFlagManager()
    evaluation = manager.evaluate(flag_name, entity_id)
    return evaluation.variant
