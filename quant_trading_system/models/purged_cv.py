"""
Purged Cross-Validation for Financial Time Series.

P2-B Enhancement: Implements purged and embargo cross-validation to prevent
data leakage in financial ML models.

Key concepts:
- Purging: Remove training samples that overlap with test period
- Embargo: Add buffer period after test to prevent information leakage
- Combinatorial: Test on multiple out-of-sample periods

Based on:
- "Advances in Financial Machine Learning" by Marcos López de Prado
- "Machine Learning for Asset Managers" by López de Prado

Expected Impact: +8-15 bps from improved model generalization.

Author: AlphaTrade System
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generator

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class CVFoldResult:
    """Result of a single CV fold."""

    fold_index: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start: datetime | None = None
    train_end: datetime | None = None
    test_start: datetime | None = None
    test_end: datetime | None = None
    purged_count: int = 0
    embargo_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fold_index": self.fold_index,
            "train_size": len(self.train_indices),
            "test_size": len(self.test_indices),
            "train_start": self.train_start.isoformat() if self.train_start else None,
            "train_end": self.train_end.isoformat() if self.train_end else None,
            "test_start": self.test_start.isoformat() if self.test_start else None,
            "test_end": self.test_end.isoformat() if self.test_end else None,
            "purged_count": self.purged_count,
            "embargo_count": self.embargo_count,
        }


@dataclass
class CVSummary:
    """Summary of cross-validation split."""

    n_splits: int
    total_samples: int
    avg_train_size: float
    avg_test_size: float
    total_purged: int
    total_embargoed: int
    folds: list[CVFoldResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_splits": self.n_splits,
            "total_samples": self.total_samples,
            "avg_train_size": self.avg_train_size,
            "avg_test_size": self.avg_test_size,
            "total_purged": self.total_purged,
            "total_embargoed": self.total_embargoed,
            "folds": [f.to_dict() for f in self.folds],
        }


# =============================================================================
# Purged K-Fold Cross-Validation
# =============================================================================


class PurgedKFold(BaseCrossValidator):
    """
    P2-B Enhancement: Purged K-Fold Cross-Validation.

    Implements time-series aware cross-validation that:
    1. Purges training samples that overlap with test period
    2. Adds embargo period after test to prevent information leakage
    3. Respects temporal ordering of financial data

    This is CRITICAL for preventing look-ahead bias in financial ML models.

    Example:
        >>> cv = PurgedKFold(n_splits=5, purge_gap=5, embargo_pct=0.01)
        >>> for train_idx, test_idx in cv.split(X, y, timestamps):
        ...     model.fit(X[train_idx], y[train_idx])
        ...     predictions = model.predict(X[test_idx])

    References:
        López de Prado, M. (2018). "Advances in Financial Machine Learning"
    """

    # CRITICAL: Minimum embargo percentage to prevent residual look-ahead bias
    # Research shows embargo < 1% allows autocorrelated returns to leak information
    MIN_EMBARGO_PCT: float = 0.01

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 0,
        embargo_pct: float = 0.01,
        prediction_horizon: int = 1,
    ):
        """Initialize Purged K-Fold CV.

        Args:
            n_splits: Number of cross-validation folds.
            purge_gap: Number of periods to purge around test set boundaries.
            embargo_pct: Percentage of test set size to use as embargo period.
                        MUST be >= 0.01 (1%) to prevent look-ahead bias from
                        autocorrelated returns. Values below 1% are automatically
                        raised with a warning.
            prediction_horizon: Forward-looking window of labels (in periods).

        Raises:
            ValueError: If n_splits < 2.
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")

        # CRITICAL: Enforce minimum embargo period
        # Research from López de Prado shows embargo < 1% allows data leakage
        if embargo_pct < self.MIN_EMBARGO_PCT:
            logger.warning(
                f"embargo_pct={embargo_pct} is below minimum {self.MIN_EMBARGO_PCT}. "
                f"Raising to {self.MIN_EMBARGO_PCT} to prevent look-ahead bias from "
                f"autocorrelated returns. See López de Prado (2018) Ch. 7."
            )
            embargo_pct = self.MIN_EMBARGO_PCT

        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
        self.prediction_horizon = prediction_horizon

    def get_n_splits(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> int:
        """Return number of splits."""
        return self.n_splits

    def split(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
        groups: np.ndarray | pd.Series | None = None,
        times: pd.Series | pd.DatetimeIndex | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/test indices with purging and embargo.

        Args:
            X: Feature matrix.
            y: Target vector (optional).
            groups: Group labels for samples (optional, used for event-based CV).
            times: Timestamps for samples (optional but recommended).

        Yields:
            Tuple of (train_indices, test_indices) for each fold.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate embargo size
        test_size = n_samples // self.n_splits
        embargo_size = max(1, int(test_size * self.embargo_pct))

        for fold in range(self.n_splits):
            # Define test fold boundaries
            test_start = fold * test_size
            test_end = (fold + 1) * test_size if fold < self.n_splits - 1 else n_samples

            # Create test indices
            test_indices = indices[test_start:test_end]

            # Create train indices with purging
            train_mask = np.ones(n_samples, dtype=bool)

            # Remove test indices from training
            train_mask[test_start:test_end] = False

            # Apply purging: remove samples before test that might overlap
            purge_start = max(0, test_start - self.purge_gap - self.prediction_horizon)
            purge_end = test_start
            train_mask[purge_start:purge_end] = False

            # Apply embargo: remove samples after test
            embargo_start = test_end
            embargo_end = min(n_samples, test_end + embargo_size)
            train_mask[embargo_start:embargo_end] = False

            train_indices = indices[train_mask]

            yield train_indices, test_indices

    def get_fold_summary(
        self,
        X: np.ndarray | pd.DataFrame,
        times: pd.Series | pd.DatetimeIndex | None = None,
    ) -> CVSummary:
        """Get detailed summary of all folds.

        Args:
            X: Feature matrix.
            times: Timestamps (optional).

        Returns:
            CVSummary with fold details.
        """
        n_samples = len(X)
        test_size = n_samples // self.n_splits
        embargo_size = max(1, int(test_size * self.embargo_pct))

        folds = []
        total_purged = 0
        total_embargoed = 0
        train_sizes = []
        test_sizes = []

        for fold_idx, (train_idx, test_idx) in enumerate(self.split(X)):
            test_start = fold_idx * test_size
            test_end = (fold_idx + 1) * test_size if fold_idx < self.n_splits - 1 else n_samples

            purge_count = self.purge_gap + self.prediction_horizon
            embargo_count = embargo_size

            fold_result = CVFoldResult(
                fold_index=fold_idx,
                train_indices=train_idx,
                test_indices=test_idx,
                purged_count=purge_count,
                embargo_count=embargo_count,
            )

            if times is not None:
                if isinstance(times, pd.DatetimeIndex):
                    time_values = times
                else:
                    time_values = times.values

                if len(train_idx) > 0:
                    fold_result.train_start = pd.Timestamp(time_values[train_idx[0]])
                    fold_result.train_end = pd.Timestamp(time_values[train_idx[-1]])
                if len(test_idx) > 0:
                    fold_result.test_start = pd.Timestamp(time_values[test_idx[0]])
                    fold_result.test_end = pd.Timestamp(time_values[test_idx[-1]])

            folds.append(fold_result)
            total_purged += purge_count
            total_embargoed += embargo_count
            train_sizes.append(len(train_idx))
            test_sizes.append(len(test_idx))

        return CVSummary(
            n_splits=self.n_splits,
            total_samples=n_samples,
            avg_train_size=np.mean(train_sizes),
            avg_test_size=np.mean(test_sizes),
            total_purged=total_purged,
            total_embargoed=total_embargoed,
            folds=folds,
        )


# =============================================================================
# Combinatorial Purged Cross-Validation
# =============================================================================


class CombinatorialPurgedKFold(BaseCrossValidator):
    """
    P2-B Enhancement: Combinatorial Purged K-Fold Cross-Validation.

    Generates multiple test paths by combining different groups of folds.
    This provides more robust out-of-sample estimates by testing on
    various combinations of time periods.

    Uses purging and embargo to prevent data leakage between train/test.

    Example:
        >>> cv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2)
        >>> for train_idx, test_idx in cv.split(X, times=timestamps):
        ...     model.fit(X[train_idx], y[train_idx])
        ...     score = model.score(X[test_idx], y[test_idx])
    """

    # CRITICAL: Minimum embargo percentage to prevent residual look-ahead bias
    MIN_EMBARGO_PCT: float = 0.01

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        purge_gap: int = 0,
        embargo_pct: float = 0.01,
        prediction_horizon: int = 1,
    ):
        """Initialize Combinatorial Purged K-Fold.

        Args:
            n_splits: Total number of time groups/splits.
            n_test_splits: Number of splits to use for testing in each iteration.
            purge_gap: Number of periods to purge around boundaries.
            embargo_pct: Percentage of test size for embargo.
                        MUST be >= 0.01 (1%) to prevent look-ahead bias.
            prediction_horizon: Forward-looking window of labels.

        Raises:
            ValueError: If n_splits < 2 or n_test_splits >= n_splits.
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if n_test_splits >= n_splits:
            raise ValueError(f"n_test_splits ({n_test_splits}) must be < n_splits ({n_splits})")

        # CRITICAL: Enforce minimum embargo period
        if embargo_pct < self.MIN_EMBARGO_PCT:
            logger.warning(
                f"embargo_pct={embargo_pct} is below minimum {self.MIN_EMBARGO_PCT}. "
                f"Raising to {self.MIN_EMBARGO_PCT} to prevent look-ahead bias."
            )
            embargo_pct = self.MIN_EMBARGO_PCT

        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
        self.prediction_horizon = prediction_horizon

        # Calculate number of paths: C(n_splits, n_test_splits)
        from math import comb
        self._n_paths = comb(n_splits, n_test_splits)

    def get_n_splits(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> int:
        """Return number of splits (paths)."""
        return self._n_paths

    def _get_test_combinations(self) -> list[tuple[int, ...]]:
        """Get all combinations of test fold indices."""
        from itertools import combinations
        return list(combinations(range(self.n_splits), self.n_test_splits))

    def split(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
        groups: np.ndarray | pd.Series | None = None,
        times: pd.Series | pd.DatetimeIndex | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/test indices for each combinatorial path.

        Args:
            X: Feature matrix.
            y: Target vector (optional).
            groups: Group labels (optional).
            times: Timestamps (optional).

        Yields:
            Tuple of (train_indices, test_indices) for each path.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        group_size = n_samples // self.n_splits

        # Get all test combinations
        test_combinations = self._get_test_combinations()

        for test_group_indices in test_combinations:
            # Create test indices from selected groups
            test_mask = np.zeros(n_samples, dtype=bool)
            for group_idx in test_group_indices:
                start = group_idx * group_size
                end = (group_idx + 1) * group_size if group_idx < self.n_splits - 1 else n_samples
                test_mask[start:end] = True

            test_indices = indices[test_mask]

            # Create train mask
            train_mask = ~test_mask.copy()

            # Apply purging and embargo around each test group
            embargo_size = max(1, int(group_size * self.embargo_pct))

            for group_idx in test_group_indices:
                group_start = group_idx * group_size
                group_end = (group_idx + 1) * group_size if group_idx < self.n_splits - 1 else n_samples

                # Purge before test group
                purge_start = max(0, group_start - self.purge_gap - self.prediction_horizon)
                train_mask[purge_start:group_start] = False

                # Embargo after test group
                embargo_end = min(n_samples, group_end + embargo_size)
                train_mask[group_end:embargo_end] = False

            train_indices = indices[train_mask]

            yield train_indices, test_indices


# =============================================================================
# Walk-Forward Optimization
# =============================================================================


class WalkForwardCV(BaseCrossValidator):
    """
    P2-B Enhancement: Walk-Forward Cross-Validation.

    Implements expanding or sliding window walk-forward validation
    with purging and embargo. This is the most realistic simulation
    of live trading where we train on past data and test on future.

    Window types:
    - Expanding: Train window grows over time (all history)
    - Sliding: Train window has fixed size (recent history only)

    Example:
        >>> cv = WalkForwardCV(
        ...     n_splits=10,
        ...     train_pct=0.6,
        ...     window_type='expanding'
        ... )
        >>> for train_idx, test_idx in cv.split(X, times=timestamps):
        ...     model.fit(X[train_idx], y[train_idx])
    """

    # CRITICAL: Minimum embargo percentage to prevent residual look-ahead bias
    MIN_EMBARGO_PCT: float = 0.01

    def __init__(
        self,
        n_splits: int = 10,
        train_pct: float = 0.6,
        window_type: str = "expanding",  # "expanding" or "sliding"
        min_train_size: int | None = None,
        purge_gap: int = 0,
        embargo_pct: float = 0.01,
        prediction_horizon: int = 1,
    ):
        """Initialize Walk-Forward CV.

        Args:
            n_splits: Number of test periods.
            train_pct: Initial training percentage (for sliding window).
            window_type: "expanding" or "sliding".
            min_train_size: Minimum training samples required.
            purge_gap: Number of periods to purge.
            embargo_pct: Embargo percentage. MUST be >= 0.01 (1%).
            prediction_horizon: Label lookahead.

        Raises:
            ValueError: If n_splits < 1, train_pct not in (0, 1), or invalid window_type.
        """
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}")
        if not 0 < train_pct < 1:
            raise ValueError(f"train_pct must be in (0, 1), got {train_pct}")
        if window_type not in ("expanding", "sliding"):
            raise ValueError(f"window_type must be 'expanding' or 'sliding', got {window_type}")

        # CRITICAL: Enforce minimum embargo period
        if embargo_pct < self.MIN_EMBARGO_PCT:
            logger.warning(
                f"embargo_pct={embargo_pct} is below minimum {self.MIN_EMBARGO_PCT}. "
                f"Raising to {self.MIN_EMBARGO_PCT} to prevent look-ahead bias."
            )
            embargo_pct = self.MIN_EMBARGO_PCT

        self.n_splits = n_splits
        self.train_pct = train_pct
        self.window_type = window_type
        self.min_train_size = min_train_size
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
        self.prediction_horizon = prediction_horizon

    def get_n_splits(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> int:
        """Return number of splits."""
        return self.n_splits

    def split(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
        groups: np.ndarray | pd.Series | None = None,
        times: pd.Series | pd.DatetimeIndex | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate walk-forward train/test indices.

        Args:
            X: Feature matrix.
            y: Target vector.
            groups: Group labels.
            times: Timestamps.

        Yields:
            Tuple of (train_indices, test_indices).
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate sizes
        initial_train_size = int(n_samples * self.train_pct)
        remaining = n_samples - initial_train_size
        test_size = remaining // self.n_splits

        if self.min_train_size and initial_train_size < self.min_train_size:
            initial_train_size = self.min_train_size

        embargo_size = max(1, int(test_size * self.embargo_pct))

        for fold in range(self.n_splits):
            # Test indices
            test_start = initial_train_size + fold * test_size
            test_end = initial_train_size + (fold + 1) * test_size
            if fold == self.n_splits - 1:
                test_end = n_samples

            test_indices = indices[test_start:test_end]

            # Train indices based on window type
            if self.window_type == "expanding":
                # Use all data from start to test_start
                train_end = max(0, test_start - self.purge_gap - self.prediction_horizon)
                train_start = 0
            else:  # sliding
                # Use fixed window
                train_end = max(0, test_start - self.purge_gap - self.prediction_horizon)
                train_start = max(0, test_start - initial_train_size)

            train_indices = indices[train_start:train_end]

            # Skip if not enough training data
            if len(train_indices) < (self.min_train_size or 10):
                continue

            yield train_indices, test_indices


# =============================================================================
# Event-Based Purged CV (for Triple Barrier Labels)
# =============================================================================


class EventPurgedKFold(BaseCrossValidator):
    """
    P2-B Enhancement: Event-Based Purged K-Fold.

    Designed for event-driven labels like triple barrier method where
    each sample has start and end times (events).

    Purges training samples whose event windows overlap with test samples'
    event windows, preventing any information leakage.

    Example:
        >>> cv = EventPurgedKFold(n_splits=5)
        >>> for train_idx, test_idx in cv.split(
        ...     X,
        ...     event_starts=event_starts,
        ...     event_ends=event_ends
        ... ):
        ...     model.fit(X[train_idx], y[train_idx])
    """

    # CRITICAL: Minimum embargo percentage to prevent residual look-ahead bias
    MIN_EMBARGO_PCT: float = 0.01

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
    ):
        """Initialize Event-Based Purged K-Fold.

        Args:
            n_splits: Number of folds.
            embargo_pct: Embargo percentage. MUST be >= 0.01 (1%).

        Raises:
            ValueError: If n_splits < 2.
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")

        # CRITICAL: Enforce minimum embargo period
        if embargo_pct < self.MIN_EMBARGO_PCT:
            logger.warning(
                f"embargo_pct={embargo_pct} is below minimum {self.MIN_EMBARGO_PCT}. "
                f"Raising to {self.MIN_EMBARGO_PCT} to prevent look-ahead bias."
            )
            embargo_pct = self.MIN_EMBARGO_PCT

        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def get_n_splits(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> int:
        """Return number of splits."""
        return self.n_splits

    def split(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
        groups: np.ndarray | pd.Series | None = None,
        event_starts: pd.Series | None = None,
        event_ends: pd.Series | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate event-aware train/test indices.

        Args:
            X: Feature matrix.
            y: Target vector.
            groups: Group labels.
            event_starts: Start times for each event/sample.
            event_ends: End times for each event/sample.

        Yields:
            Tuple of (train_indices, test_indices).
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        if event_starts is None or event_ends is None:
            # Fall back to regular purged k-fold
            logger.warning("No event times provided, using basic time purging")
            fallback = PurgedKFold(n_splits=self.n_splits, embargo_pct=self.embargo_pct)
            yield from fallback.split(X, y, groups)
            return

        # Convert to arrays if needed
        if isinstance(event_starts, pd.Series):
            event_starts = event_starts.values
        if isinstance(event_ends, pd.Series):
            event_ends = event_ends.values

        test_size = n_samples // self.n_splits
        embargo_size = max(1, int(test_size * self.embargo_pct))

        for fold in range(self.n_splits):
            # Define test fold
            test_start_idx = fold * test_size
            test_end_idx = (fold + 1) * test_size if fold < self.n_splits - 1 else n_samples
            test_indices = indices[test_start_idx:test_end_idx]

            # Get time range of test events
            test_event_starts = event_starts[test_indices]
            test_event_ends = event_ends[test_indices]

            test_min_time = pd.Timestamp(np.min(test_event_starts))
            test_max_time = pd.Timestamp(np.max(test_event_ends))

            # Purge training samples that overlap with test time range
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_start_idx:test_end_idx] = False  # Remove test indices

            for i in range(n_samples):
                if train_mask[i]:
                    # Check if this training sample's event overlaps with test
                    sample_start = pd.Timestamp(event_starts[i])
                    sample_end = pd.Timestamp(event_ends[i])

                    # Overlap check: events overlap if start < other_end AND end > other_start
                    if sample_start <= test_max_time and sample_end >= test_min_time:
                        train_mask[i] = False

            # Apply embargo after test period
            embargo_end_idx = min(n_samples, test_end_idx + embargo_size)
            train_mask[test_end_idx:embargo_end_idx] = False

            train_indices = indices[train_mask]

            yield train_indices, test_indices


# =============================================================================
# P1-3.1 Enhancement: Multiple Testing Correction
# =============================================================================


class MultipleTestingCorrector:
    """
    P1-3.1 Enhancement: Multiple Testing Correction for Cross-Validation.

    When evaluating many model configurations or hyperparameters using
    combinatorial purged cross-validation, we face the multiple testing problem:
    some configurations will appear significant by chance alone.

    This class implements:
    1. Bonferroni correction (conservative, controls FWER)
    2. Benjamini-Hochberg (less conservative, controls FDR)
    3. Deflated Sharpe Ratio (accounts for variance in backtest trials)

    Based on:
    - Bailey & López de Prado (2014): "The Deflated Sharpe Ratio"
    - Harvey et al. (2016): "...and the Cross-Section of Expected Returns"

    Example:
        >>> corrector = MultipleTestingCorrector(n_trials=100)
        >>> adjusted_pvalue = corrector.bonferroni(raw_pvalue)
        >>> is_significant = corrector.is_significant_bh(p_values, alpha=0.05)
    """

    def __init__(
        self,
        n_trials: int = 1,
        significance_level: float = 0.05,
    ):
        """Initialize Multiple Testing Corrector.

        Args:
            n_trials: Number of trials/configurations tested.
            significance_level: Base significance level (alpha). Default 0.05.
        """
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {n_trials}")
        if not 0 < significance_level < 1:
            raise ValueError(f"significance_level must be in (0, 1), got {significance_level}")

        self.n_trials = n_trials
        self.alpha = significance_level

        logger.info(
            f"MultipleTestingCorrector initialized: "
            f"n_trials={n_trials}, alpha={significance_level}"
        )

    def bonferroni(self, p_value: float) -> float:
        """Apply Bonferroni correction to a p-value.

        Bonferroni is conservative and controls Family-Wise Error Rate (FWER).
        Good when you need to be very confident about ALL significant results.

        Args:
            p_value: Raw p-value to correct.

        Returns:
            Bonferroni-corrected p-value (min of 1.0).
        """
        return min(p_value * self.n_trials, 1.0)

    def bonferroni_threshold(self) -> float:
        """Get Bonferroni-corrected significance threshold.

        Returns:
            Adjusted alpha threshold.
        """
        return self.alpha / self.n_trials

    def benjamini_hochberg(
        self,
        p_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply Benjamini-Hochberg FDR correction.

        BH controls False Discovery Rate - the expected proportion of false
        positives among rejected hypotheses. Less conservative than Bonferroni.

        Args:
            p_values: Array of raw p-values.

        Returns:
            Tuple of (adjusted_p_values, is_significant).
        """
        n = len(p_values)
        if n == 0:
            return np.array([]), np.array([], dtype=bool)

        # Sort p-values
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        # BH procedure
        thresholds = np.arange(1, n + 1) / n * self.alpha

        # Find the largest k where p(k) <= k/n * alpha
        is_significant_sorted = sorted_p <= thresholds
        significant_indices = np.where(is_significant_sorted)[0]

        if len(significant_indices) == 0:
            # No significant results
            adjusted_p = np.minimum(sorted_p * n / np.arange(1, n + 1), 1.0)
            is_significant = np.zeros(n, dtype=bool)
        else:
            k_max = significant_indices[-1]
            is_significant_sorted = np.zeros(n, dtype=bool)
            is_significant_sorted[:k_max + 1] = True

            # Calculate adjusted p-values
            adjusted_p = np.zeros(n)
            for i in range(n - 1, -1, -1):
                if i == n - 1:
                    adjusted_p[i] = sorted_p[i]
                else:
                    adjusted_p[i] = min(adjusted_p[i + 1], sorted_p[i] * n / (i + 1))
            adjusted_p = np.minimum(adjusted_p, 1.0)

            is_significant = is_significant_sorted.copy()

        # Restore original order
        original_order = np.argsort(sorted_indices)
        return adjusted_p[original_order], is_significant[original_order]

    def deflated_sharpe_ratio(
        self,
        observed_sharpe: float,
        n_trials: int | None = None,
        var_sharpe: float = 1.0,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        n_returns: int = 252,
    ) -> tuple[float, float]:
        """Calculate the Deflated Sharpe Ratio (DSR).

        DSR accounts for the variance in backtest Sharpe ratios when
        multiple trials are run. It estimates the probability that
        an observed Sharpe ratio is inflated due to multiple testing.

        Based on Bailey & López de Prado (2014).

        Args:
            observed_sharpe: The observed Sharpe ratio.
            n_trials: Number of trials (uses instance default if None).
            var_sharpe: Variance of Sharpe ratio across trials.
            skewness: Skewness of returns.
            kurtosis: Kurtosis of returns (excess kurtosis + 3).
            n_returns: Number of return observations.

        Returns:
            Tuple of (deflated_sharpe, p_value).
        """
        from scipy import stats

        n_trials = n_trials or self.n_trials

        # Expected maximum Sharpe ratio under null (no skill)
        # E[max(Z_1, ..., Z_n)] for standard normal
        if n_trials <= 1:
            # Single-trial case has no multiple-testing inflation adjustment.
            expected_max = 0.0
        else:
            gamma_approx = np.euler_gamma  # Euler-Mascheroni constant
            expected_max = np.sqrt(2 * np.log(n_trials)) - (
                gamma_approx + np.log(np.pi / 2)
            ) / (2 * np.sqrt(2 * np.log(n_trials)))

        # Standard error of Sharpe ratio
        # SR ~ N(SR*, SE(SR*)) where SE = sqrt((1 + 0.5*SR^2 - skew*SR + (kurt-3)/4*SR^2) / T)
        se_sharpe = np.sqrt(
            (1 + 0.5 * observed_sharpe**2 - skewness * observed_sharpe +
             (kurtosis - 3) / 4 * observed_sharpe**2) / n_returns
        )

        # Deflated Sharpe Ratio
        dsr = (observed_sharpe - expected_max * se_sharpe) / se_sharpe

        # P-value from normal distribution
        p_value = 1 - stats.norm.cdf(dsr)

        return dsr, p_value

    def is_significant_bonferroni(self, p_value: float) -> bool:
        """Check if p-value is significant after Bonferroni correction.

        Args:
            p_value: Raw p-value.

        Returns:
            True if significant after correction.
        """
        return p_value < self.bonferroni_threshold()

    def min_trials_for_significance(
        self,
        target_sharpe: float = 1.0,
        annual_std: float = 0.15,
        significance_level: float | None = None,
    ) -> int:
        """Calculate minimum number of trials needed for a given Sharpe to be significant.

        Based on the DSR methodology, estimates how many trials you can run
        before a given Sharpe ratio becomes "expected by chance."

        Args:
            target_sharpe: The Sharpe ratio you want to be significant.
            annual_std: Annualized standard deviation of returns.
            significance_level: Override instance alpha if provided.

        Returns:
            Maximum number of trials where target_sharpe remains significant.
        """
        from scipy import stats

        alpha = significance_level or self.alpha

        # Solve for n where DSR becomes insignificant
        # This is an approximation - for exact, would need numerical solution
        z_alpha = stats.norm.ppf(1 - alpha)

        # Rearranging DSR formula:
        # target_sharpe >= E[max(Z_1,...,Z_n)] for significance
        # E[max] ~ sqrt(2*log(n)) for large n

        # Binary search for max n
        n_min, n_max = 1, 10000
        while n_max - n_min > 1:
            n_mid = (n_min + n_max) // 2
            expected_max = np.sqrt(2 * np.log(n_mid))
            if target_sharpe > expected_max + z_alpha:
                n_min = n_mid
            else:
                n_max = n_mid

        return n_min

    def get_corrected_results(
        self,
        results: list[dict[str, Any]],
        metric_key: str = "sharpe_ratio",
        pvalue_key: str = "p_value",
    ) -> list[dict[str, Any]]:
        """Apply corrections to a list of backtest results.

        Args:
            results: List of result dictionaries.
            metric_key: Key for the performance metric.
            pvalue_key: Key for p-values (will be added if using DSR).

        Returns:
            Results with added correction fields.
        """
        if not results:
            return results

        # Extract p-values if available
        if pvalue_key in results[0]:
            p_values = np.array([r[pvalue_key] for r in results])
        else:
            # Generate p-values from Sharpe ratios using DSR
            p_values = np.array([
                self.deflated_sharpe_ratio(r[metric_key])[1]
                for r in results
            ])

        # Apply BH correction
        adjusted_p, is_significant = self.benjamini_hochberg(p_values)

        # Add corrections to results
        corrected_results = []
        for i, result in enumerate(results):
            r = result.copy()
            r["p_value_raw"] = p_values[i]
            r["p_value_bonferroni"] = self.bonferroni(p_values[i])
            r["p_value_bh"] = adjusted_p[i]
            r["is_significant_bonferroni"] = self.is_significant_bonferroni(p_values[i])
            r["is_significant_bh"] = is_significant[i]
            corrected_results.append(r)

        return corrected_results


# =============================================================================
# Utility Functions
# =============================================================================


def validate_no_leakage(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    times: pd.Series | pd.DatetimeIndex,
) -> tuple[bool, str]:
    """Validate that there's no temporal leakage between train and test.

    Args:
        train_indices: Training sample indices.
        test_indices: Test sample indices.
        times: Timestamp for each sample.

    Returns:
        Tuple of (is_valid, message).
    """
    if len(train_indices) == 0 or len(test_indices) == 0:
        return False, "Empty train or test set"

    if isinstance(times, pd.DatetimeIndex):
        time_values = times
    else:
        time_values = pd.DatetimeIndex(times)

    train_times = time_values[train_indices]
    test_times = time_values[test_indices]

    train_max = train_times.max()
    test_min = test_times.min()

    if train_max >= test_min:
        return False, f"Train max time ({train_max}) >= Test min time ({test_min})"

    return True, "No temporal leakage detected"


def create_purged_cv(
    cv_type: str = "purged_kfold",
    n_splits: int = 5,
    **kwargs: Any,
) -> BaseCrossValidator:
    """Factory function to create purged cross-validator.

    Args:
        cv_type: Type of CV ("purged_kfold", "combinatorial", "walk_forward", "event").
        n_splits: Number of splits.
        **kwargs: Additional parameters for specific CV type.

    Returns:
        Configured cross-validator instance.
    """
    if cv_type == "purged_kfold":
        return PurgedKFold(n_splits=n_splits, **kwargs)
    elif cv_type == "combinatorial":
        return CombinatorialPurgedKFold(n_splits=n_splits, **kwargs)
    elif cv_type == "walk_forward":
        return WalkForwardCV(n_splits=n_splits, **kwargs)
    elif cv_type == "event":
        return EventPurgedKFold(n_splits=n_splits, **kwargs)
    else:
        raise ValueError(f"Unknown CV type: {cv_type}")


def create_multiple_testing_corrector(
    n_trials: int,
    significance_level: float = 0.05,
) -> MultipleTestingCorrector:
    """Factory function to create multiple testing corrector.

    Args:
        n_trials: Number of trials/configurations tested.
        significance_level: Base significance level. Default 0.05.

    Returns:
        Configured MultipleTestingCorrector instance.
    """
    return MultipleTestingCorrector(
        n_trials=n_trials,
        significance_level=significance_level,
    )
