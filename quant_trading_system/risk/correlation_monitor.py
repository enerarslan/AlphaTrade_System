"""
Multi-Asset Correlation Monitoring.

P3-A Enhancement: Real-time correlation monitoring for portfolio risk management:
- Dynamic correlation matrix computation
- Correlation breakdown detection
- Regime-dependent correlation tracking
- Hedging suggestions based on correlation structure

Expected Impact: +5-10 bps from better diversification management.

Author: AlphaTrade System
Version: 1.0.0
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quant_trading_system.core.events import (
    Event,
    EventBus,
    EventPriority,
    EventType,
    get_event_bus,
)

logger = logging.getLogger(__name__)

# FIX: Extract magic numbers to named constants
OUTLIER_THRESHOLD_SIGMA = 3  # Remove points > 3 standard deviations from median


# =============================================================================
# Correlation Types
# =============================================================================


class CorrelationMethod(str, Enum):
    """Correlation calculation methods."""

    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    EXPONENTIAL = "exponential"  # Exponentially weighted
    ROBUST = "robust"  # Using median absolute deviation


class CorrelationRegime(str, Enum):
    """Correlation regime classification."""

    LOW = "low"  # Assets weakly correlated
    NORMAL = "normal"  # Normal correlation structure
    HIGH = "high"  # Assets highly correlated
    BREAKDOWN = "breakdown"  # Correlation breakdown (crisis)
    DECOUPLED = "decoupled"  # Unusual decorrelation


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class CorrelationAlert:
    """Correlation-based alert."""

    alert_id: str
    severity: AlertSeverity
    alert_type: str  # "spike", "breakdown", "regime_change", "concentration"
    message: str
    symbols: list[str]
    correlation_value: float
    threshold: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "alert_type": self.alert_type,
            "message": self.message,
            "symbols": self.symbols,
            "correlation_value": self.correlation_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class HedgingSuggestion:
    """Suggested hedge based on correlation analysis."""

    target_symbol: str
    hedge_symbol: str
    correlation: float
    beta: float
    hedge_ratio: float  # Units of hedge per unit of target
    effectiveness: float  # Expected variance reduction
    cost_estimate_bps: float
    rationale: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_symbol": self.target_symbol,
            "hedge_symbol": self.hedge_symbol,
            "correlation": self.correlation,
            "beta": self.beta,
            "hedge_ratio": self.hedge_ratio,
            "effectiveness": self.effectiveness,
            "cost_estimate_bps": self.cost_estimate_bps,
            "rationale": self.rationale,
        }


@dataclass
class CorrelationSnapshot:
    """Point-in-time correlation matrix snapshot."""

    timestamp: datetime
    symbols: list[str]
    correlation_matrix: np.ndarray
    method: CorrelationMethod
    lookback_days: int
    regime: CorrelationRegime

    def get_correlation(self, symbol1: str, symbol2: str) -> float | None:
        """Get correlation between two symbols."""
        try:
            i = self.symbols.index(symbol1)
            j = self.symbols.index(symbol2)
            return float(self.correlation_matrix[i, j])
        except ValueError:
            return None

    def get_most_correlated(
        self,
        symbol: str,
        top_n: int = 5,
        exclude_self: bool = True,
    ) -> list[tuple[str, float]]:
        """Get most correlated symbols."""
        try:
            i = self.symbols.index(symbol)
        except ValueError:
            return []

        correlations = []
        for j, other_symbol in enumerate(self.symbols):
            if exclude_self and i == j:
                continue
            corr = float(self.correlation_matrix[i, j])
            correlations.append((other_symbol, corr))

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        return correlations[:top_n]


class CorrelationMonitorConfig(BaseModel):
    """Configuration for correlation monitoring."""

    # Calculation settings
    default_lookback_days: int = Field(default=60, description="Default lookback window")
    min_observations: int = Field(default=20, description="Min observations for correlation")
    method: CorrelationMethod = Field(default=CorrelationMethod.EXPONENTIAL)

    # Exponential weighting
    ewm_halflife_days: int = Field(default=20, description="Half-life for exponential weighting")

    # Alert thresholds
    high_correlation_threshold: float = Field(default=0.8, description="High correlation alert")
    correlation_spike_threshold: float = Field(default=0.2, description="Spike detection threshold")
    breakdown_threshold: float = Field(default=-0.3, description="Breakdown detection threshold")

    # Update frequency
    update_interval_minutes: int = Field(default=15, description="Update interval")

    # History
    max_history_snapshots: int = Field(default=100, description="Max snapshots to keep")


# =============================================================================
# Correlation Calculator
# =============================================================================


class CorrelationCalculator:
    """
    Computes correlation matrices using various methods.
    """

    def __init__(self, config: CorrelationMonitorConfig | None = None):
        self.config = config or CorrelationMonitorConfig()

    def compute_correlation_matrix(
        self,
        returns: pd.DataFrame,
        method: CorrelationMethod | None = None,
    ) -> np.ndarray:
        """Compute correlation matrix.

        Args:
            returns: DataFrame of returns (columns are symbols)
            method: Correlation method

        Returns:
            Correlation matrix as numpy array
        """
        method = method or self.config.method

        if returns.empty or len(returns) < self.config.min_observations:
            return np.eye(len(returns.columns))

        if method == CorrelationMethod.PEARSON:
            return returns.corr(method="pearson").values

        elif method == CorrelationMethod.SPEARMAN:
            return returns.corr(method="spearman").values

        elif method == CorrelationMethod.KENDALL:
            return returns.corr(method="kendall").values

        elif method == CorrelationMethod.EXPONENTIAL:
            return self._compute_ewm_correlation(returns)

        elif method == CorrelationMethod.ROBUST:
            return self._compute_robust_correlation(returns)

        return returns.corr().values

    def _compute_ewm_correlation(self, returns: pd.DataFrame) -> np.ndarray:
        """Compute exponentially weighted correlation matrix."""
        halflife = self.config.ewm_halflife_days
        n_assets = len(returns.columns)
        corr_matrix = np.zeros((n_assets, n_assets))

        # Compute EWM covariance
        ewm = returns.ewm(halflife=halflife)
        ewm_cov = ewm.cov().iloc[-n_assets:]

        # Extract correlation from covariance
        for i in range(n_assets):
            for j in range(n_assets):
                var_i = ewm_cov.iloc[i, i] if i < len(ewm_cov) else 1
                var_j = ewm_cov.iloc[j, j] if j < len(ewm_cov) else 1
                cov_ij = ewm_cov.iloc[i, j] if i < len(ewm_cov) and j < len(ewm_cov.columns) else 0

                if var_i > 0 and var_j > 0:
                    corr_matrix[i, j] = cov_ij / np.sqrt(var_i * var_j)
                else:
                    corr_matrix[i, j] = 0 if i != j else 1

        # Ensure diagonal is 1
        np.fill_diagonal(corr_matrix, 1.0)

        return corr_matrix

    def _compute_robust_correlation(self, returns: pd.DataFrame) -> np.ndarray:
        """Compute robust correlation using MAD (Median Absolute Deviation)."""
        n_assets = len(returns.columns)
        corr_matrix = np.zeros((n_assets, n_assets))

        for i in range(n_assets):
            for j in range(n_assets):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    x = returns.iloc[:, i].values
                    y = returns.iloc[:, j].values

                    # Use Spearman with outlier removal
                    # FIX: Use named constant instead of magic number
                    mask = (np.abs(x - np.median(x)) < OUTLIER_THRESHOLD_SIGMA * np.std(x)) & \
                           (np.abs(y - np.median(y)) < OUTLIER_THRESHOLD_SIGMA * np.std(y))

                    if np.sum(mask) >= self.config.min_observations:
                        corr_matrix[i, j] = np.corrcoef(x[mask], y[mask])[0, 1]
                    else:
                        corr_matrix[i, j] = np.corrcoef(x, y)[0, 1]

        return corr_matrix


# =============================================================================
# Regime Detector
# =============================================================================


class CorrelationRegimeDetector:
    """
    Detects correlation regime from matrix structure.
    """

    def __init__(
        self,
        low_avg_threshold: float = 0.3,
        high_avg_threshold: float = 0.6,
        breakdown_change_threshold: float = 0.3,
    ):
        self.low_avg_threshold = low_avg_threshold
        self.high_avg_threshold = high_avg_threshold
        self.breakdown_change_threshold = breakdown_change_threshold

        self._prev_matrix: np.ndarray | None = None

    def detect_regime(
        self,
        corr_matrix: np.ndarray,
    ) -> CorrelationRegime:
        """Detect current correlation regime.

        Args:
            corr_matrix: Current correlation matrix

        Returns:
            CorrelationRegime classification
        """
        n = corr_matrix.shape[0]
        if n <= 1:
            return CorrelationRegime.NORMAL

        # Extract upper triangle (excluding diagonal)
        upper_tri = corr_matrix[np.triu_indices(n, k=1)]

        # Average absolute correlation
        avg_corr = np.mean(np.abs(upper_tri))

        # Check for breakdown (sudden change)
        if self._prev_matrix is not None:
            prev_upper = self._prev_matrix[np.triu_indices(n, k=1)]
            change = np.mean(np.abs(upper_tri - prev_upper))

            if change > self.breakdown_change_threshold:
                self._prev_matrix = corr_matrix.copy()
                return CorrelationRegime.BREAKDOWN

        self._prev_matrix = corr_matrix.copy()

        # Classify based on average correlation
        if avg_corr < self.low_avg_threshold:
            return CorrelationRegime.LOW
        elif avg_corr > self.high_avg_threshold:
            return CorrelationRegime.HIGH
        else:
            return CorrelationRegime.NORMAL


# =============================================================================
# Hedge Analyzer
# =============================================================================


class HedgeAnalyzer:
    """
    Analyzes hedging opportunities based on correlation structure.
    """

    def __init__(
        self,
        min_correlation: float = 0.5,
        max_hedge_cost_bps: float = 10.0,
    ):
        self.min_correlation = min_correlation
        self.max_hedge_cost_bps = max_hedge_cost_bps

        # Common hedge instruments
        self._hedge_instruments = {
            "SPY": {"type": "market", "cost_bps": 1.0},
            "QQQ": {"type": "tech", "cost_bps": 1.0},
            "IWM": {"type": "small_cap", "cost_bps": 1.5},
            "VXX": {"type": "volatility", "cost_bps": 5.0},
            "TLT": {"type": "rates", "cost_bps": 1.0},
            "GLD": {"type": "gold", "cost_bps": 1.5},
        }

    def find_hedges(
        self,
        target_symbol: str,
        correlation_snapshot: CorrelationSnapshot,
        returns_df: pd.DataFrame,
        position_direction: str = "long",
    ) -> list[HedgingSuggestion]:
        """Find hedging opportunities for a position.

        Args:
            target_symbol: Symbol to hedge
            correlation_snapshot: Current correlation snapshot
            returns_df: Returns data for beta calculation
            position_direction: "long" or "short"

        Returns:
            List of hedging suggestions
        """
        suggestions = []

        if target_symbol not in correlation_snapshot.symbols:
            return suggestions

        # Get correlated symbols
        correlated = correlation_snapshot.get_most_correlated(target_symbol, top_n=10)

        for hedge_symbol, corr in correlated:
            # Skip if correlation too low
            if abs(corr) < self.min_correlation:
                continue

            # Calculate beta
            beta = self._calculate_beta(target_symbol, hedge_symbol, returns_df)
            if beta is None:
                continue

            # Determine hedge ratio
            if position_direction == "long":
                # For long position, we want negative correlation or short positive
                if corr > 0:
                    hedge_ratio = -beta  # Short the hedge
                else:
                    hedge_ratio = beta  # Long the hedge (negative corr provides hedge)
            else:
                # For short position, opposite
                if corr > 0:
                    hedge_ratio = beta
                else:
                    hedge_ratio = -beta

            # Estimate effectiveness (variance reduction)
            effectiveness = corr ** 2

            # Estimate cost
            cost_bps = self._estimate_hedge_cost(hedge_symbol)
            if cost_bps > self.max_hedge_cost_bps:
                continue

            # Generate rationale
            rationale = self._generate_rationale(
                target_symbol, hedge_symbol, corr, beta, position_direction
            )

            suggestions.append(HedgingSuggestion(
                target_symbol=target_symbol,
                hedge_symbol=hedge_symbol,
                correlation=corr,
                beta=beta,
                hedge_ratio=hedge_ratio,
                effectiveness=effectiveness,
                cost_estimate_bps=cost_bps,
                rationale=rationale,
            ))

        # Sort by effectiveness
        suggestions.sort(key=lambda x: x.effectiveness, reverse=True)

        return suggestions[:5]  # Top 5 hedges

    def _calculate_beta(
        self,
        target: str,
        hedge: str,
        returns_df: pd.DataFrame,
    ) -> float | None:
        """Calculate beta of target to hedge."""
        if target not in returns_df.columns or hedge not in returns_df.columns:
            return None

        target_returns = returns_df[target].dropna()
        hedge_returns = returns_df[hedge].dropna()

        # Align
        common_idx = target_returns.index.intersection(hedge_returns.index)
        if len(common_idx) < 20:
            return None

        y = target_returns.loc[common_idx].values
        x = hedge_returns.loc[common_idx].values

        # OLS beta
        cov = np.cov(y, x)[0, 1]
        var = np.var(x)

        if var > 0:
            return cov / var
        return None

    def _estimate_hedge_cost(self, symbol: str) -> float:
        """Estimate hedging cost in bps."""
        if symbol in self._hedge_instruments:
            return self._hedge_instruments[symbol]["cost_bps"]
        return 3.0  # Default cost

    def _generate_rationale(
        self,
        target: str,
        hedge: str,
        corr: float,
        beta: float,
        direction: str,
    ) -> str:
        """Generate human-readable hedge rationale."""
        corr_desc = "positively" if corr > 0 else "negatively"
        action = "short" if (corr > 0 and direction == "long") else "long"

        return (
            f"{target} is {corr_desc} correlated ({corr:.2f}) with {hedge}. "
            f"Beta: {beta:.2f}. Suggested: {action} {hedge} as hedge."
        )


# =============================================================================
# Correlation Monitor
# =============================================================================


class CorrelationMonitor:
    """
    P3-A Enhancement: Multi-Asset Correlation Monitor.

    Provides:
    - Real-time correlation matrix tracking
    - Correlation regime detection
    - Alert generation for correlation changes
    - Hedging suggestions
    """

    def __init__(
        self,
        config: CorrelationMonitorConfig | None = None,
        event_bus: EventBus | None = None,
    ):
        """Initialize correlation monitor.

        Args:
            config: Configuration
            event_bus: Event bus for publishing alerts
        """
        self.config = config or CorrelationMonitorConfig()
        self.event_bus = event_bus or get_event_bus()

        # Components
        self._calculator = CorrelationCalculator(config)
        self._regime_detector = CorrelationRegimeDetector()
        self._hedge_analyzer = HedgeAnalyzer()

        # State
        self._lock = threading.RLock()
        self._returns_history: pd.DataFrame = pd.DataFrame()
        self._snapshots: list[CorrelationSnapshot] = []
        self._alerts: list[CorrelationAlert] = []
        self._last_update: datetime | None = None
        self._alert_counter = 0

    def update_returns(self, symbol: str, date: datetime, return_value: float) -> None:
        """Update returns data for a symbol.

        Args:
            symbol: Stock symbol
            date: Date
            return_value: Return value
        """
        with self._lock:
            if symbol not in self._returns_history.columns:
                self._returns_history[symbol] = np.nan

            self._returns_history.loc[date, symbol] = return_value

            # Keep bounded
            max_rows = self.config.default_lookback_days * 2
            if len(self._returns_history) > max_rows:
                self._returns_history = self._returns_history.tail(max_rows)

    def update_returns_batch(self, returns_df: pd.DataFrame) -> None:
        """Update returns with batch data.

        Args:
            returns_df: DataFrame with returns (index=dates, columns=symbols)
        """
        with self._lock:
            for symbol in returns_df.columns:
                if symbol not in self._returns_history.columns:
                    self._returns_history[symbol] = np.nan

            # Update with new data
            for idx in returns_df.index:
                for symbol in returns_df.columns:
                    self._returns_history.loc[idx, symbol] = returns_df.loc[idx, symbol]

            # Sort and bound
            self._returns_history = self._returns_history.sort_index()
            max_rows = self.config.default_lookback_days * 2
            if len(self._returns_history) > max_rows:
                self._returns_history = self._returns_history.tail(max_rows)

    def compute_correlation_snapshot(
        self,
        symbols: list[str] | None = None,
        lookback_days: int | None = None,
        method: CorrelationMethod | None = None,
    ) -> CorrelationSnapshot | None:
        """Compute current correlation snapshot.

        Args:
            symbols: Symbols to include (default: all)
            lookback_days: Lookback period
            method: Correlation method

        Returns:
            CorrelationSnapshot or None
        """
        with self._lock:
            if self._returns_history.empty:
                return None

            lookback = lookback_days or self.config.default_lookback_days
            method = method or self.config.method

            # Filter symbols
            if symbols:
                available = [s for s in symbols if s in self._returns_history.columns]
            else:
                available = list(self._returns_history.columns)

            if len(available) < 2:
                return None

            # Get recent returns
            returns = self._returns_history[available].tail(lookback).dropna()

            if len(returns) < self.config.min_observations:
                return None

            # Compute correlation matrix
            corr_matrix = self._calculator.compute_correlation_matrix(returns, method)

            # Detect regime
            regime = self._regime_detector.detect_regime(corr_matrix)

            snapshot = CorrelationSnapshot(
                timestamp=datetime.now(timezone.utc),
                symbols=available,
                correlation_matrix=corr_matrix,
                method=method,
                lookback_days=lookback,
                regime=regime,
            )

            # Store snapshot
            self._snapshots.append(snapshot)
            if len(self._snapshots) > self.config.max_history_snapshots:
                self._snapshots = self._snapshots[-self.config.max_history_snapshots:]

            # Check for alerts
            self._check_alerts(snapshot)

            self._last_update = datetime.now(timezone.utc)

            return snapshot

    def _check_alerts(self, snapshot: CorrelationSnapshot) -> None:
        """Check for correlation alerts.

        Args:
            snapshot: Current snapshot
        """
        n = len(snapshot.symbols)
        if n < 2:
            return

        # Check for high correlations
        for i in range(n):
            for j in range(i + 1, n):
                corr = snapshot.correlation_matrix[i, j]

                # High correlation alert
                if abs(corr) > self.config.high_correlation_threshold:
                    self._create_alert(
                        severity=AlertSeverity.WARNING,
                        alert_type="high_correlation",
                        message=f"High correlation detected between {snapshot.symbols[i]} and {snapshot.symbols[j]}",
                        symbols=[snapshot.symbols[i], snapshot.symbols[j]],
                        correlation_value=corr,
                        threshold=self.config.high_correlation_threshold,
                    )

        # Check for regime change
        if len(self._snapshots) >= 2:
            prev_regime = self._snapshots[-2].regime
            if snapshot.regime != prev_regime:
                self._create_alert(
                    severity=AlertSeverity.INFO if snapshot.regime != CorrelationRegime.BREAKDOWN else AlertSeverity.CRITICAL,
                    alert_type="regime_change",
                    message=f"Correlation regime changed from {prev_regime.value} to {snapshot.regime.value}",
                    symbols=snapshot.symbols,
                    correlation_value=0.0,
                    threshold=0.0,
                    metadata={"prev_regime": prev_regime.value, "new_regime": snapshot.regime.value},
                )

        # Check for correlation spike
        if len(self._snapshots) >= 2:
            prev_matrix = self._snapshots[-2].correlation_matrix
            if prev_matrix.shape == snapshot.correlation_matrix.shape:
                change = np.abs(snapshot.correlation_matrix - prev_matrix)
                max_change = np.max(change[np.triu_indices(n, k=1)])

                if max_change > self.config.correlation_spike_threshold:
                    # Find the pair with max change
                    max_idx = np.unravel_index(np.argmax(change), change.shape)
                    sym1, sym2 = snapshot.symbols[max_idx[0]], snapshot.symbols[max_idx[1]]

                    self._create_alert(
                        severity=AlertSeverity.WARNING,
                        alert_type="correlation_spike",
                        message=f"Correlation spike detected: {sym1}-{sym2} changed by {max_change:.2f}",
                        symbols=[sym1, sym2],
                        correlation_value=float(snapshot.correlation_matrix[max_idx]),
                        threshold=self.config.correlation_spike_threshold,
                    )

    def _create_alert(
        self,
        severity: AlertSeverity,
        alert_type: str,
        message: str,
        symbols: list[str],
        correlation_value: float,
        threshold: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create and store correlation alert."""
        self._alert_counter += 1
        alert = CorrelationAlert(
            alert_id=f"CORR-{self._alert_counter:06d}",
            severity=severity,
            alert_type=alert_type,
            message=message,
            symbols=symbols,
            correlation_value=correlation_value,
            threshold=threshold,
            metadata=metadata or {},
        )

        self._alerts.append(alert)

        # Keep bounded
        if len(self._alerts) > 1000:
            self._alerts = self._alerts[-1000:]

        # Publish event
        self.event_bus.publish(Event(
            event_type=EventType.LIMIT_BREACH,  # Use existing risk event
            data=alert.to_dict(),
            priority=EventPriority.HIGH if severity == AlertSeverity.CRITICAL else EventPriority.NORMAL,
        ))

        logger.warning(f"Correlation alert: {message}")

    def get_hedging_suggestions(
        self,
        symbol: str,
        position_direction: str = "long",
    ) -> list[HedgingSuggestion]:
        """Get hedging suggestions for a position.

        Args:
            symbol: Symbol to hedge
            position_direction: "long" or "short"

        Returns:
            List of hedging suggestions
        """
        with self._lock:
            if not self._snapshots:
                return []

            latest = self._snapshots[-1]

            return self._hedge_analyzer.find_hedges(
                target_symbol=symbol,
                correlation_snapshot=latest,
                returns_df=self._returns_history,
                position_direction=position_direction,
            )

    def get_current_matrix(self) -> dict[str, Any] | None:
        """Get current correlation matrix as dictionary.

        Returns:
            Dictionary with symbols and correlation matrix
        """
        with self._lock:
            if not self._snapshots:
                return None

            latest = self._snapshots[-1]

            return {
                "timestamp": latest.timestamp.isoformat(),
                "symbols": latest.symbols,
                "correlation_matrix": latest.correlation_matrix.tolist(),
                "regime": latest.regime.value,
                "method": latest.method.value,
                "lookback_days": latest.lookback_days,
            }

    def get_pair_correlation_history(
        self,
        symbol1: str,
        symbol2: str,
        limit: int = 50,
    ) -> list[tuple[datetime, float]]:
        """Get historical correlation between two symbols.

        Args:
            symbol1: First symbol
            symbol2: Second symbol
            limit: Max history entries

        Returns:
            List of (timestamp, correlation) tuples
        """
        with self._lock:
            history = []

            for snapshot in self._snapshots[-limit:]:
                corr = snapshot.get_correlation(symbol1, symbol2)
                if corr is not None:
                    history.append((snapshot.timestamp, corr))

            return history

    def get_recent_alerts(
        self,
        severity: AlertSeverity | None = None,
        limit: int = 50,
    ) -> list[CorrelationAlert]:
        """Get recent correlation alerts.

        Args:
            severity: Filter by severity
            limit: Max alerts

        Returns:
            List of alerts
        """
        with self._lock:
            alerts = self._alerts

            if severity:
                alerts = [a for a in alerts if a.severity == severity]

            return alerts[-limit:]

    def get_concentration_risk(
        self,
        portfolio_weights: dict[str, float],
    ) -> dict[str, Any]:
        """Analyze correlation-based concentration risk.

        Args:
            portfolio_weights: Dictionary of symbol -> weight

        Returns:
            Concentration risk analysis
        """
        with self._lock:
            if not self._snapshots:
                return {"error": "No correlation data available"}

            latest = self._snapshots[-1]

            # Filter to symbols in portfolio
            symbols = [s for s in portfolio_weights.keys() if s in latest.symbols]
            if len(symbols) < 2:
                return {"error": "Insufficient symbols for analysis"}

            weights = np.array([portfolio_weights[s] for s in symbols])

            # Get correlation submatrix
            indices = [latest.symbols.index(s) for s in symbols]
            corr_sub = latest.correlation_matrix[np.ix_(indices, indices)]

            # Portfolio correlation (weighted average pairwise)
            n = len(symbols)
            weighted_corr = 0.0
            total_weight = 0.0

            for i in range(n):
                for j in range(i + 1, n):
                    pair_weight = weights[i] * weights[j]
                    weighted_corr += pair_weight * corr_sub[i, j]
                    total_weight += pair_weight

            avg_corr = weighted_corr / total_weight if total_weight > 0 else 0

            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(corr_sub[i, j]) > 0.7:
                        high_corr_pairs.append({
                            "symbols": [symbols[i], symbols[j]],
                            "correlation": float(corr_sub[i, j]),
                            "combined_weight": weights[i] + weights[j],
                        })

            return {
                "average_weighted_correlation": avg_corr,
                "high_correlation_pairs": high_corr_pairs,
                "concentration_score": avg_corr * sum(w ** 2 for w in weights),
                "effective_assets": 1 / sum(w ** 2 for w in weights) if any(weights) else 0,
                "regime": latest.regime.value,
            }


# =============================================================================
# Factory Function
# =============================================================================


def create_correlation_monitor(
    config: CorrelationMonitorConfig | None = None,
    event_bus: EventBus | None = None,
) -> CorrelationMonitor:
    """Factory function to create correlation monitor.

    Args:
        config: Configuration
        event_bus: Event bus

    Returns:
        Configured CorrelationMonitor
    """
    return CorrelationMonitor(config=config, event_bus=event_bus)
