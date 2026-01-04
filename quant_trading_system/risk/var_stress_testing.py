"""
Intraday Value-at-Risk (VaR) and Stress Testing Framework.

P2-2.1a & P2-2.1b Enhancement: Implements real-time VaR calculation and
comprehensive stress testing scenarios for institutional risk management.

Key concepts:
- Parametric VaR (variance-covariance method)
- Historical VaR (simulation from historical returns)
- Monte Carlo VaR
- Stress testing with predefined scenarios (2008, 2020, custom)
- Conditional VaR (Expected Shortfall / CVaR)

Benefits:
- Real-time risk monitoring
- Regulatory compliance (Basel III/IV)
- Scenario-based risk assessment
- Early warning for potential losses

Author: AlphaTrade System
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class VaRMethod(str, Enum):
    """VaR calculation methods."""

    PARAMETRIC = "parametric"  # Variance-covariance
    HISTORICAL = "historical"  # Historical simulation
    MONTE_CARLO = "monte_carlo"  # Monte Carlo simulation
    CORNISH_FISHER = "cornish_fisher"  # Adjusted for skew/kurtosis


class StressScenario(str, Enum):
    """Predefined stress scenarios."""

    GFC_2008 = "gfc_2008"  # Global Financial Crisis
    COVID_2020 = "covid_2020"  # COVID-19 crash
    FLASH_CRASH = "flash_crash"  # Flash crash scenario
    RATE_SHOCK = "rate_shock"  # Interest rate shock
    TECH_BUBBLE = "tech_bubble"  # Tech bubble burst
    CUSTOM = "custom"


@dataclass
class VaRConfig:
    """Configuration for VaR calculation."""

    # Confidence levels
    confidence_levels: list[float] = field(default_factory=lambda: [0.95, 0.99])

    # Calculation settings
    default_method: VaRMethod = VaRMethod.HISTORICAL
    lookback_days: int = 252  # 1 year of trading days
    decay_factor: float = 0.94  # EWMA decay factor

    # Monte Carlo settings
    mc_simulations: int = 10000
    mc_horizon_days: int = 1

    # Intraday settings
    intraday_scaling: bool = True
    trading_hours_per_day: float = 6.5  # NYSE hours

    # Risk limits
    var_limit_pct: float = 0.05  # Max VaR as % of equity
    cvar_limit_pct: float = 0.08  # Max CVaR as % of equity


@dataclass
class StressTestConfig:
    """Configuration for stress testing."""

    # Predefined scenario shocks (% returns)
    scenarios: dict[StressScenario, dict[str, float]] = field(default_factory=lambda: {
        StressScenario.GFC_2008: {
            "equity": -0.50,  # 50% equity drop
            "volatility_mult": 4.0,  # VIX spike to 80+
            "correlation_spike": 0.9,  # Correlations go to 1
            "liquidity_factor": 0.3,  # 70% liquidity reduction
        },
        StressScenario.COVID_2020: {
            "equity": -0.35,  # 35% drop in weeks
            "volatility_mult": 5.0,  # VIX spike to 80+
            "correlation_spike": 0.85,
            "liquidity_factor": 0.5,
        },
        StressScenario.FLASH_CRASH: {
            "equity": -0.10,  # 10% intraday
            "volatility_mult": 3.0,
            "correlation_spike": 0.95,
            "liquidity_factor": 0.1,  # Near zero liquidity
        },
        StressScenario.RATE_SHOCK: {
            "equity": -0.15,
            "bond": -0.10,
            "volatility_mult": 2.0,
            "correlation_spike": 0.7,
            "liquidity_factor": 0.6,
        },
        StressScenario.TECH_BUBBLE: {
            "tech": -0.70,  # Tech sector specific
            "equity": -0.30,
            "volatility_mult": 2.5,
            "correlation_spike": 0.6,
            "liquidity_factor": 0.5,
        },
    })

    # Custom scenario settings
    custom_equity_shock: float = -0.20
    custom_vol_mult: float = 2.0


@dataclass
class VaRResult:
    """Result of VaR calculation."""

    timestamp: datetime
    method: VaRMethod
    confidence_level: float
    var_amount: Decimal  # Absolute VaR in dollars
    var_pct: float  # VaR as % of portfolio
    cvar_amount: Decimal  # Conditional VaR (Expected Shortfall)
    cvar_pct: float
    portfolio_value: Decimal
    horizon_days: int
    breached_limit: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "method": self.method.value,
            "confidence_level": self.confidence_level,
            "var_amount": float(self.var_amount),
            "var_pct": self.var_pct,
            "cvar_amount": float(self.cvar_amount),
            "cvar_pct": self.cvar_pct,
            "portfolio_value": float(self.portfolio_value),
            "horizon_days": self.horizon_days,
            "breached_limit": self.breached_limit,
        }


@dataclass
class StressTestResult:
    """Result of stress test."""

    scenario: StressScenario
    scenario_name: str
    portfolio_value_before: Decimal
    portfolio_value_after: Decimal
    loss_amount: Decimal
    loss_pct: float
    positions_impacted: dict[str, float]  # symbol -> loss
    recovery_estimate_days: int | None
    severity: str  # "low", "medium", "high", "extreme"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario": self.scenario.value,
            "scenario_name": self.scenario_name,
            "portfolio_value_before": float(self.portfolio_value_before),
            "portfolio_value_after": float(self.portfolio_value_after),
            "loss_amount": float(self.loss_amount),
            "loss_pct": self.loss_pct,
            "positions_impacted": self.positions_impacted,
            "recovery_estimate_days": self.recovery_estimate_days,
            "severity": self.severity,
        }


class IntradayVaRCalculator:
    """
    P2-2.1a Enhancement: Intraday Value-at-Risk Calculator.

    Provides real-time VaR calculation using multiple methods:
    - Parametric (fastest, assumes normality)
    - Historical simulation (most intuitive)
    - Monte Carlo (most flexible)
    - Cornish-Fisher (adjusts for fat tails)

    Example:
        >>> calculator = IntradayVaRCalculator(config)
        >>> result = calculator.calculate_var(
        ...     portfolio_value=Decimal("1000000"),
        ...     positions=positions,
        ...     returns_history=returns_df,
        ...     confidence=0.99
        ... )
        >>> print(f"99% VaR: ${result.var_amount}")
    """

    def __init__(self, config: VaRConfig | None = None):
        """Initialize VaR Calculator.

        Args:
            config: VaR configuration.
        """
        self.config = config or VaRConfig()
        self._returns_cache: pd.DataFrame | None = None

        logger.info(
            f"IntradayVaRCalculator initialized: "
            f"method={self.config.default_method.value}, "
            f"lookback={self.config.lookback_days} days"
        )

    def calculate_var(
        self,
        portfolio_value: Decimal,
        positions: dict[str, Decimal],  # symbol -> position value
        returns_history: pd.DataFrame,
        confidence: float = 0.99,
        method: VaRMethod | None = None,
        horizon_days: int = 1,
        current_mtm_loss_pct: float | None = None,
    ) -> VaRResult:
        """Calculate Value-at-Risk with optional intraday adjustment.

        P1 FIX: Added current_mtm_loss_pct parameter to incorporate current
        unrealized losses into VaR calculation. This prevents VaR from being
        stale during intraday market crashes.

        Args:
            portfolio_value: Total portfolio value.
            positions: Dictionary of position values by symbol.
            returns_history: Historical returns DataFrame (columns = symbols).
            confidence: Confidence level (e.g., 0.99 for 99%).
            method: VaR calculation method.
            horizon_days: VaR horizon in days.
            current_mtm_loss_pct: P1 FIX - Current unrealized loss as percentage
                                  (e.g., -0.05 for a 5% intraday loss). If provided,
                                  this is added to the returns distribution.

        Returns:
            VaRResult with calculated VaR and CVaR.
        """
        method = method or self.config.default_method

        # Calculate portfolio returns
        weights = self._calculate_weights(positions, portfolio_value)
        portfolio_returns = self._calculate_portfolio_returns(returns_history, weights)

        # P1 FIX: Incorporate current intraday MTM loss into returns distribution
        # This ensures VaR reflects current market conditions, not just historical data
        if current_mtm_loss_pct is not None and current_mtm_loss_pct != 0:
            # Add current loss as an additional data point
            # This is critical during market crashes where historical data
            # doesn't reflect current conditions
            augmented_returns = pd.concat([
                portfolio_returns,
                pd.Series([current_mtm_loss_pct])
            ])
            portfolio_returns = augmented_returns
            logger.info(
                f"VaR calculation augmented with current MTM loss: {current_mtm_loss_pct:.2%}"
            )

        # Scale returns to horizon
        if horizon_days > 1:
            # Square root of time scaling
            portfolio_returns = portfolio_returns * np.sqrt(horizon_days)

        # Calculate VaR based on method
        if method == VaRMethod.PARAMETRIC:
            var_pct, cvar_pct = self._parametric_var(portfolio_returns, confidence)
        elif method == VaRMethod.HISTORICAL:
            var_pct, cvar_pct = self._historical_var(portfolio_returns, confidence)
        elif method == VaRMethod.MONTE_CARLO:
            var_pct, cvar_pct = self._monte_carlo_var(portfolio_returns, confidence)
        elif method == VaRMethod.CORNISH_FISHER:
            var_pct, cvar_pct = self._cornish_fisher_var(portfolio_returns, confidence)
        else:
            var_pct, cvar_pct = self._historical_var(portfolio_returns, confidence)

        # Convert to absolute values
        var_amount = Decimal(str(abs(var_pct) * float(portfolio_value)))
        cvar_amount = Decimal(str(abs(cvar_pct) * float(portfolio_value)))

        # Check limits
        breached = abs(var_pct) > self.config.var_limit_pct

        return VaRResult(
            timestamp=datetime.now(timezone.utc),
            method=method,
            confidence_level=confidence,
            var_amount=var_amount,
            var_pct=abs(var_pct),
            cvar_amount=cvar_amount,
            cvar_pct=abs(cvar_pct),
            portfolio_value=portfolio_value,
            horizon_days=horizon_days,
            breached_limit=breached,
        )

    def _calculate_weights(
        self,
        positions: dict[str, Decimal],
        portfolio_value: Decimal,
    ) -> dict[str, float]:
        """Calculate position weights."""
        weights = {}
        total = float(portfolio_value)
        if total <= 0:
            return weights

        for symbol, value in positions.items():
            weights[symbol] = float(value) / total

        return weights

    def _calculate_portfolio_returns(
        self,
        returns_history: pd.DataFrame,
        weights: dict[str, float],
    ) -> pd.Series:
        """Calculate weighted portfolio returns."""
        portfolio_returns = pd.Series(0.0, index=returns_history.index)

        for symbol, weight in weights.items():
            if symbol in returns_history.columns:
                portfolio_returns += returns_history[symbol] * weight

        return portfolio_returns.dropna()

    def _parametric_var(
        self,
        returns: pd.Series,
        confidence: float,
    ) -> tuple[float, float]:
        """Calculate parametric (variance-covariance) VaR.

        Assumes returns are normally distributed.

        Args:
            returns: Portfolio returns series.
            confidence: Confidence level.

        Returns:
            Tuple of (VaR, CVaR) as percentages.
        """
        mean = returns.mean()
        std = returns.std()

        # Z-score for confidence level
        z = stats.norm.ppf(1 - confidence)

        # VaR
        var = mean + z * std

        # CVaR (Expected Shortfall)
        cvar = mean - std * stats.norm.pdf(z) / (1 - confidence)

        return var, cvar

    def _historical_var(
        self,
        returns: pd.Series,
        confidence: float,
    ) -> tuple[float, float]:
        """Calculate historical simulation VaR.

        Uses actual historical returns distribution.

        Args:
            returns: Portfolio returns series.
            confidence: Confidence level.

        Returns:
            Tuple of (VaR, CVaR) as percentages.
        """
        # VaR is the percentile of losses
        var = returns.quantile(1 - confidence)

        # CVaR is average of losses beyond VaR
        cvar = returns[returns <= var].mean()

        return var, cvar if not np.isnan(cvar) else var

    def _monte_carlo_var(
        self,
        returns: pd.Series,
        confidence: float,
    ) -> tuple[float, float]:
        """Calculate Monte Carlo VaR.

        Simulates future returns based on historical distribution.

        Args:
            returns: Portfolio returns series.
            confidence: Confidence level.

        Returns:
            Tuple of (VaR, CVaR) as percentages.
        """
        mean = returns.mean()
        std = returns.std()
        skew = returns.skew()
        kurt = returns.kurtosis()

        # Generate simulated returns
        np.random.seed(42)  # For reproducibility
        simulated = np.random.normal(mean, std, self.config.mc_simulations)

        # Sort for percentile calculation
        simulated_sorted = np.sort(simulated)

        # VaR
        var_idx = int((1 - confidence) * self.config.mc_simulations)
        var = simulated_sorted[var_idx]

        # CVaR
        cvar = simulated_sorted[:var_idx].mean() if var_idx > 0 else var

        return var, cvar

    def _cornish_fisher_var(
        self,
        returns: pd.Series,
        confidence: float,
    ) -> tuple[float, float]:
        """Calculate Cornish-Fisher VaR (adjusted for skewness/kurtosis).

        Better for fat-tailed distributions common in finance.

        Args:
            returns: Portfolio returns series.
            confidence: Confidence level.

        Returns:
            Tuple of (VaR, CVaR) as percentages.
        """
        mean = returns.mean()
        std = returns.std()
        skew = returns.skew()
        kurt = returns.kurtosis()

        # Standard normal quantile
        z = stats.norm.ppf(1 - confidence)

        # Cornish-Fisher expansion
        z_cf = (z + (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * (kurt - 3) / 24 -
                (2*z**3 - 5*z) * skew**2 / 36)

        var = mean + z_cf * std

        # Approximate CVaR
        cvar = mean + z_cf * std * 1.2  # Rough approximation

        return var, cvar

    def calculate_component_var(
        self,
        portfolio_value: Decimal,
        positions: dict[str, Decimal],
        returns_history: pd.DataFrame,
        confidence: float = 0.99,
    ) -> dict[str, VaRResult]:
        """Calculate component VaR for each position.

        Shows marginal VaR contribution of each position.

        Args:
            portfolio_value: Total portfolio value.
            positions: Position values by symbol.
            returns_history: Historical returns.
            confidence: Confidence level.

        Returns:
            Dictionary of symbol -> VaRResult.
        """
        results = {}

        for symbol, value in positions.items():
            single_position = {symbol: value}
            result = self.calculate_var(
                portfolio_value=value,
                positions=single_position,
                returns_history=returns_history[[symbol]] if symbol in returns_history.columns else returns_history,
                confidence=confidence,
            )
            results[symbol] = result

        return results


class StressTestFramework:
    """
    P2-2.1b Enhancement: Stress Testing Framework.

    Evaluates portfolio resilience under extreme market scenarios.
    Provides scenario-based analysis for risk management and
    regulatory compliance.

    Example:
        >>> framework = StressTestFramework(config)
        >>> results = framework.run_all_scenarios(
        ...     portfolio_value=Decimal("1000000"),
        ...     positions=positions
        ... )
        >>> for result in results:
        ...     print(f"{result.scenario_name}: {result.loss_pct:.1%} loss")
    """

    def __init__(self, config: StressTestConfig | None = None):
        """Initialize Stress Test Framework.

        Args:
            config: Stress test configuration.
        """
        self.config = config or StressTestConfig()

        logger.info(
            f"StressTestFramework initialized with "
            f"{len(self.config.scenarios)} predefined scenarios"
        )

    def run_scenario(
        self,
        scenario: StressScenario,
        portfolio_value: Decimal,
        positions: dict[str, Decimal],
        sector_exposure: dict[str, float] | None = None,
        custom_shocks: dict[str, float] | None = None,
    ) -> StressTestResult:
        """Run a single stress test scenario.

        Args:
            scenario: Stress scenario to run.
            portfolio_value: Current portfolio value.
            positions: Position values by symbol.
            sector_exposure: Sector exposures (optional).
            custom_shocks: Custom shock parameters (for CUSTOM scenario).

        Returns:
            StressTestResult with scenario impact.
        """
        if scenario == StressScenario.CUSTOM:
            shocks = custom_shocks or {
                "equity": self.config.custom_equity_shock,
                "volatility_mult": self.config.custom_vol_mult,
            }
        else:
            shocks = self.config.scenarios.get(scenario, {"equity": -0.20})

        # Calculate position impacts
        position_losses = {}
        total_loss = Decimal("0")

        equity_shock = shocks.get("equity", -0.20)
        tech_shock = shocks.get("tech", equity_shock)

        for symbol, value in positions.items():
            # Determine shock based on sector
            if sector_exposure and symbol in sector_exposure:
                sector = sector_exposure.get(symbol, "equity")
                if "tech" in str(sector).lower():
                    shock = tech_shock
                else:
                    shock = equity_shock
            else:
                shock = equity_shock

            loss = Decimal(str(float(value) * abs(shock)))
            position_losses[symbol] = float(loss)
            total_loss += loss

        # Calculate stressed portfolio value
        stressed_value = portfolio_value - total_loss
        loss_pct = float(total_loss) / float(portfolio_value) if float(portfolio_value) > 0 else 0

        # Estimate recovery time
        recovery_days = self._estimate_recovery(scenario, loss_pct)

        # Determine severity
        severity = self._classify_severity(loss_pct)

        return StressTestResult(
            scenario=scenario,
            scenario_name=self._get_scenario_name(scenario),
            portfolio_value_before=portfolio_value,
            portfolio_value_after=stressed_value,
            loss_amount=total_loss,
            loss_pct=loss_pct,
            positions_impacted=position_losses,
            recovery_estimate_days=recovery_days,
            severity=severity,
        )

    def run_all_scenarios(
        self,
        portfolio_value: Decimal,
        positions: dict[str, Decimal],
        sector_exposure: dict[str, float] | None = None,
    ) -> list[StressTestResult]:
        """Run all predefined stress scenarios.

        Args:
            portfolio_value: Current portfolio value.
            positions: Position values by symbol.
            sector_exposure: Sector exposures (optional).

        Returns:
            List of StressTestResult for each scenario.
        """
        results = []

        for scenario in self.config.scenarios.keys():
            result = self.run_scenario(
                scenario=scenario,
                portfolio_value=portfolio_value,
                positions=positions,
                sector_exposure=sector_exposure,
            )
            results.append(result)

        # Sort by severity
        severity_order = {"extreme": 0, "high": 1, "medium": 2, "low": 3}
        results.sort(key=lambda r: severity_order.get(r.severity, 4))

        return results

    def _get_scenario_name(self, scenario: StressScenario) -> str:
        """Get human-readable scenario name."""
        names = {
            StressScenario.GFC_2008: "2008 Global Financial Crisis",
            StressScenario.COVID_2020: "2020 COVID-19 Crash",
            StressScenario.FLASH_CRASH: "Flash Crash Scenario",
            StressScenario.RATE_SHOCK: "Interest Rate Shock",
            StressScenario.TECH_BUBBLE: "Tech Bubble Burst",
            StressScenario.CUSTOM: "Custom Scenario",
        }
        return names.get(scenario, scenario.value)

    def _estimate_recovery(
        self,
        scenario: StressScenario,
        loss_pct: float,
    ) -> int | None:
        """Estimate recovery time in trading days."""
        # Historical recovery times (approximate)
        recovery_times = {
            StressScenario.GFC_2008: 500,  # ~2 years
            StressScenario.COVID_2020: 150,  # ~6 months
            StressScenario.FLASH_CRASH: 5,  # Days
            StressScenario.RATE_SHOCK: 60,
            StressScenario.TECH_BUBBLE: 750,  # ~3 years
        }

        base_recovery = recovery_times.get(scenario, 250)

        # Scale by actual loss
        scaled = int(base_recovery * (loss_pct / 0.30))

        return max(5, min(scaled, 1000))

    def _classify_severity(self, loss_pct: float) -> str:
        """Classify loss severity."""
        if loss_pct >= 0.40:
            return "extreme"
        elif loss_pct >= 0.25:
            return "high"
        elif loss_pct >= 0.10:
            return "medium"
        else:
            return "low"

    def generate_report(
        self,
        results: list[StressTestResult],
    ) -> dict[str, Any]:
        """Generate stress test report.

        Args:
            results: List of stress test results.

        Returns:
            Report dictionary.
        """
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_scenarios": len(results),
            "worst_case": {
                "scenario": results[0].scenario_name if results else None,
                "loss_pct": results[0].loss_pct if results else 0,
                "severity": results[0].severity if results else None,
            },
            "average_loss_pct": np.mean([r.loss_pct for r in results]) if results else 0,
            "scenarios": [r.to_dict() for r in results],
        }


def create_var_calculator(
    confidence_levels: list[float] | None = None,
    method: VaRMethod = VaRMethod.HISTORICAL,
    lookback_days: int = 252,
    **kwargs: Any,
) -> IntradayVaRCalculator:
    """Factory function to create VaR calculator.

    Args:
        confidence_levels: VaR confidence levels.
        method: Default calculation method.
        lookback_days: Historical lookback period.
        **kwargs: Additional config parameters.

    Returns:
        Configured IntradayVaRCalculator instance.
    """
    config = VaRConfig(
        confidence_levels=confidence_levels or [0.95, 0.99],
        default_method=method,
        lookback_days=lookback_days,
        **kwargs,
    )
    return IntradayVaRCalculator(config)


def create_stress_test_framework(
    custom_scenarios: dict[str, dict[str, float]] | None = None,
    **kwargs: Any,
) -> StressTestFramework:
    """Factory function to create stress test framework.

    Args:
        custom_scenarios: Additional custom scenarios.
        **kwargs: Additional config parameters.

    Returns:
        Configured StressTestFramework instance.
    """
    config = StressTestConfig(**kwargs)

    if custom_scenarios:
        for name, shocks in custom_scenarios.items():
            config.scenarios[StressScenario.CUSTOM] = shocks

    return StressTestFramework(config)
