"""
JPMORGAN FIX: Performance Attribution Module.

Provides comprehensive performance attribution analysis including:
- Brinson-Fachler attribution (allocation, selection, interaction)
- Factor-based attribution (Fama-French, Carhart)
- Risk attribution (VaR contribution, marginal risk)
- Trade-level attribution
- Time-period attribution

Essential for understanding sources of alpha and risk.

Author: AlphaTrade System
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class AttributionMethod(str, Enum):
    """Performance attribution methods."""

    BRINSON_FACHLER = "brinson_fachler"
    BRINSON_HOOD_BEEBOWER = "brinson_hood_beebower"
    FACTOR_BASED = "factor_based"
    RISK_BASED = "risk_based"
    TRADE_LEVEL = "trade_level"


class AttributionPeriod(str, Enum):
    """Attribution analysis periods."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    FULL_PERIOD = "full_period"


@dataclass
class AllocationEffect:
    """Allocation effect in Brinson attribution."""

    sector: str
    portfolio_weight: float
    benchmark_weight: float
    benchmark_return: float
    allocation_effect: float
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sector": self.sector,
            "portfolio_weight": self.portfolio_weight,
            "benchmark_weight": self.benchmark_weight,
            "benchmark_return": self.benchmark_return,
            "allocation_effect": self.allocation_effect,
            "description": self.description,
        }


@dataclass
class SelectionEffect:
    """Selection effect in Brinson attribution."""

    sector: str
    portfolio_return: float
    benchmark_return: float
    benchmark_weight: float
    selection_effect: float
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sector": self.sector,
            "portfolio_return": self.portfolio_return,
            "benchmark_return": self.benchmark_return,
            "benchmark_weight": self.benchmark_weight,
            "selection_effect": self.selection_effect,
            "description": self.description,
        }


@dataclass
class InteractionEffect:
    """Interaction effect in Brinson attribution."""

    sector: str
    weight_diff: float
    return_diff: float
    interaction_effect: float
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sector": self.sector,
            "weight_diff": self.weight_diff,
            "return_diff": self.return_diff,
            "interaction_effect": self.interaction_effect,
            "description": self.description,
        }


@dataclass
class BrinsonAttribution:
    """Complete Brinson-Fachler attribution results."""

    portfolio_return: float
    benchmark_return: float
    active_return: float
    allocation_effects: list[AllocationEffect]
    selection_effects: list[SelectionEffect]
    interaction_effects: list[InteractionEffect]
    total_allocation: float
    total_selection: float
    total_interaction: float
    period_start: datetime
    period_end: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def attribution_error(self) -> float:
        """Calculate attribution error (should be close to zero)."""
        explained = self.total_allocation + self.total_selection + self.total_interaction
        return self.active_return - explained

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "portfolio_return": self.portfolio_return,
            "benchmark_return": self.benchmark_return,
            "active_return": self.active_return,
            "total_allocation": self.total_allocation,
            "total_selection": self.total_selection,
            "total_interaction": self.total_interaction,
            "attribution_error": self.attribution_error,
            "allocation_effects": [e.to_dict() for e in self.allocation_effects],
            "selection_effects": [e.to_dict() for e in self.selection_effects],
            "interaction_effects": [e.to_dict() for e in self.interaction_effects],
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class FactorExposure:
    """Factor exposure and contribution."""

    factor_name: str
    exposure: float  # Beta to factor
    factor_return: float
    contribution: float  # exposure * factor_return
    t_statistic: float | None = None
    p_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "factor_name": self.factor_name,
            "exposure": self.exposure,
            "factor_return": self.factor_return,
            "contribution": self.contribution,
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
        }


@dataclass
class FactorAttribution:
    """Factor-based attribution results."""

    total_return: float
    factor_exposures: list[FactorExposure]
    alpha: float  # Unexplained return (idiosyncratic)
    r_squared: float
    adjusted_r_squared: float
    total_factor_contribution: float
    period_start: datetime
    period_end: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "alpha": self.alpha,
            "r_squared": self.r_squared,
            "adjusted_r_squared": self.adjusted_r_squared,
            "total_factor_contribution": self.total_factor_contribution,
            "factor_exposures": [f.to_dict() for f in self.factor_exposures],
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class RiskContribution:
    """Risk contribution by position or factor."""

    name: str
    weight: float
    volatility: float
    correlation_with_portfolio: float
    marginal_contribution: float
    contribution_pct: float  # % of total portfolio risk

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "weight": self.weight,
            "volatility": self.volatility,
            "correlation_with_portfolio": self.correlation_with_portfolio,
            "marginal_contribution": self.marginal_contribution,
            "contribution_pct": self.contribution_pct,
        }


@dataclass
class RiskAttribution:
    """Risk-based attribution results."""

    portfolio_volatility: float
    portfolio_var_95: float
    portfolio_cvar_95: float
    position_contributions: list[RiskContribution]
    diversification_ratio: float
    concentration_ratio: float
    period_start: datetime
    period_end: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "portfolio_volatility": self.portfolio_volatility,
            "portfolio_var_95": self.portfolio_var_95,
            "portfolio_cvar_95": self.portfolio_cvar_95,
            "diversification_ratio": self.diversification_ratio,
            "concentration_ratio": self.concentration_ratio,
            "position_contributions": [p.to_dict() for p in self.position_contributions],
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class TradeAttribution:
    """Attribution for individual trades."""

    symbol: str
    entry_time: datetime
    exit_time: datetime
    side: str
    quantity: float
    gross_pnl: float
    commission: float
    slippage: float
    net_pnl: float
    return_contribution: float  # Contribution to portfolio return
    timing_effect: float  # Return from entry/exit timing
    selection_effect: float  # Return from security selection
    execution_effect: float  # Impact of execution quality

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "side": self.side,
            "quantity": self.quantity,
            "gross_pnl": self.gross_pnl,
            "commission": self.commission,
            "slippage": self.slippage,
            "net_pnl": self.net_pnl,
            "return_contribution": self.return_contribution,
            "timing_effect": self.timing_effect,
            "selection_effect": self.selection_effect,
            "execution_effect": self.execution_effect,
        }


class BrinsonAttributor:
    """
    Brinson-Fachler performance attribution.

    Decomposes active return into:
    - Allocation effect: Return from over/under-weighting sectors
    - Selection effect: Return from security selection within sectors
    - Interaction effect: Combined effect of allocation and selection
    """

    def __init__(
        self,
        sector_mapping: dict[str, str] | None = None,
    ):
        """Initialize Brinson attributor.

        Args:
            sector_mapping: Mapping of symbols to sectors.
        """
        self.sector_mapping = sector_mapping or {}

    def _get_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        return self.sector_mapping.get(symbol, "Other")

    def attribute(
        self,
        portfolio_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_weights: pd.Series,
        benchmark_returns: pd.Series,
        period_start: datetime,
        period_end: datetime,
    ) -> BrinsonAttribution:
        """Perform Brinson-Fachler attribution.

        Args:
            portfolio_weights: Portfolio weights by symbol.
            portfolio_returns: Portfolio returns by symbol.
            benchmark_weights: Benchmark weights by symbol.
            benchmark_returns: Benchmark returns by symbol.
            period_start: Start of attribution period.
            period_end: End of attribution period.

        Returns:
            BrinsonAttribution results.
        """
        # Group by sector
        portfolio_sectors = portfolio_weights.groupby(
            portfolio_weights.index.map(self._get_sector)
        ).sum()
        benchmark_sectors = benchmark_weights.groupby(
            benchmark_weights.index.map(self._get_sector)
        ).sum()

        # Calculate sector returns (weighted average)
        portfolio_sector_returns = {}
        benchmark_sector_returns = {}

        for symbol in portfolio_returns.index:
            sector = self._get_sector(symbol)
            if sector not in portfolio_sector_returns:
                portfolio_sector_returns[sector] = 0.0
            weight = portfolio_weights.get(symbol, 0)
            if portfolio_sectors.get(sector, 0) > 0:
                portfolio_sector_returns[sector] += (
                    weight / portfolio_sectors[sector] * portfolio_returns[symbol]
                )

        for symbol in benchmark_returns.index:
            sector = self._get_sector(symbol)
            if sector not in benchmark_sector_returns:
                benchmark_sector_returns[sector] = 0.0
            weight = benchmark_weights.get(symbol, 0)
            if benchmark_sectors.get(sector, 0) > 0:
                benchmark_sector_returns[sector] += (
                    weight / benchmark_sectors[sector] * benchmark_returns[symbol]
                )

        # Calculate total returns
        portfolio_return = sum(
            portfolio_weights[s] * portfolio_returns[s]
            for s in portfolio_returns.index
            if s in portfolio_weights
        )
        benchmark_return = sum(
            benchmark_weights[s] * benchmark_returns[s]
            for s in benchmark_returns.index
            if s in benchmark_weights
        )
        active_return = portfolio_return - benchmark_return

        # Calculate attribution effects
        allocation_effects = []
        selection_effects = []
        interaction_effects = []

        all_sectors = set(portfolio_sectors.index) | set(benchmark_sectors.index)

        for sector in all_sectors:
            p_weight = portfolio_sectors.get(sector, 0.0)
            b_weight = benchmark_sectors.get(sector, 0.0)
            p_return = portfolio_sector_returns.get(sector, 0.0)
            b_return = benchmark_sector_returns.get(sector, 0.0)

            # Allocation effect: (wp - wb) * (Rb - RB)
            alloc = (p_weight - b_weight) * (b_return - benchmark_return)
            allocation_effects.append(
                AllocationEffect(
                    sector=sector,
                    portfolio_weight=p_weight,
                    benchmark_weight=b_weight,
                    benchmark_return=b_return,
                    allocation_effect=alloc,
                    description=f"Effect from {'over' if p_weight > b_weight else 'under'}-weighting {sector}",
                )
            )

            # Selection effect: wb * (Rp - Rb)
            selec = b_weight * (p_return - b_return)
            selection_effects.append(
                SelectionEffect(
                    sector=sector,
                    portfolio_return=p_return,
                    benchmark_return=b_return,
                    benchmark_weight=b_weight,
                    selection_effect=selec,
                    description=f"Effect from security selection in {sector}",
                )
            )

            # Interaction effect: (wp - wb) * (Rp - Rb)
            inter = (p_weight - b_weight) * (p_return - b_return)
            interaction_effects.append(
                InteractionEffect(
                    sector=sector,
                    weight_diff=p_weight - b_weight,
                    return_diff=p_return - b_return,
                    interaction_effect=inter,
                    description=f"Interaction effect for {sector}",
                )
            )

        total_allocation = sum(e.allocation_effect for e in allocation_effects)
        total_selection = sum(e.selection_effect for e in selection_effects)
        total_interaction = sum(e.interaction_effect for e in interaction_effects)

        return BrinsonAttribution(
            portfolio_return=portfolio_return,
            benchmark_return=benchmark_return,
            active_return=active_return,
            allocation_effects=allocation_effects,
            selection_effects=selection_effects,
            interaction_effects=interaction_effects,
            total_allocation=total_allocation,
            total_selection=total_selection,
            total_interaction=total_interaction,
            period_start=period_start,
            period_end=period_end,
        )


class FactorAttributor:
    """
    Factor-based performance attribution.

    Decomposes returns into factor exposures using regression analysis.
    Supports standard factor models (CAPM, Fama-French, Carhart).
    """

    def __init__(
        self,
        factor_names: list[str] | None = None,
        annualization_factor: float = 252.0,
    ):
        """Initialize factor attributor.

        Args:
            factor_names: Names of factors to use.
            annualization_factor: Factor for annualizing returns.
        """
        self.factor_names = factor_names or ["market", "size", "value", "momentum"]
        self.annualization_factor = annualization_factor

    def attribute(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        period_start: datetime,
        period_end: datetime,
    ) -> FactorAttribution:
        """Perform factor-based attribution.

        Args:
            portfolio_returns: Time series of portfolio returns.
            factor_returns: DataFrame of factor returns (columns = factors).
            period_start: Start of attribution period.
            period_end: End of attribution period.

        Returns:
            FactorAttribution results.
        """
        # Align data
        common_index = portfolio_returns.index.intersection(factor_returns.index)
        y = portfolio_returns.loc[common_index].values
        X = factor_returns.loc[common_index].values

        if len(y) < 10:
            logger.warning("Insufficient data for factor attribution")
            return self._empty_attribution(period_start, period_end)

        # Add constant for intercept
        X_with_const = np.column_stack([np.ones(len(X)), X])

        # OLS regression
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)
            alpha = coeffs[0]
            betas = coeffs[1:]

            # Predictions and R-squared
            y_pred = X_with_const @ coeffs
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Adjusted R-squared
            n = len(y)
            p = len(betas)
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else r_squared

            # Standard errors and t-statistics
            mse = ss_res / (n - p - 1) if n > p + 1 else 0
            var_covar = mse * np.linalg.inv(X_with_const.T @ X_with_const)
            se = np.sqrt(np.diag(var_covar))

            t_stats = coeffs / se if np.all(se > 0) else np.zeros_like(coeffs)
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - p - 1))

        except Exception as e:
            logger.warning(f"Regression failed: {e}")
            return self._empty_attribution(period_start, period_end)

        # Calculate factor contributions
        factor_exposures = []
        factor_columns = factor_returns.columns.tolist()

        for i, factor_name in enumerate(factor_columns):
            factor_ret = factor_returns[factor_name].loc[common_index].mean()
            contribution = betas[i] * factor_ret

            factor_exposures.append(
                FactorExposure(
                    factor_name=factor_name,
                    exposure=betas[i],
                    factor_return=factor_ret,
                    contribution=contribution,
                    t_statistic=t_stats[i + 1] if i + 1 < len(t_stats) else None,
                    p_value=p_values[i + 1] if i + 1 < len(p_values) else None,
                )
            )

        total_factor_contribution = sum(f.contribution for f in factor_exposures)
        total_return = portfolio_returns.loc[common_index].mean()

        return FactorAttribution(
            total_return=total_return,
            factor_exposures=factor_exposures,
            alpha=alpha,
            r_squared=r_squared,
            adjusted_r_squared=adj_r_squared,
            total_factor_contribution=total_factor_contribution,
            period_start=period_start,
            period_end=period_end,
            metadata={
                "n_observations": len(y),
                "alpha_annualized": alpha * self.annualization_factor,
            },
        )

    def _empty_attribution(
        self,
        period_start: datetime,
        period_end: datetime,
    ) -> FactorAttribution:
        """Return empty attribution when calculation fails."""
        return FactorAttribution(
            total_return=0.0,
            factor_exposures=[],
            alpha=0.0,
            r_squared=0.0,
            adjusted_r_squared=0.0,
            total_factor_contribution=0.0,
            period_start=period_start,
            period_end=period_end,
            metadata={"error": "insufficient_data"},
        )


class RiskAttributor:
    """
    Risk-based performance attribution.

    Analyzes contribution of each position to portfolio risk.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        lookback_period: int = 252,
    ):
        """Initialize risk attributor.

        Args:
            confidence_level: Confidence level for VaR/CVaR.
            lookback_period: Historical lookback for risk calculations.
        """
        self.confidence_level = confidence_level
        self.lookback_period = lookback_period

    def attribute(
        self,
        position_returns: pd.DataFrame,
        position_weights: pd.Series,
        period_start: datetime,
        period_end: datetime,
    ) -> RiskAttribution:
        """Perform risk attribution.

        Args:
            position_returns: DataFrame of position returns.
            position_weights: Current position weights.
            period_start: Start of attribution period.
            period_end: End of attribution period.

        Returns:
            RiskAttribution results.
        """
        # Calculate portfolio returns
        aligned_weights = position_weights.reindex(position_returns.columns).fillna(0)
        portfolio_returns = (position_returns * aligned_weights).sum(axis=1)

        # Portfolio risk metrics
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)
        portfolio_var = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
        portfolio_cvar = portfolio_returns[portfolio_returns <= portfolio_var].mean()

        # Covariance matrix
        cov_matrix = position_returns.cov() * 252

        # Calculate risk contributions
        contributions = []
        total_variance = 0.0

        for symbol in position_returns.columns:
            weight = aligned_weights.get(symbol, 0)
            if weight == 0:
                continue

            # Position volatility
            pos_vol = position_returns[symbol].std() * np.sqrt(252)

            # Correlation with portfolio
            if portfolio_returns.std() > 0 and pos_vol > 0:
                corr = np.corrcoef(
                    position_returns[symbol].fillna(0),
                    portfolio_returns.fillna(0),
                )[0, 1]
            else:
                corr = 0.0

            # Marginal contribution to risk
            # MCR = d(sigma_p) / d(w_i) = cov(r_i, r_p) / sigma_p
            if portfolio_vol > 0:
                cov_with_port = np.cov(
                    position_returns[symbol].fillna(0),
                    portfolio_returns.fillna(0),
                )[0, 1] * 252
                marginal_contribution = cov_with_port / portfolio_vol
            else:
                marginal_contribution = 0.0

            # Total contribution = weight * marginal contribution
            total_contribution = weight * marginal_contribution
            total_variance += weight * cov_with_port if portfolio_vol > 0 else 0

            contributions.append(
                RiskContribution(
                    name=symbol,
                    weight=weight,
                    volatility=pos_vol,
                    correlation_with_portfolio=corr if not np.isnan(corr) else 0.0,
                    marginal_contribution=marginal_contribution,
                    contribution_pct=(
                        total_contribution / portfolio_vol * 100
                        if portfolio_vol > 0
                        else 0.0
                    ),
                )
            )

        # Diversification ratio
        weighted_avg_vol = sum(
            abs(aligned_weights[s]) * position_returns[s].std() * np.sqrt(252)
            for s in position_returns.columns
            if s in aligned_weights
        )
        diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0

        # Concentration ratio (Herfindahl index on risk contributions)
        if contributions:
            risk_weights = [abs(c.contribution_pct) / 100 for c in contributions]
            concentration_ratio = sum(w ** 2 for w in risk_weights)
        else:
            concentration_ratio = 0.0

        return RiskAttribution(
            portfolio_volatility=portfolio_vol,
            portfolio_var_95=portfolio_var,
            portfolio_cvar_95=portfolio_cvar,
            position_contributions=contributions,
            diversification_ratio=diversification_ratio,
            concentration_ratio=concentration_ratio,
            period_start=period_start,
            period_end=period_end,
        )


class TradeAttributor:
    """
    Trade-level performance attribution.

    Analyzes the contribution of individual trades to overall performance.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
    ):
        """Initialize trade attributor.

        Args:
            initial_capital: Starting capital for return calculation.
        """
        self.initial_capital = initial_capital

    def attribute(
        self,
        trades: list[dict[str, Any]],
        price_data: pd.DataFrame | None = None,
    ) -> list[TradeAttribution]:
        """Perform trade-level attribution.

        Args:
            trades: List of trade dictionaries with fields:
                - symbol, entry_time, exit_time, side, quantity
                - entry_price, exit_price, pnl, commission, slippage
            price_data: Optional price data for timing analysis.

        Returns:
            List of TradeAttribution results.
        """
        attributions = []
        cumulative_capital = self.initial_capital

        for trade in trades:
            symbol = trade.get("symbol", "UNKNOWN")
            entry_time = trade.get("entry_time")
            exit_time = trade.get("exit_time")
            side = trade.get("side", "buy")
            quantity = float(trade.get("quantity", 0))
            entry_price = float(trade.get("entry_price", 0))
            exit_price = float(trade.get("exit_price", 0))
            pnl = float(trade.get("pnl", 0))
            commission = float(trade.get("commission", 0))
            slippage = float(trade.get("slippage", 0))

            # Parse times if needed
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
            if isinstance(exit_time, str):
                exit_time = datetime.fromisoformat(exit_time.replace("Z", "+00:00"))

            # Gross P&L (before costs)
            gross_pnl = pnl + commission + slippage

            # Return contribution
            return_contribution = pnl / cumulative_capital if cumulative_capital > 0 else 0

            # Timing effect: Compare actual return to VWAP or period average
            # Simplified: use midpoint of period
            timing_effect = 0.0
            selection_effect = 0.0
            execution_effect = 0.0

            if price_data is not None and symbol in price_data.columns:
                try:
                    # Get prices during holding period
                    mask = (price_data.index >= entry_time) & (price_data.index <= exit_time)
                    period_prices = price_data.loc[mask, symbol]

                    if len(period_prices) > 0:
                        period_avg = period_prices.mean()
                        period_start_price = period_prices.iloc[0]
                        period_end_price = period_prices.iloc[-1]

                        # Selection effect: return if held at average prices
                        if side == "buy":
                            avg_return = (period_end_price - period_start_price) / period_start_price
                        else:
                            avg_return = (period_start_price - period_end_price) / period_start_price

                        # Timing effect: difference from average
                        actual_return = gross_pnl / (entry_price * quantity) if entry_price * quantity > 0 else 0
                        timing_effect = (actual_return - avg_return) * entry_price * quantity

                        # Selection effect: would we have made money with average entry/exit?
                        selection_effect = avg_return * entry_price * quantity

                        # Execution effect: slippage and commission impact
                        execution_effect = -(commission + slippage)
                except Exception as e:
                    logger.debug(f"Timing analysis failed for {symbol}: {e}")

            attributions.append(
                TradeAttribution(
                    symbol=symbol,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    side=side,
                    quantity=quantity,
                    gross_pnl=gross_pnl,
                    commission=commission,
                    slippage=slippage,
                    net_pnl=pnl,
                    return_contribution=return_contribution,
                    timing_effect=timing_effect,
                    selection_effect=selection_effect,
                    execution_effect=execution_effect,
                )
            )

            # Update cumulative capital
            cumulative_capital += pnl

        return attributions


class PerformanceAttributionService:
    """
    Unified service for performance attribution.

    Provides comprehensive attribution analysis using multiple methods.
    """

    def __init__(
        self,
        brinson_attributor: BrinsonAttributor | None = None,
        factor_attributor: FactorAttributor | None = None,
        risk_attributor: RiskAttributor | None = None,
        trade_attributor: TradeAttributor | None = None,
    ):
        """Initialize attribution service.

        Args:
            brinson_attributor: Brinson attribution instance.
            factor_attributor: Factor attribution instance.
            risk_attributor: Risk attribution instance.
            trade_attributor: Trade attribution instance.
        """
        self.brinson = brinson_attributor or BrinsonAttributor()
        self.factor = factor_attributor or FactorAttributor()
        self.risk = risk_attributor or RiskAttributor()
        self.trade = trade_attributor or TradeAttributor()

    def generate_full_report(
        self,
        portfolio_returns: pd.Series,
        portfolio_weights: pd.Series,
        benchmark_returns: pd.Series | None = None,
        benchmark_weights: pd.Series | None = None,
        factor_returns: pd.DataFrame | None = None,
        position_returns: pd.DataFrame | None = None,
        trades: list[dict] | None = None,
        period_start: datetime | None = None,
        period_end: datetime | None = None,
    ) -> dict[str, Any]:
        """Generate comprehensive attribution report.

        Args:
            portfolio_returns: Portfolio return series.
            portfolio_weights: Portfolio weights.
            benchmark_returns: Optional benchmark returns.
            benchmark_weights: Optional benchmark weights.
            factor_returns: Optional factor return data.
            position_returns: Optional position-level returns.
            trades: Optional trade list.
            period_start: Attribution period start.
            period_end: Attribution period end.

        Returns:
            Complete attribution report dictionary.
        """
        period_start = period_start or portfolio_returns.index[0]
        period_end = period_end or portfolio_returns.index[-1]

        if isinstance(period_start, pd.Timestamp):
            period_start = period_start.to_pydatetime()
        if isinstance(period_end, pd.Timestamp):
            period_end = period_end.to_pydatetime()

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "summary": {
                "total_return": float(portfolio_returns.sum()),
                "annualized_return": float(
                    portfolio_returns.mean() * 252
                ),
                "volatility": float(portfolio_returns.std() * np.sqrt(252)),
            },
        }

        # Brinson attribution
        if benchmark_returns is not None and benchmark_weights is not None:
            try:
                # Get single-period returns
                p_returns = portfolio_returns.groupby(level=0).last()
                b_returns = benchmark_returns.groupby(level=0).last() if hasattr(benchmark_returns, 'groupby') else benchmark_returns

                brinson_result = self.brinson.attribute(
                    portfolio_weights=portfolio_weights,
                    portfolio_returns=p_returns if isinstance(p_returns, pd.Series) else portfolio_returns,
                    benchmark_weights=benchmark_weights,
                    benchmark_returns=b_returns if isinstance(b_returns, pd.Series) else benchmark_returns,
                    period_start=period_start,
                    period_end=period_end,
                )
                report["brinson_attribution"] = brinson_result.to_dict()
            except Exception as e:
                report["brinson_attribution"] = {"error": str(e)}

        # Factor attribution
        if factor_returns is not None:
            try:
                factor_result = self.factor.attribute(
                    portfolio_returns=portfolio_returns,
                    factor_returns=factor_returns,
                    period_start=period_start,
                    period_end=period_end,
                )
                report["factor_attribution"] = factor_result.to_dict()
            except Exception as e:
                report["factor_attribution"] = {"error": str(e)}

        # Risk attribution
        if position_returns is not None:
            try:
                risk_result = self.risk.attribute(
                    position_returns=position_returns,
                    position_weights=portfolio_weights,
                    period_start=period_start,
                    period_end=period_end,
                )
                report["risk_attribution"] = risk_result.to_dict()
            except Exception as e:
                report["risk_attribution"] = {"error": str(e)}

        # Trade attribution
        if trades:
            try:
                trade_results = self.trade.attribute(trades=trades)
                report["trade_attribution"] = {
                    "total_trades": len(trade_results),
                    "trades": [t.to_dict() for t in trade_results],
                    "summary": {
                        "total_gross_pnl": sum(t.gross_pnl for t in trade_results),
                        "total_net_pnl": sum(t.net_pnl for t in trade_results),
                        "total_commission": sum(t.commission for t in trade_results),
                        "total_slippage": sum(t.slippage for t in trade_results),
                    },
                }
            except Exception as e:
                report["trade_attribution"] = {"error": str(e)}

        return report


def create_attribution_service(
    sector_mapping: dict[str, str] | None = None,
    factor_names: list[str] | None = None,
    initial_capital: float = 100000.0,
) -> PerformanceAttributionService:
    """Factory function to create attribution service.

    Args:
        sector_mapping: Symbol to sector mapping.
        factor_names: Factor names for factor attribution.
        initial_capital: Starting capital for trade attribution.

    Returns:
        Configured PerformanceAttributionService.
    """
    return PerformanceAttributionService(
        brinson_attributor=BrinsonAttributor(sector_mapping=sector_mapping),
        factor_attributor=FactorAttributor(factor_names=factor_names),
        risk_attributor=RiskAttributor(),
        trade_attributor=TradeAttributor(initial_capital=initial_capital),
    )
