"""
Performance analytics module for backtesting.

Provides comprehensive performance analysis including:
- Return metrics (total, annualized, risk-adjusted)
- Drawdown analysis (max, average, duration)
- Trade analysis (win rate, profit factor, expectancy)
- Benchmark comparison (alpha, beta, information ratio)
- Statistical tests and robustness analysis
- Visualization support
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from quant_trading_system.backtest.engine import BacktestState, Trade

logger = logging.getLogger(__name__)


@dataclass
class ReturnMetrics:
    """Return-based performance metrics."""

    total_return: float
    annualized_return: float
    monthly_returns: list[float]
    daily_returns: list[float]
    best_day: float
    worst_day: float
    best_month: float
    worst_month: float
    positive_days_pct: float
    positive_months_pct: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "best_day": self.best_day,
            "worst_day": self.worst_day,
            "best_month": self.best_month,
            "worst_month": self.worst_month,
            "positive_days_pct": self.positive_days_pct,
            "positive_months_pct": self.positive_months_pct,
        }


@dataclass
class RiskAdjustedMetrics:
    """Risk-adjusted performance metrics."""

    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    treynor_ratio: float
    omega_ratio: float
    volatility_annual: float
    downside_volatility: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "information_ratio": self.information_ratio,
            "treynor_ratio": self.treynor_ratio,
            "omega_ratio": self.omega_ratio,
            "volatility_annual": self.volatility_annual,
            "downside_volatility": self.downside_volatility,
        }


@dataclass
class DrawdownMetrics:
    """Drawdown-based metrics."""

    max_drawdown: float
    max_drawdown_duration_days: int
    avg_drawdown: float
    avg_drawdown_duration_days: float
    recovery_time_days: int | None
    current_drawdown: float
    drawdown_periods: list[dict[str, Any]]
    ulcer_index: float
    pain_index: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "avg_drawdown": self.avg_drawdown,
            "avg_drawdown_duration_days": self.avg_drawdown_duration_days,
            "recovery_time_days": self.recovery_time_days,
            "current_drawdown": self.current_drawdown,
            "ulcer_index": self.ulcer_index,
            "pain_index": self.pain_index,
        }


@dataclass
class TradeMetrics:
    """Trade-based performance metrics."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    profit_factor: float
    expectancy: float
    payoff_ratio: float
    avg_trade_duration_bars: float
    avg_winning_duration_bars: float
    avg_losing_duration_bars: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    gross_profit: float
    gross_loss: float
    net_profit: float
    total_commission: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "win_loss_ratio": self.win_loss_ratio,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "payoff_ratio": self.payoff_ratio,
            "avg_trade_duration_bars": self.avg_trade_duration_bars,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "net_profit": self.net_profit,
            "total_commission": self.total_commission,
        }


@dataclass
class BenchmarkMetrics:
    """Benchmark comparison metrics."""

    alpha: float
    beta: float
    r_squared: float
    tracking_error: float
    information_ratio: float
    active_return: float
    correlation: float
    up_capture: float
    down_capture: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "r_squared": self.r_squared,
            "tracking_error": self.tracking_error,
            "information_ratio": self.information_ratio,
            "active_return": self.active_return,
            "correlation": self.correlation,
            "up_capture": self.up_capture,
            "down_capture": self.down_capture,
        }


@dataclass
class StatisticalTests:
    """Statistical significance tests.

    P0 Enhancement: Added Deflated Sharpe Ratio (DSR) and Probability of
    Backtest Overfitting (PBO) for institutional-grade validation.
    """

    returns_tstat: float
    returns_pvalue: float
    sharpe_confidence_95: tuple[float, float]
    bootstrap_mean_return: float
    bootstrap_std_return: float
    monte_carlo_prob_positive: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    is_normal: bool

    # P0: Deflated Sharpe Ratio - accounts for multiple testing bias
    deflated_sharpe_ratio: float = 0.0
    dsr_pvalue: float = 1.0
    n_trials_tested: int = 1

    # P0: Probability of Backtest Overfitting
    pbo: float = 0.5
    pbo_interpretation: str = "Not calculated"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "returns_tstat": self.returns_tstat,
            "returns_pvalue": self.returns_pvalue,
            "sharpe_confidence_95": list(self.sharpe_confidence_95),
            "bootstrap_mean_return": self.bootstrap_mean_return,
            "monte_carlo_prob_positive": self.monte_carlo_prob_positive,
            "is_normal": self.is_normal,
            # P0: DSR metrics
            "deflated_sharpe_ratio": self.deflated_sharpe_ratio,
            "dsr_pvalue": self.dsr_pvalue,
            "n_trials_tested": self.n_trials_tested,
            # P0: PBO metrics
            "pbo": self.pbo,
            "pbo_interpretation": self.pbo_interpretation,
        }


@dataclass
class RegimeMetrics:
    """P1 Enhancement: Per-regime performance metrics.

    Breaks down performance by market regime for institutional-grade
    all-weather strategy validation.
    """

    regime: str
    n_bars: int
    total_return: float
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "regime": self.regime,
            "n_bars": self.n_bars,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
        }


@dataclass
class CostSensitivity:
    """P1 Enhancement: Transaction cost sensitivity analysis.

    Determines the break-even cost level and how strategy performance
    degrades with increasing transaction costs.
    """

    break_even_cost_bps: float
    sharpe_at_costs: dict[float, float]  # cost_bps -> sharpe
    return_at_costs: dict[float, float]  # cost_bps -> return

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "break_even_cost_bps": self.break_even_cost_bps,
            "sharpe_at_costs": self.sharpe_at_costs,
            "return_at_costs": self.return_at_costs,
        }


@dataclass
class PerformanceReport:
    """Complete performance report.

    P0/P1 Enhancement: Added regime_metrics and cost_sensitivity
    for institutional-grade analysis.
    """

    return_metrics: ReturnMetrics
    risk_adjusted_metrics: RiskAdjustedMetrics
    drawdown_metrics: DrawdownMetrics
    trade_metrics: TradeMetrics
    benchmark_metrics: BenchmarkMetrics | None
    statistical_tests: StatisticalTests
    start_date: datetime
    end_date: datetime
    trading_days: int
    # P1: Optional advanced metrics
    regime_metrics: list[RegimeMetrics] | None = None
    cost_sensitivity: CostSensitivity | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "return_metrics": self.return_metrics.to_dict(),
            "risk_adjusted_metrics": self.risk_adjusted_metrics.to_dict(),
            "drawdown_metrics": self.drawdown_metrics.to_dict(),
            "trade_metrics": self.trade_metrics.to_dict(),
            "benchmark_metrics": self.benchmark_metrics.to_dict() if self.benchmark_metrics else None,
            "statistical_tests": self.statistical_tests.to_dict(),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "trading_days": self.trading_days,
        }
        # P1: Add optional advanced metrics
        if self.regime_metrics:
            result["regime_metrics"] = [rm.to_dict() for rm in self.regime_metrics]
        if self.cost_sensitivity:
            result["cost_sensitivity"] = self.cost_sensitivity.to_dict()
        return result

    def summary(self) -> str:
        """Get a summary string of key metrics."""
        summary = f"""
Performance Summary ({self.start_date.date()} to {self.end_date.date()})
{'=' * 60}
Total Return:      {self.return_metrics.total_return:>10.2%}
Annual Return:     {self.return_metrics.annualized_return:>10.2%}
Sharpe Ratio:      {self.risk_adjusted_metrics.sharpe_ratio:>10.2f}
Sortino Ratio:     {self.risk_adjusted_metrics.sortino_ratio:>10.2f}
Max Drawdown:      {self.drawdown_metrics.max_drawdown:>10.2%}
Volatility:        {self.risk_adjusted_metrics.volatility_annual:>10.2%}
Win Rate:          {self.trade_metrics.win_rate:>10.2%}
Profit Factor:     {self.trade_metrics.profit_factor:>10.2f}
Total Trades:      {self.trade_metrics.total_trades:>10d}
{'=' * 60}
Statistical Validation (P0):
  Deflated Sharpe: {self.statistical_tests.deflated_sharpe_ratio:>10.2f}
  DSR p-value:     {self.statistical_tests.dsr_pvalue:>10.4f}
  PBO:             {self.statistical_tests.pbo:>10.2%}
  PBO Status:      {self.statistical_tests.pbo_interpretation}
{'=' * 60}
"""
        return summary


class PerformanceAnalyzer:
    """Analyzes backtest performance and generates metrics."""

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ) -> None:
        """Initialize performance analyzer.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation.
            periods_per_year: Trading periods per year (252 for daily).
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def analyze(
        self,
        backtest_state: BacktestState,
        benchmark_returns: np.ndarray | None = None,
        n_trials: int = 1,
    ) -> PerformanceReport:
        """Generate complete performance report.

        P0/P1 Enhancement: Added DSR, PBO, and optional regime/cost analysis.

        Args:
            backtest_state: Completed backtest state.
            benchmark_returns: Optional benchmark returns for comparison.
            n_trials: Number of strategies/parameters tested (for multiple testing).

        Returns:
            Complete performance report.
        """
        # Extract equity curve
        equity_curve = np.array([e for _, e in backtest_state.equity_curve])
        timestamps = [t for t, _ in backtest_state.equity_curve]

        if len(equity_curve) < 2:
            raise ValueError("Insufficient data for analysis")

        # Calculate returns
        returns = np.diff(equity_curve) / equity_curve[:-1]

        # Generate all metrics
        return_metrics = self._calculate_return_metrics(returns, timestamps)
        risk_adjusted = self._calculate_risk_adjusted_metrics(returns)
        drawdown_metrics = self._calculate_drawdown_metrics(equity_curve, timestamps)
        trade_metrics = self._calculate_trade_metrics(backtest_state.trades)

        benchmark_metrics = None
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            benchmark_metrics = self._calculate_benchmark_metrics(returns, benchmark_returns)

        # P0: Pass sharpe_ratio and n_trials for DSR/PBO calculation
        statistical_tests = self._calculate_statistical_tests(
            returns,
            sharpe_ratio=risk_adjusted.sharpe_ratio,
            n_trials=n_trials,
        )

        return PerformanceReport(
            return_metrics=return_metrics,
            risk_adjusted_metrics=risk_adjusted,
            drawdown_metrics=drawdown_metrics,
            trade_metrics=trade_metrics,
            benchmark_metrics=benchmark_metrics,
            statistical_tests=statistical_tests,
            start_date=timestamps[0] if timestamps else datetime.now(timezone.utc),
            end_date=timestamps[-1] if timestamps else datetime.now(timezone.utc),
            trading_days=len(returns),
        )

    def _calculate_return_metrics(
        self,
        returns: np.ndarray,
        timestamps: list[datetime],
    ) -> ReturnMetrics:
        """Calculate return-based metrics."""
        # Total and annualized return
        total_return = np.prod(1 + returns) - 1
        n_periods = len(returns)
        annualized_return = (1 + total_return) ** (self.periods_per_year / n_periods) - 1

        # Monthly returns (approximate)
        monthly_returns = []
        if len(timestamps) > 20:
            monthly_period = max(1, n_periods // 12)
            for i in range(0, n_periods, monthly_period):
                chunk = returns[i:i + monthly_period]
                if len(chunk) > 0:
                    monthly_returns.append(np.prod(1 + chunk) - 1)

        # Daily stats
        best_day = float(np.max(returns))
        worst_day = float(np.min(returns))
        positive_days = np.sum(returns > 0)
        positive_days_pct = positive_days / len(returns) if len(returns) > 0 else 0

        # Monthly stats
        best_month = float(np.max(monthly_returns)) if monthly_returns else 0.0
        worst_month = float(np.min(monthly_returns)) if monthly_returns else 0.0
        positive_months = sum(1 for r in monthly_returns if r > 0)
        positive_months_pct = positive_months / len(monthly_returns) if monthly_returns else 0

        return ReturnMetrics(
            total_return=float(total_return),
            annualized_return=float(annualized_return),
            monthly_returns=monthly_returns,
            daily_returns=returns.tolist(),
            best_day=best_day,
            worst_day=worst_day,
            best_month=best_month,
            worst_month=worst_month,
            positive_days_pct=float(positive_days_pct),
            positive_months_pct=float(positive_months_pct),
        )

    def _calculate_risk_adjusted_metrics(
        self,
        returns: np.ndarray,
    ) -> RiskAdjustedMetrics:
        """Calculate risk-adjusted metrics."""
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Annualize
        annual_return = mean_return * self.periods_per_year
        annual_vol = std_return * np.sqrt(self.periods_per_year)

        # Sharpe ratio
        daily_rf = self.risk_free_rate / self.periods_per_year
        excess_returns = returns - daily_rf
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(self.periods_per_year) if std_return > 0 else 0

        # Sortino ratio (downside deviation)
        # BUG FIX: Correct downside deviation formula uses ALL samples
        # Formula: sqrt(mean(min(0, return - threshold)^2))
        # NOT the std of only negative returns
        threshold = daily_rf  # Use risk-free rate as threshold
        downside_returns = np.minimum(returns - threshold, 0)  # Negative deviations only
        downside_variance = np.mean(downside_returns ** 2)  # Mean of squared deviations
        downside_std = np.sqrt(downside_variance) if downside_variance > 0 else 0
        downside_vol = downside_std * np.sqrt(self.periods_per_year)
        sortino = (annual_return - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0

        # Calmar ratio
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_dd = abs(np.min(drawdown))
        calmar = annual_return / max_dd if max_dd > 0 else 0

        # Omega ratio
        threshold = daily_rf
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        omega = np.sum(gains) / np.sum(losses) if np.sum(losses) > 0 else float('inf')

        return RiskAdjustedMetrics(
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            calmar_ratio=float(calmar),
            information_ratio=0.0,  # Calculated with benchmark
            treynor_ratio=0.0,  # Calculated with benchmark
            omega_ratio=float(omega) if omega != float('inf') else 10.0,
            volatility_annual=float(annual_vol),
            downside_volatility=float(downside_vol),
        )

    def _calculate_drawdown_metrics(
        self,
        equity_curve: np.ndarray,
        timestamps: list[datetime],
    ) -> DrawdownMetrics:
        """Calculate drawdown metrics."""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak

        # Max drawdown
        max_dd = abs(np.min(drawdown))
        max_dd_idx = np.argmin(drawdown)

        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        dd_start_idx = 0

        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                dd_start_idx = i
            elif dd == 0 and in_drawdown:
                in_drawdown = False
                period = {
                    "start_idx": dd_start_idx,
                    "end_idx": i,
                    "trough_idx": dd_start_idx + np.argmin(drawdown[dd_start_idx:i]),
                    "depth": float(np.min(drawdown[dd_start_idx:i])),
                    "duration": i - dd_start_idx,
                }
                if len(timestamps) > i:
                    period["start_date"] = timestamps[dd_start_idx].isoformat()
                    period["end_date"] = timestamps[i].isoformat()
                drawdown_periods.append(period)

        # Calculate duration of max drawdown
        max_dd_period = None
        for period in drawdown_periods:
            if period["depth"] == -max_dd:
                max_dd_period = period
                break

        max_dd_duration = max_dd_period["duration"] if max_dd_period else 0

        # Average drawdown
        all_dds = [abs(p["depth"]) for p in drawdown_periods]
        avg_dd = np.mean(all_dds) if all_dds else 0

        all_durations = [p["duration"] for p in drawdown_periods]
        avg_duration = np.mean(all_durations) if all_durations else 0

        # Recovery time (time from max drawdown to new peak)
        recovery_time = None
        if max_dd_idx < len(drawdown) - 1:
            recovery_idx = np.where(drawdown[max_dd_idx:] == 0)[0]
            if len(recovery_idx) > 0:
                recovery_time = int(recovery_idx[0])

        # Ulcer Index (RMS of drawdowns)
        ulcer_index = np.sqrt(np.mean(drawdown ** 2))

        # Pain Index (mean of absolute drawdowns)
        pain_index = np.mean(np.abs(drawdown))

        return DrawdownMetrics(
            max_drawdown=float(max_dd),
            max_drawdown_duration_days=int(max_dd_duration),
            avg_drawdown=float(avg_dd),
            avg_drawdown_duration_days=float(avg_duration),
            recovery_time_days=recovery_time,
            current_drawdown=float(abs(drawdown[-1])),
            drawdown_periods=drawdown_periods[:10],  # Top 10 periods
            ulcer_index=float(ulcer_index),
            pain_index=float(pain_index),
        )

    def _calculate_trade_metrics(
        self,
        trades: list[Trade],
    ) -> TradeMetrics:
        """Calculate trade-based metrics."""
        if not trades:
            return TradeMetrics(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                win_loss_ratio=0.0,
                profit_factor=0.0,
                expectancy=0.0,
                payoff_ratio=0.0,
                avg_trade_duration_bars=0.0,
                avg_winning_duration_bars=0.0,
                avg_losing_duration_bars=0.0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                gross_profit=0.0,
                gross_loss=0.0,
                net_profit=0.0,
                total_commission=0.0,
            )

        # Separate winning and losing trades
        pnls = [float(t.pnl) for t in trades]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p < 0]

        total_trades = len(trades)
        winning_trades = len(winning)
        losing_trades = len(losing)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        avg_win = np.mean(winning) if winning else 0
        avg_loss = abs(np.mean(losing)) if losing else 0

        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

        gross_profit = sum(winning)
        gross_loss = abs(sum(losing))

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

        # Trade durations
        durations = [t.holding_period_bars for t in trades]
        avg_duration = np.mean(durations) if durations else 0

        winning_durations = [t.holding_period_bars for t in trades if float(t.pnl) > 0]
        avg_winning_duration = np.mean(winning_durations) if winning_durations else 0

        losing_durations = [t.holding_period_bars for t in trades if float(t.pnl) < 0]
        avg_losing_duration = np.mean(losing_durations) if losing_durations else 0

        # Consecutive wins/losses
        max_consecutive_wins = self._max_consecutive(pnls, lambda x: x > 0)
        max_consecutive_losses = self._max_consecutive(pnls, lambda x: x < 0)

        # Total commission
        total_commission = sum(float(t.commission) for t in trades)

        return TradeMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=float(win_rate),
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            win_loss_ratio=float(win_loss_ratio) if win_loss_ratio != float('inf') else 10.0,
            profit_factor=float(profit_factor) if profit_factor != float('inf') else 10.0,
            expectancy=float(expectancy),
            payoff_ratio=float(payoff_ratio) if payoff_ratio != float('inf') else 10.0,
            avg_trade_duration_bars=float(avg_duration),
            avg_winning_duration_bars=float(avg_winning_duration),
            avg_losing_duration_bars=float(avg_losing_duration),
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            gross_profit=float(gross_profit),
            gross_loss=float(gross_loss),
            net_profit=float(gross_profit - gross_loss),
            total_commission=float(total_commission),
        )

    def _max_consecutive(
        self,
        values: list[float],
        condition: callable,
    ) -> int:
        """Calculate maximum consecutive occurrences matching condition."""
        max_count = 0
        current_count = 0

        for v in values:
            if condition(v):
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def _calculate_benchmark_metrics(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> BenchmarkMetrics:
        """Calculate benchmark comparison metrics."""
        # Align lengths
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]

        # Beta and alpha (CAPM)
        covariance = np.cov(returns, benchmark_returns)
        beta = covariance[0, 1] / covariance[1, 1] if covariance[1, 1] > 0 else 1.0

        alpha = np.mean(returns) - beta * np.mean(benchmark_returns)
        alpha_annual = alpha * self.periods_per_year

        # R-squared
        correlation = np.corrcoef(returns, benchmark_returns)[0, 1]
        r_squared = correlation ** 2

        # Tracking error
        tracking_diff = returns - benchmark_returns
        tracking_error = np.std(tracking_diff) * np.sqrt(self.periods_per_year)

        # Information ratio
        active_return = np.mean(tracking_diff) * self.periods_per_year
        information_ratio = active_return / tracking_error if tracking_error > 0 else 0

        # Up/Down capture
        up_periods = benchmark_returns > 0
        down_periods = benchmark_returns < 0

        up_capture = (
            np.mean(returns[up_periods]) / np.mean(benchmark_returns[up_periods])
            if np.any(up_periods) and np.mean(benchmark_returns[up_periods]) != 0
            else 1.0
        )

        down_capture = (
            np.mean(returns[down_periods]) / np.mean(benchmark_returns[down_periods])
            if np.any(down_periods) and np.mean(benchmark_returns[down_periods]) != 0
            else 1.0
        )

        return BenchmarkMetrics(
            alpha=float(alpha_annual),
            beta=float(beta),
            r_squared=float(r_squared),
            tracking_error=float(tracking_error),
            information_ratio=float(information_ratio),
            active_return=float(active_return),
            correlation=float(correlation),
            up_capture=float(up_capture),
            down_capture=float(down_capture),
        )

    def _calculate_statistical_tests(
        self,
        returns: np.ndarray,
        sharpe_ratio: float = 0.0,
        n_trials: int = 1,
    ) -> StatisticalTests:
        """Calculate statistical significance tests.

        P0 Enhancement: Added Deflated Sharpe Ratio (DSR) and Probability
        of Backtest Overfitting (PBO) for institutional-grade validation.

        Args:
            returns: Array of period returns.
            sharpe_ratio: Observed Sharpe ratio for DSR calculation.
            n_trials: Number of trials/strategies tested (for multiple testing).

        Returns:
            StatisticalTests with all validation metrics.
        """
        # T-test for mean return
        t_stat, p_value = stats.ttest_1samp(returns, 0)

        # Sharpe ratio confidence interval (bootstrap)
        n_bootstrap = 1000
        bootstrap_sharpes = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(returns, size=len(returns), replace=True)
            sharpe = np.mean(sample) / np.std(sample) * np.sqrt(self.periods_per_year) if np.std(sample) > 0 else 0
            bootstrap_sharpes.append(sharpe)

        sharpe_ci = (np.percentile(bootstrap_sharpes, 2.5), np.percentile(bootstrap_sharpes, 97.5))

        # Bootstrap mean return
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(returns, size=len(returns), replace=True)
            bootstrap_means.append(np.mean(sample))

        bootstrap_mean = np.mean(bootstrap_means)
        bootstrap_std = np.std(bootstrap_means)

        # Monte Carlo probability of positive return
        mc_simulations = 10000
        mc_returns = np.random.choice(returns, size=(mc_simulations, len(returns)), replace=True)
        mc_total_returns = np.prod(1 + mc_returns, axis=1) - 1
        prob_positive = np.mean(mc_total_returns > 0)

        # Jarque-Bera normality test
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        is_normal = jb_pvalue > 0.05

        # P0: Calculate Deflated Sharpe Ratio (DSR)
        dsr, dsr_pvalue = self._calculate_deflated_sharpe(
            observed_sharpe=sharpe_ratio,
            returns=returns,
            n_trials=n_trials,
        )

        # P0: Calculate Probability of Backtest Overfitting (PBO)
        pbo, pbo_interpretation = self._calculate_pbo(returns)

        return StatisticalTests(
            returns_tstat=float(t_stat),
            returns_pvalue=float(p_value),
            sharpe_confidence_95=sharpe_ci,
            bootstrap_mean_return=float(bootstrap_mean),
            bootstrap_std_return=float(bootstrap_std),
            monte_carlo_prob_positive=float(prob_positive),
            jarque_bera_stat=float(jb_stat),
            jarque_bera_pvalue=float(jb_pvalue),
            is_normal=is_normal,
            # P0: DSR metrics
            deflated_sharpe_ratio=dsr,
            dsr_pvalue=dsr_pvalue,
            n_trials_tested=n_trials,
            # P0: PBO metrics
            pbo=pbo,
            pbo_interpretation=pbo_interpretation,
        )

    def _calculate_deflated_sharpe(
        self,
        observed_sharpe: float,
        returns: np.ndarray,
        n_trials: int = 1,
    ) -> tuple[float, float]:
        """P0: Calculate Deflated Sharpe Ratio (DSR).

        DSR accounts for the variance in backtest Sharpe ratios when
        multiple trials are run. Based on Bailey & López de Prado (2014).

        Args:
            observed_sharpe: The observed Sharpe ratio.
            returns: Array of returns for skewness/kurtosis.
            n_trials: Number of strategies/parameters tested.

        Returns:
            Tuple of (deflated_sharpe_ratio, p_value).
        """
        try:
            from quant_trading_system.models.purged_cv import MultipleTestingCorrector

            # Calculate return distribution characteristics
            skewness = float(stats.skew(returns)) if len(returns) > 2 else 0.0
            kurtosis = float(stats.kurtosis(returns)) + 3.0 if len(returns) > 3 else 3.0  # Excess -> raw

            corrector = MultipleTestingCorrector(n_trials=max(1, n_trials))
            dsr, dsr_pvalue = corrector.deflated_sharpe_ratio(
                observed_sharpe=observed_sharpe,
                n_trials=n_trials,
                skewness=skewness,
                kurtosis=kurtosis,
                n_returns=len(returns),
            )

            return float(dsr), float(dsr_pvalue)

        except ImportError:
            logger.warning("MultipleTestingCorrector not available, using fallback DSR")
            # Fallback: simple DSR approximation
            return observed_sharpe, 0.5

    def _calculate_pbo(
        self,
        returns: np.ndarray,
        n_partitions: int = 16,
    ) -> tuple[float, str]:
        """P0: Calculate Probability of Backtest Overfitting (PBO).

        PBO estimates the probability that a strategy's good backtest
        performance is due to overfitting rather than genuine skill.

        Uses Combinatorial Symmetric Cross-Validation (CSCV) approach
        from Bailey et al. (2014).

        Args:
            returns: Array of period returns.
            n_partitions: Number of partitions for CSCV (must be even).

        Returns:
            Tuple of (pbo, interpretation_string).
        """
        n = len(returns)

        # Need sufficient data for meaningful PBO
        if n < n_partitions * 5:
            return 0.5, "Insufficient data for PBO calculation"

        try:
            from itertools import combinations

            # Ensure even number of partitions
            n_partitions = n_partitions if n_partitions % 2 == 0 else n_partitions - 1
            if n_partitions < 4:
                return 0.5, "Too few partitions for PBO"

            # Split returns into partitions
            partition_size = n // n_partitions
            partitions = [
                returns[i * partition_size:(i + 1) * partition_size]
                for i in range(n_partitions)
            ]

            # Generate all combinations of train/test splits
            half = n_partitions // 2
            train_combos = list(combinations(range(n_partitions), half))

            # Limit combinations for computational efficiency
            max_combos = 100
            if len(train_combos) > max_combos:
                np.random.seed(42)  # Reproducibility
                indices = np.random.choice(len(train_combos), max_combos, replace=False)
                train_combos = [train_combos[i] for i in indices]

            logits = []

            for train_indices in train_combos:
                test_indices = [i for i in range(n_partitions) if i not in train_indices]

                # Aggregate train and test returns
                train_returns = np.concatenate([partitions[i] for i in train_indices])
                test_returns = np.concatenate([partitions[i] for i in test_indices])

                # Calculate Sharpe for this split
                train_sharpe = (
                    np.mean(train_returns) / np.std(train_returns) * np.sqrt(self.periods_per_year)
                    if np.std(train_returns) > 0 else 0
                )
                test_sharpe = (
                    np.mean(test_returns) / np.std(test_returns) * np.sqrt(self.periods_per_year)
                    if np.std(test_returns) > 0 else 0
                )

                # Compare performance: is OOS worse than IS?
                # Logit: positive if OOS underperforms
                if train_sharpe > 0:
                    performance_ratio = test_sharpe / train_sharpe if train_sharpe != 0 else 0
                    # Convert to logit: log(p / (1-p)) where p is "probability of overfit"
                    # If test_sharpe << train_sharpe, likely overfit
                    prob_overfit = 1.0 / (1.0 + performance_ratio) if performance_ratio > 0 else 0.9
                    prob_overfit = np.clip(prob_overfit, 0.01, 0.99)
                    logit = np.log(prob_overfit / (1 - prob_overfit))
                    logits.append(logit)

            if not logits:
                return 0.5, "Could not compute PBO logits"

            # PBO = probability that best IS strategy underperforms median OOS
            # Approximated by proportion of negative performance ratios
            pbo = np.mean(np.array(logits) > 0)

            # Interpretation
            if pbo < 0.15:
                interpretation = "LOW RISK: Strategy likely has genuine edge"
            elif pbo < 0.30:
                interpretation = "MODERATE RISK: Some overfitting possible"
            elif pbo < 0.50:
                interpretation = "HIGH RISK: Significant overfitting likely"
            else:
                interpretation = "CRITICAL: Strategy is almost certainly overfit"

            return float(pbo), interpretation

        except Exception as e:
            logger.warning(f"PBO calculation failed: {e}")
            return 0.5, f"PBO calculation error: {str(e)}"

    def analyze_by_regime(
        self,
        backtest_state: BacktestState,
        regime_history: list[tuple[datetime, Any]],
    ) -> list[RegimeMetrics]:
        """P1: Break down performance by market regime.

        Institutional-grade analysis that shows how strategy performs
        in different market conditions (bull, bear, high vol, etc.).

        Args:
            backtest_state: Completed backtest state.
            regime_history: List of (timestamp, RegimeState) tuples.

        Returns:
            List of RegimeMetrics, one per regime encountered.
        """
        from collections import defaultdict

        if not regime_history:
            return []

        # Extract equity curve
        equity_curve = np.array([e for _, e in backtest_state.equity_curve])
        timestamps = [t for t, _ in backtest_state.equity_curve]

        if len(equity_curve) < 2:
            return []

        # Calculate returns
        returns = np.diff(equity_curve) / equity_curve[:-1]

        # Map timestamps to indices
        timestamp_to_idx = {ts: i for i, ts in enumerate(timestamps[:-1])}

        # Group returns by regime
        regime_returns: dict[str, list[float]] = defaultdict(list)

        for ts, regime_state in regime_history:
            if ts in timestamp_to_idx:
                idx = timestamp_to_idx[ts]
                if idx < len(returns):
                    regime_name = (
                        regime_state.regime.name
                        if hasattr(regime_state, 'regime') and hasattr(regime_state.regime, 'name')
                        else str(regime_state)
                    )
                    regime_returns[regime_name].append(returns[idx])

        # Calculate metrics per regime
        regime_metrics = []
        for regime_name, rets in regime_returns.items():
            if len(rets) < 5:
                continue  # Need minimum data

            rets_array = np.array(rets)

            # Total return
            total_return = np.prod(1 + rets_array) - 1

            # Sharpe ratio
            if np.std(rets_array) > 0:
                sharpe = np.mean(rets_array) / np.std(rets_array) * np.sqrt(self.periods_per_year)
            else:
                sharpe = 0.0

            # Win rate
            win_rate = np.mean(rets_array > 0)

            # Max drawdown
            cumulative = np.cumprod(1 + rets_array)
            peak = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - peak) / peak
            max_dd = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0

            regime_metrics.append(RegimeMetrics(
                regime=regime_name,
                n_bars=len(rets),
                total_return=float(total_return),
                sharpe_ratio=float(sharpe),
                win_rate=float(win_rate),
                max_drawdown=float(max_dd),
            ))

        # Sort by number of bars (most data first)
        regime_metrics.sort(key=lambda x: x.n_bars, reverse=True)

        return regime_metrics

    def analyze_cost_sensitivity(
        self,
        backtest_state: BacktestState,
        cost_range_bps: list[float] | None = None,
    ) -> CostSensitivity:
        """P1: Analyze sensitivity to transaction costs.

        Determines break-even cost level and how strategy performance
        degrades with increasing costs.

        Args:
            backtest_state: Completed backtest state.
            cost_range_bps: List of cost levels to test (in basis points).

        Returns:
            CostSensitivity with break-even and cost impact analysis.
        """
        if cost_range_bps is None:
            cost_range_bps = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0]

        # Extract equity curve
        equity_curve = np.array([e for _, e in backtest_state.equity_curve])

        if len(equity_curve) < 2:
            return CostSensitivity(
                break_even_cost_bps=0.0,
                sharpe_at_costs={},
                return_at_costs={},
            )

        # Base returns (already includes some costs)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        base_return = np.prod(1 + returns) - 1
        base_sharpe = (
            np.mean(returns) / np.std(returns) * np.sqrt(self.periods_per_year)
            if np.std(returns) > 0 else 0
        )

        # Number of trades determines cost impact
        n_trades = len(backtest_state.trades) if backtest_state.trades else 0

        if n_trades == 0:
            return CostSensitivity(
                break_even_cost_bps=float('inf'),
                sharpe_at_costs={c: base_sharpe for c in cost_range_bps},
                return_at_costs={c: base_return for c in cost_range_bps},
            )

        # Estimate cost impact per trade
        avg_trade_size = base_return / n_trades if n_trades > 0 else 0

        sharpe_at_costs = {}
        return_at_costs = {}
        break_even_cost = 0.0

        for cost_bps in cost_range_bps:
            # Cost impact per trade (bps to decimal)
            cost_per_trade = cost_bps / 10000

            # Adjust returns by subtracting cost per bar with trades
            # Simplified: distribute cost impact across all bars proportionally
            total_cost = n_trades * cost_per_trade
            cost_per_bar = total_cost / len(returns) if len(returns) > 0 else 0

            adjusted_returns = returns - cost_per_bar
            adjusted_total_return = np.prod(1 + adjusted_returns) - 1

            if np.std(adjusted_returns) > 0:
                adjusted_sharpe = (
                    np.mean(adjusted_returns) / np.std(adjusted_returns) * np.sqrt(self.periods_per_year)
                )
            else:
                adjusted_sharpe = 0.0

            sharpe_at_costs[cost_bps] = float(adjusted_sharpe)
            return_at_costs[cost_bps] = float(adjusted_total_return)

            # Track break-even (where returns go negative)
            if adjusted_total_return > 0:
                break_even_cost = cost_bps

        return CostSensitivity(
            break_even_cost_bps=float(break_even_cost),
            sharpe_at_costs=sharpe_at_costs,
            return_at_costs=return_at_costs,
        )


class VisualizationData:
    """Prepares data for visualization."""

    @staticmethod
    def get_equity_curve_data(
        backtest_state: BacktestState,
    ) -> pd.DataFrame:
        """Get equity curve data for plotting."""
        data = []
        for timestamp, equity in backtest_state.equity_curve:
            data.append({"timestamp": timestamp, "equity": equity})
        return pd.DataFrame(data)

    @staticmethod
    def get_drawdown_data(
        backtest_state: BacktestState,
    ) -> pd.DataFrame:
        """Get drawdown data for plotting."""
        equity = np.array([e for _, e in backtest_state.equity_curve])
        timestamps = [t for t, _ in backtest_state.equity_curve]

        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak

        data = []
        for i, (ts, dd) in enumerate(zip(timestamps, drawdown)):
            data.append({"timestamp": ts, "drawdown": dd, "equity": equity[i], "peak": peak[i]})
        return pd.DataFrame(data)

    @staticmethod
    def get_monthly_returns_heatmap(
        backtest_state: BacktestState,
    ) -> pd.DataFrame:
        """Get monthly returns for heatmap visualization."""
        if len(backtest_state.equity_curve) < 2:
            return pd.DataFrame()

        # Create DataFrame
        data = []
        for timestamp, equity in backtest_state.equity_curve:
            data.append({"timestamp": timestamp, "equity": equity})
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        # Resample to monthly
        monthly = df["equity"].resample("ME").last()
        monthly_returns = monthly.pct_change().dropna()

        # Create pivot table
        monthly_returns = monthly_returns.to_frame()
        monthly_returns["year"] = monthly_returns.index.year
        monthly_returns["month"] = monthly_returns.index.month

        return monthly_returns.pivot(index="year", columns="month", values="equity")

    @staticmethod
    def get_trade_distribution_data(
        trades: list[Trade],
    ) -> pd.DataFrame:
        """Get trade P&L distribution data."""
        data = []
        for trade in trades:
            data.append({
                "pnl": float(trade.pnl),
                "pnl_pct": trade.pnl_pct,
                "duration": trade.holding_period_bars,
                "symbol": trade.symbol,
            })
        return pd.DataFrame(data)
