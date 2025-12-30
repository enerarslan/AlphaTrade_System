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
    """Statistical significance tests."""

    returns_tstat: float
    returns_pvalue: float
    sharpe_confidence_95: tuple[float, float]
    bootstrap_mean_return: float
    bootstrap_std_return: float
    monte_carlo_prob_positive: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    is_normal: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "returns_tstat": self.returns_tstat,
            "returns_pvalue": self.returns_pvalue,
            "sharpe_confidence_95": list(self.sharpe_confidence_95),
            "bootstrap_mean_return": self.bootstrap_mean_return,
            "monte_carlo_prob_positive": self.monte_carlo_prob_positive,
            "is_normal": self.is_normal,
        }


@dataclass
class PerformanceReport:
    """Complete performance report."""

    return_metrics: ReturnMetrics
    risk_adjusted_metrics: RiskAdjustedMetrics
    drawdown_metrics: DrawdownMetrics
    trade_metrics: TradeMetrics
    benchmark_metrics: BenchmarkMetrics | None
    statistical_tests: StatisticalTests
    start_date: datetime
    end_date: datetime
    trading_days: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
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

    def summary(self) -> str:
        """Get a summary string of key metrics."""
        return f"""
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
"""


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
    ) -> PerformanceReport:
        """Generate complete performance report.

        Args:
            backtest_state: Completed backtest state.
            benchmark_returns: Optional benchmark returns for comparison.

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

        statistical_tests = self._calculate_statistical_tests(returns)

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
    ) -> StatisticalTests:
        """Calculate statistical significance tests."""
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
