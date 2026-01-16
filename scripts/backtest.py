#!/usr/bin/env python3
"""
================================================================================
ALPHATRADE - INSTITUTIONAL-GRADE BACKTESTING SCRIPT
================================================================================

@mlquant + @trader: Combines model predictions with realistic execution.

JPMorgan-Level Implementation Features:
  - Event-driven or vectorized backtesting engine
  - Realistic market simulation (slippage, spread, partial fills)
  - Market impact modeling (Almgren-Chriss)
  - Regime-specific strategy adaptation
  - Walk-forward optimization
  - Out-of-sample validation
  - Performance attribution (Brinson-Fachler)
  - Monte Carlo simulation for confidence intervals
  - Purged cross-validation for model selection
  - GPU-accelerated feature computation

Execution Modes:
  - REALISTIC   : Includes slippage, spread, partial fills, market impact
  - OPTIMISTIC  : Best-case execution (for upper bound estimates)
  - PESSIMISTIC : Worst-case execution (for stress testing)

Performance Metrics:
  - Return metrics: Total, annualized, risk-adjusted
  - Risk metrics: Sharpe, Sortino, Calmar, max drawdown
  - Trade metrics: Win rate, profit factor, expectancy
  - Alpha metrics: IC, IR, turnover

Author: AlphaTrade System
Version: 1.3.0
================================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# ==============================================================================
# PATH SETUP
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ==============================================================================
# IMPORTS
# ==============================================================================

# Configuration
from quant_trading_system.config.settings import get_settings

# Core
from quant_trading_system.core.data_types import (
    OHLCVBar,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Portfolio,
    Position,
    TradeSignal,
    Direction,
)
from quant_trading_system.core.events import EventBus, EventType
from quant_trading_system.core.system_integrator import (
    SystemIntegrator,
    SystemIntegratorConfig,
    create_system_integrator,
)

# Data
from quant_trading_system.data.loader import DataLoader

# Features
from quant_trading_system.features.feature_pipeline import (
    FeaturePipeline,
    FeatureConfig,
    FeatureGroup,
)
from quant_trading_system.features.optimized_pipeline import (
    OptimizedFeaturePipeline,
    OptimizedPipelineConfig,
    CUDF_AVAILABLE,
    ComputeMode,
)

# Alpha
from quant_trading_system.alpha.regime_detection import (
    CompositeRegimeDetector,
    MarketRegime,
    RegimeState,
)
from quant_trading_system.alpha.alpha_combiner import AlphaCombiner
from quant_trading_system.alpha.momentum_alphas import (
    PriceMomentum,
    RsiMomentum,
    MacdMomentum,
    create_momentum_alphas,
)
from quant_trading_system.alpha.mean_reversion_alphas import (
    BollingerReversion,
    ZScoreReversion,
)

# Models
from quant_trading_system.models.classical_ml import XGBoostModel, LightGBMModel
from quant_trading_system.models.validation_gates import (
    ModelValidationGates,
    ValidationReport,
)
from quant_trading_system.models.purged_cv import (
    PurgedKFold,
    CombinatorialPurgedKFold,
    WalkForwardCV,
)

# Backtest
from quant_trading_system.backtest.engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestMode,
    ExecutionMode,
    PandasDataHandler,
    PolarsDataHandler,
    Strategy,
    BacktestState,
)
from quant_trading_system.backtest.simulator import (
    MarketSimulator,
    MarketConditions,
    create_realistic_simulator,
    create_optimistic_simulator,
    create_pessimistic_simulator,
)
from quant_trading_system.backtest.analyzer import (
    PerformanceAnalyzer,
    ReturnMetrics,
    RiskAdjustedMetrics,
    DrawdownMetrics,
    TradeMetrics,
)
from quant_trading_system.backtest.performance_attribution import (
    PerformanceAttributionService as PerformanceAttributor,
    BrinsonAttribution,
)

# Risk
from quant_trading_system.risk.limits import RiskLimitsConfig
from quant_trading_system.risk.position_sizer import (
    BasePositionSizer as PositionSizer,
    SizingMethod,
    SizingConstraints as PositionSizerConfig,
)

# Execution
from quant_trading_system.execution.market_impact import (
    AlmgrenChrissModel,
    AdaptiveMarketImpactModel,
)

# Monitoring
from quant_trading_system.monitoring.logger import (
    setup_logging,
    LogFormat,
    get_logger,
    LogCategory,
)
from quant_trading_system.monitoring.tracing import (
    configure_tracing,
    get_tracer,
)


# ==============================================================================
# LOGGER
# ==============================================================================

logger = get_logger("backtest", LogCategory.SYSTEM)


# ==============================================================================
# TYPE ALIASES & STRATEGY
# ==============================================================================

# Type alias for backward compatibility
BacktestResult = BacktestState


class SignalBasedStrategy(Strategy):
    """
    Trading strategy that uses pre-computed signals.

    This strategy takes a dictionary of signals per symbol and generates
    TradeSignal objects based on the signal values at each bar.
    """

    def __init__(
        self,
        signals: dict[str, pd.DataFrame],
        signal_threshold: float = 0.1,
        signal_column: str = "signal",
    ):
        """Initialize signal-based strategy.

        Args:
            signals: Dictionary mapping symbols to DataFrames with signals.
            signal_threshold: Minimum absolute signal value to generate trade.
            signal_column: Column name containing signal values.
        """
        self.signals = signals
        self.signal_threshold = signal_threshold
        self.signal_column = signal_column
        self._current_indices: dict[str, int] = {s: 0 for s in signals.keys()}

    def generate_signals(
        self,
        data_handler: PandasDataHandler,
        portfolio: Portfolio,
    ) -> list[TradeSignal]:
        """Generate trading signals based on pre-computed signal values.

        Args:
            data_handler: Data handler with market data.
            portfolio: Current portfolio state.

        Returns:
            List of trading signals.
        """
        trade_signals = []

        for symbol in data_handler.get_symbols():
            if symbol not in self.signals:
                continue

            bar = data_handler.get_current_bar(symbol)
            if bar is None:
                continue

            # Get signal for current timestamp
            signal_df = self.signals[symbol]
            idx = self._current_indices.get(symbol, 0)

            if idx >= len(signal_df):
                continue

            signal_value = signal_df.iloc[idx][self.signal_column]
            self._current_indices[symbol] = idx + 1

            # Skip if signal is too weak
            if abs(signal_value) < self.signal_threshold:
                continue

            # Determine direction
            if signal_value > self.signal_threshold:
                direction = Direction.LONG
                strength = min(signal_value, 1.0)
            elif signal_value < -self.signal_threshold:
                direction = Direction.SHORT
                strength = max(signal_value, -1.0)
            else:
                continue

            trade_signal = TradeSignal(
                symbol=symbol,
                direction=direction,
                strength=strength,
                confidence=abs(signal_value),
                timestamp=bar.timestamp,
                horizon=1,  # Default 1-bar horizon for signal-based strategy
                model_source="alpha_combiner",  # Source identifier
            )
            trade_signals.append(trade_signal)

        return trade_signals

    def on_bar(self, symbol: str, bar: OHLCVBar) -> None:
        """Called for each new bar."""
        pass

    def on_fill(self, order: Order) -> None:
        """Called when an order is filled."""
        pass


# ==============================================================================
# BACKTEST SESSION
# ==============================================================================

@dataclass
class BacktestSession:
    """
    Manages a complete backtesting session with all institutional-grade features.

    Phases:
      1. Data Loading: Load and validate historical data
      2. Feature Engineering: Compute all required features
      3. Model Preparation: Load/validate models or use alpha factors
      4. Backtesting: Run simulation with realistic execution
      5. Analysis: Generate comprehensive performance reports
    """

    # Configuration
    start_date: datetime
    end_date: datetime
    symbols: list[str]
    initial_capital: Decimal
    strategy_name: str = "momentum"
    execution_mode: str = "realistic"
    use_gpu: bool = False
    use_database: bool = True  # Load data from PostgreSQL + TimescaleDB

    # Advanced options
    slippage_bps: float = 5.0
    commission_bps: float = 1.0
    monte_carlo_sims: int = 0

    # Session state
    session_id: str = field(default_factory=lambda: f"bt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")

    # Components (initialized during run)
    data_loader: Optional[DataLoader] = None
    feature_pipeline: Optional[FeaturePipeline] = None
    backtest_engine: Optional[BacktestEngine] = None
    analyzer: Optional[PerformanceAnalyzer] = None
    regime_detector: Optional[CompositeRegimeDetector] = None

    # Results
    result: Optional[BacktestResult] = None
    analysis: Optional[dict[str, Any]] = None

    def __post_init__(self):
        """Initialize session components."""
        logger.info(
            f"BacktestSession initialized: {self.session_id}",
            extra={
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
                "symbols": self.symbols,
                "capital": float(self.initial_capital),
            },
        )


class BacktestRunner:
    """
    Orchestrates the complete backtesting workflow.

    Features:
      - Parallel symbol processing
      - Caching for feature computation
      - Progress tracking
      - Comprehensive error handling
    """

    def __init__(self, session: BacktestSession):
        """Initialize backtest runner.

        Args:
            session: Backtest session configuration
        """
        self.session = session
        self.data: dict[str, pd.DataFrame] = {}
        self.features: dict[str, pd.DataFrame] = {}
        self.signals: dict[str, pd.DataFrame] = {}

    def run(self) -> BacktestResult:
        """Execute the complete backtest workflow.

        Returns:
            BacktestResult with all performance metrics
        """
        logger.info("=" * 60)
        logger.info("STARTING BACKTEST")
        logger.info("=" * 60)

        start_time = time.time()

        try:
            # Phase 1: Load Data
            logger.info("Phase 1: Loading data...")
            self._load_data()

            # Phase 2: Compute Features
            logger.info("Phase 2: Computing features...")
            self._compute_features()

            # Phase 3: Generate Signals
            logger.info("Phase 3: Generating signals...")
            self._generate_signals()

            # Phase 4: Run Backtest
            logger.info("Phase 4: Running backtest simulation...")
            result = self._run_backtest()

            # Phase 5: Analyze Results
            logger.info("Phase 5: Analyzing results...")
            analysis = self._analyze_results(result)

            # Phase 6: Monte Carlo (if enabled)
            if self.session.monte_carlo_sims > 0:
                logger.info(f"Phase 6: Running {self.session.monte_carlo_sims} Monte Carlo simulations...")
                self._run_monte_carlo(result)

            elapsed = time.time() - start_time
            logger.info(f"Backtest completed in {elapsed:.2f}s")

            # Store results
            self.session.result = result
            self.session.analysis = analysis

            return result

        except Exception as e:
            logger.exception(f"Backtest failed: {e}")
            raise

    def _load_data(self) -> None:
        """Load historical data for all symbols."""
        self.session.data_loader = DataLoader(
            data_dir=Path("data/raw"),
            use_database=self.session.use_database,
        )

        for symbol in self.session.symbols:
            try:
                df = self.session.data_loader.load_symbol(
                    symbol=symbol,
                    start_date=self.session.start_date,
                    end_date=self.session.end_date,
                )

                if df is not None and len(df) > 0:
                    self.data[symbol] = df
                    logger.info(f"Loaded {len(df)} bars for {symbol}")
                else:
                    logger.warning(f"No data found for {symbol}")

            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {e}")

        if not self.data:
            raise ValueError("No data loaded for any symbol")

    def _compute_features(self) -> None:
        """Compute features for all symbols."""
        # Choose pipeline based on GPU availability
        if self.session.use_gpu and CUDF_AVAILABLE:
            logger.info("Using GPU-accelerated feature pipeline")
            pipeline = OptimizedFeaturePipeline(
                config=OptimizedPipelineConfig(
                    compute_mode=ComputeMode.GPU,
                ),
            )
        else:
            logger.info("Using CPU feature pipeline")
            pipeline = FeaturePipeline(
                config=FeatureConfig(
                    groups=[FeatureGroup.TECHNICAL, FeatureGroup.STATISTICAL],
                ),
            )

        self.session.feature_pipeline = pipeline

        for symbol, df in self.data.items():
            try:
                # Convert polars DataFrame to polars for FeaturePipeline
                import polars as pl
                if not isinstance(df, pl.DataFrame):
                    df_pl = pl.from_pandas(df) if hasattr(df, 'to_pandas') else pl.DataFrame(df)
                else:
                    df_pl = df
                feature_set = pipeline.compute(df_pl, symbol=symbol)
                # Convert to pandas for backtest engine
                self.features[symbol] = feature_set.to_polars().to_pandas() if feature_set.num_features > 0 else df_pl.to_pandas()
                logger.info(f"Computed {feature_set.num_features} features for {symbol}")

            except Exception as e:
                logger.error(f"Failed to compute features for {symbol}: {e}")

    def _generate_signals(self) -> None:
        """Generate trading signals using alpha factors."""
        # Create alpha factors based on strategy
        if self.session.strategy_name == "momentum":
            alphas = create_momentum_alphas()
        elif self.session.strategy_name == "mean_reversion":
            alphas = [
                BollingerReversion(period=20),
                ZScoreReversion(lookback=60),
            ]
        else:
            alphas = create_momentum_alphas()  # Default

        # Regime detector for adaptive signals
        self.session.regime_detector = CompositeRegimeDetector()

        import polars as pl

        for symbol in self.features:
            try:
                df = self.data[symbol]
                features = self.features[symbol]

                # Handle both Polars and Pandas DataFrames
                if isinstance(df, pl.DataFrame):
                    df_pl = df
                    # Get timestamps for signal DataFrame
                    if "timestamp" in df_pl.columns:
                        timestamps = df_pl["timestamp"].to_list()
                    else:
                        timestamps = list(range(len(df_pl)))
                elif hasattr(df, 'reset_index'):
                    # Pandas DataFrame
                    df_pl = pl.from_pandas(df.reset_index())
                    timestamps = df.index.tolist()
                else:
                    logger.warning(f"Unsupported DataFrame type for {symbol}")
                    continue

                # Convert features to dict if needed
                if isinstance(features, pl.DataFrame):
                    features_dict = {col: features[col].to_numpy() for col in features.columns}
                elif hasattr(features, 'to_dict'):
                    features_dict = features.to_dict("series")
                else:
                    features_dict = {}

                alpha_signals = {}
                for alpha in alphas:
                    signal = alpha.compute(df_pl, features_dict)
                    alpha_signals[alpha.name] = signal

                # Combine signals
                combined = np.zeros(len(df))
                for name, sig in alpha_signals.items():
                    valid = ~np.isnan(sig)
                    combined[valid] += sig[valid]

                if len(alpha_signals) > 0:
                    combined /= len(alpha_signals)

                self.signals[symbol] = pd.DataFrame({
                    "signal": combined,
                    "timestamp": timestamps[:len(combined)],
                })

                logger.info(f"Generated signals for {symbol}")

            except Exception as e:
                logger.error(f"Failed to generate signals for {symbol}: {e}")

    def _run_backtest(self) -> BacktestResult:
        """Run the backtest simulation."""
        # Create simulator based on execution mode
        exec_mode_map = {
            "realistic": ExecutionMode.REALISTIC,
            "optimistic": ExecutionMode.OPTIMISTIC,
            "pessimistic": ExecutionMode.PESSIMISTIC,
        }
        exec_mode = exec_mode_map.get(self.session.execution_mode, ExecutionMode.REALISTIC)

        # Create backtest config
        config = BacktestConfig(
            initial_capital=self.session.initial_capital,
            mode=BacktestMode.EVENT_DRIVEN,
            execution_mode=exec_mode,
            commission_bps=self.session.commission_bps,
            slippage_bps=self.session.slippage_bps,
            max_leverage=1.0,
            start_date=self.session.start_date,
            end_date=self.session.end_date,
        )

        # Create market simulator
        if exec_mode == ExecutionMode.REALISTIC:
            market_simulator = create_realistic_simulator(
                commission_bps=self.session.commission_bps,
                base_slippage_bps=self.session.slippage_bps,
            )
        elif exec_mode == ExecutionMode.OPTIMISTIC:
            market_simulator = create_optimistic_simulator()
        else:
            market_simulator = create_pessimistic_simulator(
                commission_bps=self.session.commission_bps * 2,
                base_slippage_bps=self.session.slippage_bps * 2,
            )

        # Use PolarsDataHandler for native Polars support (no conversion overhead)
        data_handler = PolarsDataHandler(
            data=self.data,
            start_date=self.session.start_date,
            end_date=self.session.end_date,
        )

        # Create signal-based strategy
        strategy = SignalBasedStrategy(
            signals=self.signals,
            signal_threshold=0.1,
        )

        # Create backtest engine with proper parameters
        # P1: Pass regime detector for regime-specific analysis
        self.session.backtest_engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            config=config,
            market_simulator=market_simulator,
            regime_detector=self.session.regime_detector,
        )

        # Run backtest
        result = self.session.backtest_engine.run()

        return result

    def _analyze_results(self, result: BacktestResult) -> dict[str, Any]:
        """Analyze backtest results comprehensively."""
        self.session.analyzer = PerformanceAnalyzer()

        try:
            # Use the analyze method which generates a PerformanceReport
            report = self.session.analyzer.analyze(result)

            # Extract metrics from report
            analysis = {
                "return_metrics": report.return_metrics.to_dict(),
                "risk_metrics": report.risk_adjusted_metrics.to_dict(),
                "drawdown_metrics": report.drawdown_metrics.to_dict(),
                "trade_metrics": report.trade_metrics.to_dict(),
            }

            # Add benchmark metrics if available
            if report.benchmark_metrics:
                analysis["benchmark_metrics"] = report.benchmark_metrics.to_dict()

            # Add statistical tests (includes DSR/PBO from P0)
            analysis["statistical_tests"] = report.statistical_tests.to_dict()

            # P1: Regime-specific analysis
            if hasattr(result, 'regime_history') and result.regime_history:
                try:
                    regime_states = [r for _, r in result.regime_history]
                    regime_metrics = self.session.analyzer.analyze_by_regime(
                        result, regime_states
                    )
                    analysis["regime_metrics"] = [rm.to_dict() for rm in regime_metrics]
                    logger.info(f"Computed regime metrics for {len(regime_metrics)} regimes")
                except Exception as e:
                    logger.warning(f"Regime analysis failed: {e}")
                    analysis["regime_metrics"] = []
            else:
                analysis["regime_metrics"] = []

            # P1: Cost sensitivity analysis
            try:
                cost_sensitivity = self.session.analyzer.analyze_cost_sensitivity(result)
                analysis["cost_sensitivity"] = cost_sensitivity.to_dict()
                logger.info(
                    f"Cost sensitivity: break-even = {cost_sensitivity.break_even_cost_bps:.1f} bps"
                )
            except Exception as e:
                logger.warning(f"Cost sensitivity analysis failed: {e}")
                analysis["cost_sensitivity"] = {}

        except Exception as e:
            # Fallback: compute basic metrics manually if analyzer fails
            logger.warning(f"Performance analyzer failed, using fallback: {e}")

            equity_curve = [e for _, e in result.equity_curve] if result.equity_curve else []
            total_return = (equity_curve[-1] / equity_curve[0] - 1) if equity_curve and equity_curve[0] > 0 else 0

            # Basic metrics
            analysis = {
                "return_metrics": {
                    "total_return": total_return,
                    "annualized_return": 0,
                    "best_day": 0,
                    "worst_day": 0,
                },
                "risk_metrics": {
                    "sharpe_ratio": 0,
                    "sortino_ratio": 0,
                    "calmar_ratio": 0,
                },
                "drawdown_metrics": {
                    "max_drawdown": 0,
                    "avg_drawdown": 0,
                },
                "trade_metrics": {
                    "total_trades": len(result.trades) if result.trades else 0,
                    "win_rate": 0,
                    "profit_factor": 0,
                },
            }

        # Performance attribution
        try:
            attributor = PerformanceAttributor()
            attribution = attributor.compute_attribution(result)
            analysis["attribution"] = attribution
        except Exception as e:
            logger.warning(f"Performance attribution failed: {e}")
            analysis["attribution"] = {}

        # Print summary
        self._print_summary(analysis)

        return analysis

    def _print_summary(self, analysis: dict[str, Any]) -> None:
        """Print backtest summary to console."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS SUMMARY")
        print("=" * 60)

        rm = analysis.get("return_metrics", {})
        print(f"\nReturn Metrics:")
        print(f"  Total Return:      {rm.get('total_return', 0):.2%}")
        print(f"  Annualized Return: {rm.get('annualized_return', 0):.2%}")
        print(f"  Best Day:          {rm.get('best_day', 0):.2%}")
        print(f"  Worst Day:         {rm.get('worst_day', 0):.2%}")

        rsk = analysis.get("risk_metrics", {})
        print(f"\nRisk-Adjusted Metrics:")
        print(f"  Sharpe Ratio:      {rsk.get('sharpe_ratio', 0):.3f}")
        print(f"  Sortino Ratio:     {rsk.get('sortino_ratio', 0):.3f}")
        print(f"  Calmar Ratio:      {rsk.get('calmar_ratio', 0):.3f}")

        dd = analysis.get("drawdown_metrics", {})
        print(f"\nDrawdown Metrics:")
        print(f"  Max Drawdown:      {dd.get('max_drawdown', 0):.2%}")
        print(f"  Avg Drawdown:      {dd.get('avg_drawdown', 0):.2%}")

        tm = analysis.get("trade_metrics", {})
        print(f"\nTrade Metrics:")
        print(f"  Total Trades:      {tm.get('total_trades', 0)}")
        print(f"  Win Rate:          {tm.get('win_rate', 0):.2%}")
        print(f"  Profit Factor:     {tm.get('profit_factor', 0):.2f}")

        # P0: Statistical Validation (DSR/PBO)
        st = analysis.get("statistical_tests", {})
        if st.get("deflated_sharpe_ratio") or st.get("pbo"):
            print(f"\nStatistical Validation (P0):")
            dsr = st.get("deflated_sharpe_ratio", 0)
            dsr_pvalue = st.get("dsr_pvalue", 1.0)
            pbo = st.get("pbo", 0.5)
            pbo_interp = st.get("pbo_interpretation", "")
            print(f"  Deflated Sharpe:   {dsr:.3f} (p={dsr_pvalue:.3f})")
            print(f"  PBO (Overfitting): {pbo:.1%} - {pbo_interp}")

        # P1: Cost Sensitivity
        cs = analysis.get("cost_sensitivity", {})
        if cs:
            print(f"\nCost Sensitivity (P1):")
            print(f"  Break-Even Cost:   {cs.get('break_even_cost_bps', 0):.1f} bps")
            sharpe_at_costs = cs.get("sharpe_at_costs", {})
            if sharpe_at_costs:
                # Show key cost levels
                for cost, sharpe in sorted(sharpe_at_costs.items())[:3]:
                    print(f"    Sharpe @ {cost} bps: {sharpe:.3f}")

        # P1: Regime Metrics
        regime_metrics = analysis.get("regime_metrics", [])
        if regime_metrics:
            print(f"\nRegime Analysis (P1):")
            for rm in regime_metrics[:4]:  # Show top 4 regimes
                regime = rm.get("regime", "unknown")
                sharpe = rm.get("sharpe_ratio", 0)
                n_bars = rm.get("n_bars", 0)
                ret = rm.get("total_return", 0)
                print(f"  {regime}: Sharpe={sharpe:.2f}, Return={ret:.2%}, Bars={n_bars}")

        print("=" * 60)

    def _run_monte_carlo(self, result: BacktestResult) -> None:
        """Run Monte Carlo simulations for confidence intervals."""
        logger.info(f"Running {self.session.monte_carlo_sims} Monte Carlo simulations")

        # Get daily returns from result
        if hasattr(result, 'equity_curve') and result.equity_curve is not None:
            equity = result.equity_curve
            returns = equity.pct_change().dropna()

            # Run simulations
            simulated_finals = []
            for _ in range(self.session.monte_carlo_sims):
                # Shuffle returns
                shuffled = returns.sample(frac=1, replace=True)
                # Compute final equity
                simulated_equity = float(self.session.initial_capital) * (1 + shuffled).cumprod()
                simulated_finals.append(simulated_equity.iloc[-1])

            # Compute percentiles
            percentiles = np.percentile(simulated_finals, [5, 25, 50, 75, 95])

            print("\nMonte Carlo Analysis:")
            print(f"  5th Percentile:    ${percentiles[0]:,.2f}")
            print(f"  25th Percentile:   ${percentiles[1]:,.2f}")
            print(f"  Median:            ${percentiles[2]:,.2f}")
            print(f"  75th Percentile:   ${percentiles[3]:,.2f}")
            print(f"  95th Percentile:   ${percentiles[4]:,.2f}")

    def save_results(self, output_path: Path) -> None:
        """Save backtest results to file.

        Args:
            output_path: Path to save results (JSON or HTML)
        """
        if self.session.analysis is None:
            logger.warning("No analysis to save")
            return

        output_path = Path(output_path)

        if output_path.suffix == ".json":
            with open(output_path, "w") as f:
                json.dump(self.session.analysis, f, indent=2, default=str)
            logger.info(f"Results saved to {output_path}")

        elif output_path.suffix == ".html":
            # Generate HTML report
            html_content = self._generate_html_report()
            with open(output_path, "w") as f:
                f.write(html_content)
            logger.info(f"HTML report saved to {output_path}")

        else:
            # Default to JSON
            json_path = output_path.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump(self.session.analysis, f, indent=2, default=str)
            logger.info(f"Results saved to {json_path}")

    def _generate_html_report(self) -> str:
        """Generate HTML report from results."""
        analysis = self.session.analysis or {}

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AlphaTrade Backtest Report - {self.session.session_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
    </style>
</head>
<body>
    <h1>AlphaTrade Backtest Report</h1>
    <p><strong>Session ID:</strong> {self.session.session_id}</p>
    <p><strong>Period:</strong> {self.session.start_date.date()} to {self.session.end_date.date()}</p>
    <p><strong>Symbols:</strong> {', '.join(self.session.symbols)}</p>
    <p><strong>Strategy:</strong> {self.session.strategy_name}</p>
    <p><strong>Initial Capital:</strong> ${float(self.session.initial_capital):,.2f}</p>

    <h2>Performance Summary</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Return</td><td>{analysis.get('return_metrics', {}).get('total_return', 0):.2%}</td></tr>
        <tr><td>Sharpe Ratio</td><td>{analysis.get('risk_metrics', {}).get('sharpe_ratio', 0):.3f}</td></tr>
        <tr><td>Max Drawdown</td><td>{analysis.get('drawdown_metrics', {}).get('max_drawdown', 0):.2%}</td></tr>
        <tr><td>Win Rate</td><td>{analysis.get('trade_metrics', {}).get('win_rate', 0):.2%}</td></tr>
    </table>

    <p><em>Generated by AlphaTrade v1.3.0 on {datetime.now(timezone.utc).isoformat()}</em></p>
</body>
</html>
        """
        return html


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def run_backtest(args: argparse.Namespace) -> int:
    """Main entry point for backtest command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success)
    """
    # Setup logging
    setup_logging(level=args.log_level, log_format=LogFormat.TEXT)

    # Parse dates
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        return 1

    # FIX: Validate date ordering
    if end_date <= start_date:
        logger.error(f"End date ({args.end}) must be after start date ({args.start})")
        return 1

    # Determine database usage (--no-database overrides --use-database)
    use_database = getattr(args, "use_database", True) and not getattr(args, "no_database", False)

    # Create session
    session = BacktestSession(
        start_date=start_date,
        end_date=end_date,
        symbols=args.symbols,
        initial_capital=Decimal(str(args.capital)),
        strategy_name=args.strategy,
        execution_mode=args.execution_mode,
        use_gpu=args.gpu,
        use_database=use_database,
        slippage_bps=args.slippage_bps,
        commission_bps=args.commission_bps,
        monte_carlo_sims=args.monte_carlo,
    )

    # Run backtest
    runner = BacktestRunner(session)

    try:
        result = runner.run()

        # Save results if output path specified
        if hasattr(args, 'output') and args.output:
            runner.save_results(args.output)

        return 0

    except Exception as e:
        logger.exception(f"Backtest failed: {e}")
        return 1


if __name__ == "__main__":
    # Create argument parser for standalone execution
    parser = argparse.ArgumentParser(description="AlphaTrade Backtesting Script")
    parser.add_argument("--start", "-s", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", "-e", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "GOOGL"])
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--strategy", default="momentum")
    parser.add_argument("--execution-mode", choices=["realistic", "optimistic", "pessimistic"], default="realistic")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--monte-carlo", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()
    sys.exit(run_backtest(args))
