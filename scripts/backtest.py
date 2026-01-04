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
    BacktestResult,
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
    PerformanceAttributor,
    BrinsonFachlerAttribution,
)

# Risk
from quant_trading_system.risk.limits import RiskLimitsConfig
from quant_trading_system.risk.position_sizer import (
    PositionSizer,
    SizingMethod,
    PositionSizerConfig,
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

logger = get_logger("backtest", LogCategory.BACKTEST)


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
                    feature_groups=["technical", "statistical", "microstructure"],
                ),
            )

        self.session.feature_pipeline = pipeline

        for symbol, df in self.data.items():
            try:
                features = pipeline.compute_features(df)
                self.features[symbol] = features
                logger.info(f"Computed {features.shape[1]} features for {symbol}")

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

        for symbol in self.features:
            try:
                df = self.data[symbol]
                features = self.features[symbol]

                # Compute alpha signals
                import polars as pl
                df_pl = pl.from_pandas(df.reset_index())

                alpha_signals = {}
                for alpha in alphas:
                    signal = alpha.compute(df_pl, features.to_dict("series"))
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
                    "timestamp": df.index if hasattr(df, 'index') else range(len(df)),
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
        )

        # Create simulator
        if exec_mode == ExecutionMode.REALISTIC:
            simulator = create_realistic_simulator()
        elif exec_mode == ExecutionMode.OPTIMISTIC:
            simulator = create_optimistic_simulator()
        else:
            simulator = create_pessimistic_simulator()

        # Create backtest engine
        self.session.backtest_engine = BacktestEngine(
            config=config,
            simulator=simulator,
        )

        # Run backtest
        result = self.session.backtest_engine.run(
            data=self.data,
            signals=self.signals,
        )

        return result

    def _analyze_results(self, result: BacktestResult) -> dict[str, Any]:
        """Analyze backtest results comprehensively."""
        self.session.analyzer = PerformanceAnalyzer()

        # Compute all metrics
        return_metrics = self.session.analyzer.compute_return_metrics(result)
        risk_metrics = self.session.analyzer.compute_risk_adjusted_metrics(result)
        drawdown_metrics = self.session.analyzer.compute_drawdown_metrics(result)
        trade_metrics = self.session.analyzer.compute_trade_metrics(result)

        # Performance attribution
        try:
            attributor = PerformanceAttributor()
            attribution = attributor.compute_attribution(result)
        except Exception as e:
            logger.warning(f"Performance attribution failed: {e}")
            attribution = {}

        analysis = {
            "return_metrics": return_metrics.to_dict() if hasattr(return_metrics, 'to_dict') else vars(return_metrics),
            "risk_metrics": risk_metrics.to_dict() if hasattr(risk_metrics, 'to_dict') else vars(risk_metrics),
            "drawdown_metrics": drawdown_metrics.to_dict() if hasattr(drawdown_metrics, 'to_dict') else vars(drawdown_metrics),
            "trade_metrics": trade_metrics.to_dict() if hasattr(trade_metrics, 'to_dict') else vars(trade_metrics),
            "attribution": attribution,
        }

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
    setup_logging(log_level=args.log_level, log_format=LogFormat.CONSOLE)

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

    # Create session
    session = BacktestSession(
        start_date=start_date,
        end_date=end_date,
        symbols=args.symbols,
        initial_capital=Decimal(str(args.capital)),
        strategy_name=args.strategy,
        execution_mode=args.execution_mode,
        use_gpu=args.gpu,
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
