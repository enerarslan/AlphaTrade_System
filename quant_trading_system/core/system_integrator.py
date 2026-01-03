"""
System Integrator - Wires all enhancement components together.

This module provides a unified interface for initializing, configuring,
and coordinating all the advanced enhancement components:

P1 (Critical):
- P1-A: VIX-Aware Adaptive Vol-Scaling
- P1-B: Sector Exposure/Rebalancing
- P1-C: Order-Book Imbalance Alpha
- P1-D: Transaction Cost Analysis (TCA)

P2 (High Priority):
- P2-A: Alternative Data Integration (News/Sentiment)
- P2-B: Purged Cross-Validation
- P2-C: IC-Based Ensemble Aggregation
- P2-D: RL Meta-Learning
- P2-E: Intraday Drawdown Alerts

P3 (Medium Priority):
- P3-A: Multi-Asset Correlation Monitoring
- P3-B: Adaptive Market Impact Models
- P3-C: Feature Computation Optimization

Author: AlphaTrade System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable

import pandas as pd
from pydantic import BaseModel, Field

from quant_trading_system.core.events import EventBus, get_event_bus

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Models
# =============================================================================


class SystemIntegratorConfig(BaseModel):
    """Configuration for the System Integrator."""

    # P1: Critical Enhancements
    enable_vix_scaling: bool = Field(default=True, description="P1-A: VIX-aware vol scaling")
    enable_sector_rebalancing: bool = Field(default=True, description="P1-B: Sector exposure monitoring")
    enable_order_book_alpha: bool = Field(default=True, description="P1-C: Order-book imbalance")
    enable_tca: bool = Field(default=True, description="P1-D: Transaction cost analysis")

    # P2: High Priority Enhancements
    enable_alt_data: bool = Field(default=True, description="P2-A: Alternative data")
    enable_purged_cv: bool = Field(default=True, description="P2-B: Purged cross-validation")
    enable_ic_ensemble: bool = Field(default=True, description="P2-C: IC-based ensemble")
    enable_rl_meta: bool = Field(default=True, description="P2-D: RL meta-learning")
    enable_drawdown_alerts: bool = Field(default=True, description="P2-E: Drawdown alerts")

    # P3: Medium Priority Enhancements
    enable_correlation_monitor: bool = Field(default=True, description="P3-A: Correlation monitoring")
    enable_market_impact: bool = Field(default=True, description="P3-B: Market impact models")
    enable_optimized_features: bool = Field(default=True, description="P3-C: Optimized features")

    # Integration settings
    update_interval_seconds: int = Field(default=5, description="Real-time update interval")
    enable_logging: bool = Field(default=True, description="Enable component logging")


@dataclass
class IntegratorState:
    """Current state of the system integrator."""

    initialized: bool = False
    running: bool = False
    start_time: datetime | None = None

    # Component states
    components_initialized: dict[str, bool] = field(default_factory=dict)
    components_running: dict[str, bool] = field(default_factory=dict)
    component_errors: dict[str, str] = field(default_factory=dict)

    # Current values from enhancements
    vix_level: float = 0.0
    vix_regime: str = "normal"
    current_drawdown_pct: float = 0.0
    correlation_regime: str = "normal"
    sector_exposures: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "initialized": self.initialized,
            "running": self.running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "components_initialized": self.components_initialized,
            "components_running": self.components_running,
            "component_errors": self.component_errors,
            "vix_level": self.vix_level,
            "vix_regime": self.vix_regime,
            "current_drawdown_pct": self.current_drawdown_pct,
            "correlation_regime": self.correlation_regime,
            "sector_exposures": self.sector_exposures,
        }


# =============================================================================
# System Integrator
# =============================================================================


class SystemIntegrator:
    """
    Central orchestrator for all enhancement components.

    Provides unified initialization, coordination, and monitoring
    of all enhancement modules for institutional-grade operation.
    """

    def __init__(
        self,
        config: SystemIntegratorConfig | None = None,
        event_bus: EventBus | None = None,
    ):
        """Initialize the System Integrator.

        Args:
            config: Integration configuration
            event_bus: Event bus for component communication
        """
        self.config = config or SystemIntegratorConfig()
        self.event_bus = event_bus or get_event_bus()

        self.state = IntegratorState()

        # Component references (initialized on demand)
        self._vix_scaler = None
        self._sector_rebalancer = None
        self._order_book_alpha = None
        self._tca_analyzer = None
        self._alt_data_aggregator = None
        self._purged_cv = None
        self._ic_ensemble = None
        self._rl_meta_agent = None
        self._drawdown_monitor = None
        self._correlation_monitor = None
        self._market_impact_model = None
        self._optimized_pipeline = None

        # Background tasks
        self._tasks: list[asyncio.Task] = []

    async def initialize(
        self,
        symbols: list[str] | None = None,
        initial_equity: Decimal | None = None,
    ) -> IntegratorState:
        """Initialize all enabled enhancement components.

        Args:
            symbols: Trading symbols
            initial_equity: Starting portfolio equity

        Returns:
            Current IntegratorState
        """
        logger.info("Initializing System Integrator with all enhancements")
        symbols = symbols or ["SPY", "QQQ", "IWM"]

        # P1: Critical Enhancements
        if self.config.enable_vix_scaling:
            await self._init_vix_scaler()

        if self.config.enable_sector_rebalancing:
            await self._init_sector_rebalancer(symbols)

        if self.config.enable_order_book_alpha:
            await self._init_order_book_alpha()

        if self.config.enable_tca:
            await self._init_tca()

        # P2: High Priority Enhancements
        if self.config.enable_alt_data:
            await self._init_alt_data()

        if self.config.enable_purged_cv:
            await self._init_purged_cv()

        if self.config.enable_ic_ensemble:
            await self._init_ic_ensemble()

        if self.config.enable_rl_meta:
            await self._init_rl_meta()

        if self.config.enable_drawdown_alerts and initial_equity:
            await self._init_drawdown_monitor(initial_equity)

        # P3: Medium Priority Enhancements
        if self.config.enable_correlation_monitor:
            await self._init_correlation_monitor(symbols)

        if self.config.enable_market_impact:
            await self._init_market_impact()

        if self.config.enable_optimized_features:
            await self._init_optimized_features()

        self.state.initialized = True
        self.state.start_time = datetime.now(timezone.utc)

        logger.info(f"System Integrator initialized. Components: {self.state.components_initialized}")
        return self.state

    # =========================================================================
    # P1: Critical Enhancement Initializers
    # =========================================================================

    async def _init_vix_scaler(self) -> None:
        """Initialize VIX-aware volatility scaling (P1-A)."""
        try:
            from quant_trading_system.alpha.vix_integration import (
                VIXIntegration,
                VIXConfig,
            )
            self._vix_scaler = VIXIntegration(VIXConfig())
            self.state.components_initialized["vix_scaler"] = True
            logger.info("P1-A: VIX-Aware Vol Scaling initialized")
        except Exception as e:
            self.state.component_errors["vix_scaler"] = str(e)
            logger.warning(f"P1-A VIX Scaler init failed: {e}")

    async def _init_sector_rebalancer(self, symbols: list[str]) -> None:
        """Initialize sector exposure monitoring (P1-B)."""
        try:
            from quant_trading_system.risk.sector_rebalancer import (
                SectorRebalancer,
                SectorRebalanceConfig,
                create_sector_rebalancer,
            )
            self._sector_rebalancer = create_sector_rebalancer(
                SectorRebalanceConfig(),
                symbols,
            )
            self.state.components_initialized["sector_rebalancer"] = True
            logger.info("P1-B: Sector Exposure/Rebalancing initialized")
        except Exception as e:
            self.state.component_errors["sector_rebalancer"] = str(e)
            logger.warning(f"P1-B Sector Rebalancer init failed: {e}")

    async def _init_order_book_alpha(self) -> None:
        """Initialize order book imbalance alpha (P1-C)."""
        try:
            from quant_trading_system.alpha.order_book_imbalance import (
                OrderBookImbalanceAlpha,
                OrderBookConfig,
            )
            self._order_book_alpha = OrderBookImbalanceAlpha(OrderBookConfig())
            self.state.components_initialized["order_book_alpha"] = True
            logger.info("P1-C: Order-Book Imbalance Alpha initialized")
        except Exception as e:
            self.state.component_errors["order_book_alpha"] = str(e)
            logger.warning(f"P1-C Order Book Alpha init failed: {e}")

    async def _init_tca(self) -> None:
        """Initialize transaction cost analysis (P1-D)."""
        try:
            from quant_trading_system.execution.tca import (
                TransactionCostAnalyzer,
                TCAConfig,
            )
            self._tca_analyzer = TransactionCostAnalyzer(TCAConfig())
            self.state.components_initialized["tca"] = True
            logger.info("P1-D: Transaction Cost Analysis initialized")
        except Exception as e:
            self.state.component_errors["tca"] = str(e)
            logger.warning(f"P1-D TCA init failed: {e}")

    # =========================================================================
    # P2: High Priority Enhancement Initializers
    # =========================================================================

    async def _init_alt_data(self) -> None:
        """Initialize alternative data integration (P2-A)."""
        try:
            from quant_trading_system.data.alternative_data import (
                AltDataAggregator,
                AltDataConfig,
            )
            self._alt_data_aggregator = AltDataAggregator(AltDataConfig())
            self.state.components_initialized["alt_data"] = True
            logger.info("P2-A: Alternative Data Integration initialized")
        except Exception as e:
            self.state.component_errors["alt_data"] = str(e)
            logger.warning(f"P2-A Alt Data init failed: {e}")

    async def _init_purged_cv(self) -> None:
        """Initialize purged cross-validation (P2-B)."""
        try:
            from quant_trading_system.models.purged_cv import (
                PurgedKFold,
                create_purged_cv,
            )
            self._purged_cv = create_purged_cv(
                n_splits=5,
                purge_gap=5,
                embargo_pct=0.01,
            )
            self.state.components_initialized["purged_cv"] = True
            logger.info("P2-B: Purged Cross-Validation initialized")
        except Exception as e:
            self.state.component_errors["purged_cv"] = str(e)
            logger.warning(f"P2-B Purged CV init failed: {e}")

    async def _init_ic_ensemble(self) -> None:
        """Initialize IC-based ensemble aggregation (P2-C)."""
        try:
            from quant_trading_system.models.ensemble import (
                ICBasedEnsemble,
            )
            self._ic_ensemble = ICBasedEnsemble(
                name="ic_ensemble",
                ic_lookback=60,
                min_ic_threshold=0.02,
            )
            self.state.components_initialized["ic_ensemble"] = True
            logger.info("P2-C: IC-Based Ensemble Aggregation initialized")
        except Exception as e:
            self.state.component_errors["ic_ensemble"] = str(e)
            logger.warning(f"P2-C IC Ensemble init failed: {e}")

    async def _init_rl_meta(self) -> None:
        """Initialize RL meta-learning (P2-D)."""
        try:
            from quant_trading_system.models.rl_meta_learning import (
                MetaLearningAgent,
                RLMetaConfig,
                create_meta_learning_agent,
            )
            config = RLMetaConfig(
                state_dim=64,
                action_dim=3,  # buy, hold, sell
                hidden_dim=128,
            )
            self._rl_meta_agent = create_meta_learning_agent(config)
            self.state.components_initialized["rl_meta"] = True
            logger.info("P2-D: RL Meta-Learning initialized")
        except Exception as e:
            self.state.component_errors["rl_meta"] = str(e)
            logger.warning(f"P2-D RL Meta init failed: {e}")

    async def _init_drawdown_monitor(self, initial_equity: Decimal) -> None:
        """Initialize intraday drawdown monitoring (P2-E)."""
        try:
            from quant_trading_system.risk.drawdown_monitor import (
                IntradayDrawdownMonitor,
                DrawdownMonitorConfig,
                create_drawdown_monitor,
            )
            self._drawdown_monitor = create_drawdown_monitor(
                DrawdownMonitorConfig(
                    warning_threshold_pct=3.0,
                    critical_threshold_pct=5.0,
                    emergency_threshold_pct=8.0,
                    kill_switch_threshold_pct=10.0,
                )
            )
            self._drawdown_monitor.start_session(initial_equity)
            self.state.components_initialized["drawdown_monitor"] = True
            logger.info("P2-E: Intraday Drawdown Alerts initialized")
        except Exception as e:
            self.state.component_errors["drawdown_monitor"] = str(e)
            logger.warning(f"P2-E Drawdown Monitor init failed: {e}")

    # =========================================================================
    # P3: Medium Priority Enhancement Initializers
    # =========================================================================

    async def _init_correlation_monitor(self, symbols: list[str]) -> None:
        """Initialize correlation monitoring (P3-A)."""
        try:
            from quant_trading_system.risk.correlation_monitor import (
                CorrelationMonitor,
                CorrelationMonitorConfig,
                create_correlation_monitor,
            )
            self._correlation_monitor = create_correlation_monitor(
                CorrelationMonitorConfig(
                    high_correlation_threshold=0.8,
                    correlation_breakdown_threshold=0.3,
                ),
                symbols,
            )
            self.state.components_initialized["correlation_monitor"] = True
            logger.info("P3-A: Multi-Asset Correlation Monitoring initialized")
        except Exception as e:
            self.state.component_errors["correlation_monitor"] = str(e)
            logger.warning(f"P3-A Correlation Monitor init failed: {e}")

    async def _init_market_impact(self) -> None:
        """Initialize market impact models (P3-B)."""
        try:
            from quant_trading_system.execution.market_impact import (
                AdaptiveMarketImpactModel,
                MarketImpactConfig,
            )
            self._market_impact_model = AdaptiveMarketImpactModel(
                MarketImpactConfig()
            )
            self.state.components_initialized["market_impact"] = True
            logger.info("P3-B: Adaptive Market Impact Models initialized")
        except Exception as e:
            self.state.component_errors["market_impact"] = str(e)
            logger.warning(f"P3-B Market Impact init failed: {e}")

    async def _init_optimized_features(self) -> None:
        """Initialize optimized feature pipeline (P3-C)."""
        try:
            from quant_trading_system.features.optimized_pipeline import (
                OptimizedFeaturePipeline,
                OptimizedPipelineConfig,
            )
            self._optimized_pipeline = OptimizedFeaturePipeline(
                OptimizedPipelineConfig(
                    enable_numba=True,
                    enable_cache=True,
                    cache_type="hybrid",
                )
            )
            self.state.components_initialized["optimized_features"] = True
            logger.info("P3-C: Feature Computation Optimization initialized")
        except Exception as e:
            self.state.component_errors["optimized_features"] = str(e)
            logger.warning(f"P3-C Optimized Features init failed: {e}")

    # =========================================================================
    # Runtime Operations
    # =========================================================================

    async def start(self, equity_provider: Callable[[], Decimal] | None = None) -> None:
        """Start all enhancement components.

        Args:
            equity_provider: Callback to get current portfolio equity
        """
        if not self.state.initialized:
            raise RuntimeError("System Integrator not initialized. Call initialize() first.")

        logger.info("Starting System Integrator real-time operations")
        self.state.running = True

        # Start drawdown monitoring
        if self._drawdown_monitor and equity_provider:
            self._tasks.append(
                asyncio.create_task(self._drawdown_monitor.start_monitoring(equity_provider))
            )
            self.state.components_running["drawdown_monitor"] = True

        # Start periodic update task
        self._tasks.append(
            asyncio.create_task(self._periodic_update_loop())
        )

        logger.info("System Integrator started")

    async def stop(self) -> None:
        """Stop all enhancement components."""
        logger.info("Stopping System Integrator")
        self.state.running = False

        # Stop drawdown monitoring
        if self._drawdown_monitor:
            await self._drawdown_monitor.stop_monitoring()

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()
        logger.info("System Integrator stopped")

    async def _periodic_update_loop(self) -> None:
        """Periodic update loop for enhancement components."""
        while self.state.running:
            try:
                await self._update_state()
            except Exception as e:
                logger.error(f"Error in periodic update: {e}")

            await asyncio.sleep(self.config.update_interval_seconds)

    async def _update_state(self) -> None:
        """Update current state from all components."""
        # Update VIX state
        if self._vix_scaler:
            try:
                vix_state = self._vix_scaler.get_current_state()
                self.state.vix_level = vix_state.get("vix_level", 0.0)
                self.state.vix_regime = vix_state.get("regime", "normal")
            except Exception as e:
                logger.debug(f"VIX state update failed: {e}")

        # Update drawdown state
        if self._drawdown_monitor:
            try:
                dd_state = self._drawdown_monitor.get_current_state()
                if dd_state:
                    self.state.current_drawdown_pct = dd_state.peak_to_trough_pct
            except Exception as e:
                logger.debug(f"Drawdown state update failed: {e}")

        # Update correlation state
        if self._correlation_monitor:
            try:
                corr_state = self._correlation_monitor.get_current_regime()
                self.state.correlation_regime = corr_state.value if corr_state else "normal"
            except Exception as e:
                logger.debug(f"Correlation state update failed: {e}")

    # =========================================================================
    # Component Accessors
    # =========================================================================

    def get_vix_scaling_factor(self, base_volatility: float) -> float:
        """Get VIX-adjusted scaling factor for position sizing.

        Args:
            base_volatility: Base volatility estimate

        Returns:
            Adjusted scaling factor
        """
        if not self._vix_scaler:
            return 1.0

        try:
            return self._vix_scaler.get_position_scale_factor(base_volatility)
        except Exception as e:
            logger.warning(f"VIX scaling failed: {e}")
            return 1.0

    def get_sector_rebalance_actions(
        self,
        current_positions: dict[str, float],
        prices: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Get sector rebalancing recommendations.

        Args:
            current_positions: Current position sizes by symbol
            prices: Current prices by symbol

        Returns:
            List of rebalance actions
        """
        if not self._sector_rebalancer:
            return []

        try:
            return self._sector_rebalancer.get_rebalance_actions(
                current_positions, prices
            )
        except Exception as e:
            logger.warning(f"Sector rebalancing failed: {e}")
            return []

    def compute_market_impact(
        self,
        symbol: str,
        quantity: int,
        side: str,
        urgency: float = 0.5,
    ) -> float:
        """Compute expected market impact for a trade.

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            side: 'buy' or 'sell'
            urgency: Trade urgency (0-1)

        Returns:
            Expected market impact in basis points
        """
        if not self._market_impact_model:
            return 0.0

        try:
            return self._market_impact_model.estimate_impact(
                symbol, quantity, side, urgency
            )
        except Exception as e:
            logger.warning(f"Market impact estimation failed: {e}")
            return 0.0

    def update_equity(self, equity: Decimal) -> None:
        """Update current equity for drawdown monitoring.

        Args:
            equity: Current portfolio equity
        """
        if self._drawdown_monitor:
            try:
                self._drawdown_monitor.update_equity(equity)
            except Exception as e:
                logger.warning(f"Equity update failed: {e}")

    def compute_features(
        self,
        data: pd.DataFrame,
        symbol: str,
        feature_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compute features using optimized pipeline.

        Args:
            data: OHLCV data
            symbol: Symbol for caching
            feature_names: Optional feature names to compute

        Returns:
            DataFrame with computed features
        """
        if not self._optimized_pipeline:
            return pd.DataFrame()

        try:
            return self._optimized_pipeline.compute_features(
                data, symbol, feature_names
            )
        except Exception as e:
            logger.warning(f"Feature computation failed: {e}")
            return pd.DataFrame()

    def get_ic_ensemble_weights(
        self,
        model_predictions: dict[str, float],
        model_ics: dict[str, float],
    ) -> dict[str, float]:
        """Get IC-weighted ensemble weights for models.

        Args:
            model_predictions: Predictions from each model
            model_ics: Information coefficients for each model

        Returns:
            Weights for each model
        """
        if not self._ic_ensemble:
            # Equal weighting fallback
            n = len(model_predictions)
            return {k: 1.0 / n for k in model_predictions}

        try:
            return self._ic_ensemble.get_weights(model_ics)
        except Exception as e:
            logger.warning(f"IC ensemble weighting failed: {e}")
            n = len(model_predictions)
            return {k: 1.0 / n for k in model_predictions}

    # =========================================================================
    # Status and Monitoring
    # =========================================================================

    def get_state(self) -> IntegratorState:
        """Get current integrator state.

        Returns:
            Current IntegratorState
        """
        return self.state

    def get_component_status(self) -> dict[str, dict[str, Any]]:
        """Get detailed status of all components.

        Returns:
            Dictionary with component statuses
        """
        status = {}

        components = [
            ("P1-A: VIX Scaler", "vix_scaler", self._vix_scaler),
            ("P1-B: Sector Rebalancer", "sector_rebalancer", self._sector_rebalancer),
            ("P1-C: Order Book Alpha", "order_book_alpha", self._order_book_alpha),
            ("P1-D: TCA", "tca", self._tca_analyzer),
            ("P2-A: Alt Data", "alt_data", self._alt_data_aggregator),
            ("P2-B: Purged CV", "purged_cv", self._purged_cv),
            ("P2-C: IC Ensemble", "ic_ensemble", self._ic_ensemble),
            ("P2-D: RL Meta", "rl_meta", self._rl_meta_agent),
            ("P2-E: Drawdown Monitor", "drawdown_monitor", self._drawdown_monitor),
            ("P3-A: Correlation Monitor", "correlation_monitor", self._correlation_monitor),
            ("P3-B: Market Impact", "market_impact", self._market_impact_model),
            ("P3-C: Optimized Features", "optimized_features", self._optimized_pipeline),
        ]

        for display_name, key, component in components:
            status[display_name] = {
                "key": key,
                "initialized": self.state.components_initialized.get(key, False),
                "running": self.state.components_running.get(key, False),
                "error": self.state.component_errors.get(key),
                "available": component is not None,
            }

        return status

    def get_summary(self) -> str:
        """Get a human-readable summary of integrator status.

        Returns:
            Formatted summary string
        """
        lines = [
            "=" * 70,
            "SYSTEM INTEGRATOR STATUS",
            "=" * 70,
            f"Initialized: {self.state.initialized}",
            f"Running: {self.state.running}",
            f"Start Time: {self.state.start_time.isoformat() if self.state.start_time else 'N/A'}",
            "",
            "Component Status:",
            "-" * 70,
        ]

        for name, status in self.get_component_status().items():
            status_str = "OK" if status["initialized"] else "DISABLED"
            if status["error"]:
                status_str = f"ERROR: {status['error'][:30]}"
            lines.append(f"  {name:35} {status_str}")

        lines.extend([
            "-" * 70,
            "Current State:",
            f"  VIX Level: {self.state.vix_level:.2f} ({self.state.vix_regime})",
            f"  Drawdown: {self.state.current_drawdown_pct:.2f}%",
            f"  Correlation Regime: {self.state.correlation_regime}",
            "=" * 70,
        ])

        return "\n".join(lines)


# =============================================================================
# Factory Function
# =============================================================================


def create_system_integrator(
    config: SystemIntegratorConfig | None = None,
) -> SystemIntegrator:
    """Factory function to create a System Integrator.

    Args:
        config: Integration configuration

    Returns:
        Configured SystemIntegrator instance
    """
    return SystemIntegrator(config=config)


# =============================================================================
# Singleton Instance
# =============================================================================

_integrator_instance: SystemIntegrator | None = None


def get_system_integrator() -> SystemIntegrator:
    """Get the singleton System Integrator instance.

    Returns:
        The global SystemIntegrator instance
    """
    global _integrator_instance
    if _integrator_instance is None:
        _integrator_instance = SystemIntegrator()
    return _integrator_instance
