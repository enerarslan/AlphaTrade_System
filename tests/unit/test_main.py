"""
Unit tests for main.py
"""

import asyncio
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import after patching to avoid issues
with patch.dict('sys.modules', {'uvicorn': MagicMock()}):
    from main import (
        TradingSystemApp,
        parse_args,
    )


class TestParseArgs:
    """Tests for argument parsing."""

    def test_parse_trade_command(self):
        """Test parsing trade command."""
        with patch.object(sys, 'argv', ['main.py', 'trade', '--mode', 'paper']):
            args = parse_args()
            assert args.command == "trade"
            assert args.mode == "paper"

    def test_parse_trade_live_mode(self):
        """Test parsing trade command with live mode."""
        with patch.object(sys, 'argv', ['main.py', 'trade', '--mode', 'live']):
            args = parse_args()
            assert args.mode == "live"

    def test_parse_trade_dry_run(self):
        """Test parsing trade command with dry-run."""
        with patch.object(sys, 'argv', ['main.py', 'trade', '--dry-run']):
            args = parse_args()
            assert args.dry_run is True

    def test_parse_backtest_command(self):
        """Test parsing backtest command."""
        with patch.object(sys, 'argv', [
            'main.py', 'backtest',
            '--start-date', '2024-01-01',
            '--end-date', '2024-12-31',
        ]):
            args = parse_args()
            assert args.command == "backtest"
            assert args.start_date == "2024-01-01"
            assert args.end_date == "2024-12-31"

    def test_parse_backtest_with_capital(self):
        """Test parsing backtest command with initial capital."""
        with patch.object(sys, 'argv', [
            'main.py', 'backtest',
            '--start-date', '2024-01-01',
            '--end-date', '2024-12-31',
            '--initial-capital', '200000',
        ]):
            args = parse_args()
            assert args.initial_capital == 200000.0

    def test_parse_dashboard_command(self):
        """Test parsing dashboard command."""
        with patch.object(sys, 'argv', ['main.py', 'dashboard']):
            args = parse_args()
            assert args.command == "dashboard"
            assert args.host == "0.0.0.0"
            assert args.port == 8000

    def test_parse_dashboard_with_options(self):
        """Test parsing dashboard command with options."""
        with patch.object(sys, 'argv', [
            'main.py', 'dashboard',
            '--host', 'localhost',
            '--port', '9000',
            '--reload',
        ]):
            args = parse_args()
            assert args.host == "localhost"
            assert args.port == 9000
            assert args.reload is True

    def test_parse_train_command_with_new_flags(self):
        """Train parser should accept institutional fail-fast flags."""
        with patch.object(sys, "argv", [
            "main.py",
            "train",
            "--model",
            "elastic_net",
            "--name",
            "elastic_v1",
            "--no-redis-cache",
            "--min-accuracy",
            "0.55",
        ]):
            args = parse_args()
            assert args.command == "train"
            assert args.model == "elastic_net"
            assert args.name == "elastic_v1"
            assert args.no_redis_cache is True
            assert args.min_accuracy == pytest.approx(0.55)

    def test_parse_train_replay_and_nested_flags(self):
        with patch.object(
            sys,
            "argv",
            [
                "main.py",
                "train",
                "--replay-manifest",
                "models/replays/example.replay_manifest.json",
                "--nested-outer-splits",
                "5",
                "--nested-inner-splits",
                "3",
                "--seed",
                "7",
            ],
        ):
            args = parse_args()
            assert args.command == "train"
            assert str(args.replay_manifest).endswith("example.replay_manifest.json")
            assert args.nested_outer_splits == 5
            assert args.nested_inner_splits == 3
            assert args.seed == 7

    def test_parse_train_advanced_risk_and_execution_flags(self):
        with patch.object(
            sys,
            "argv",
            [
                "main.py",
                "train",
                "--objective-weight-cvar",
                "0.6",
                "--objective-weight-skew",
                "0.2",
                "--objective-weight-tail-risk",
                "0.8",
                "--objective-weight-symbol-concentration",
                "0.35",
                "--holdout-pct",
                "0.2",
                "--min-holdout-sharpe",
                "0.1",
                "--min-holdout-regime-sharpe",
                "-0.04",
                "--max-holdout-drawdown",
                "0.25",
                "--max-regime-shift",
                "0.3",
                "--max-symbol-concentration-hhi",
                "0.57",
                "--label-edge-cost-buffer-bps",
                "3.0",
                "--disable-symbol-quality-filter",
                "--symbol-quality-min-rows",
                "1600",
                "--symbol-quality-min-symbols",
                "11",
                "--symbol-quality-max-missing-ratio",
                "0.09",
                "--symbol-quality-max-extreme-move-ratio",
                "0.06",
                "--symbol-quality-max-corporate-action-ratio",
                "0.015",
                "--symbol-quality-min-median-dollar-volume",
                "2200000",
                "--disable-dynamic-no-trade-band",
                "--allow-partial-feature-fallback",
                "--execution-vol-target-daily",
                "0.01",
                "--execution-turnover-cap",
                "0.7",
                "--execution-cooldown-bars",
                "4",
                "--execution-max-symbol-entry-share",
                "0.63",
                "--primary-horizon-sweep",
                "1",
                "5",
                "20",
                "--meta-label-min-confidence",
                "0.61",
                "--disable-meta-dynamic-threshold",
            ],
        ):
            args = parse_args()
            assert args.objective_weight_cvar == pytest.approx(0.6)
            assert args.objective_weight_skew == pytest.approx(0.2)
            assert args.objective_weight_tail_risk == pytest.approx(0.8)
            assert args.objective_weight_symbol_concentration == pytest.approx(0.35)
            assert args.holdout_pct == pytest.approx(0.2)
            assert args.min_holdout_sharpe == pytest.approx(0.1)
            assert args.min_holdout_regime_sharpe == pytest.approx(-0.04)
            assert args.max_holdout_drawdown == pytest.approx(0.25)
            assert args.max_regime_shift == pytest.approx(0.3)
            assert args.max_symbol_concentration_hhi == pytest.approx(0.57)
            assert args.disable_symbol_quality_filter is True
            assert args.symbol_quality_min_rows == 1600
            assert args.symbol_quality_min_symbols == 11
            assert args.symbol_quality_max_missing_ratio == pytest.approx(0.09)
            assert args.symbol_quality_max_extreme_move_ratio == pytest.approx(0.06)
            assert args.symbol_quality_max_corporate_action_ratio == pytest.approx(0.015)
            assert args.symbol_quality_min_median_dollar_volume == pytest.approx(2_200_000.0)
            assert args.label_edge_cost_buffer_bps == pytest.approx(3.0)
            assert args.disable_dynamic_no_trade_band is True
            assert args.allow_partial_feature_fallback is True
            assert args.execution_vol_target_daily == pytest.approx(0.01)
            assert args.execution_turnover_cap == pytest.approx(0.7)
            assert args.execution_cooldown_bars == 4
            assert args.execution_max_symbol_entry_share == pytest.approx(0.63)
            assert args.primary_horizon_sweep == [1, 5, 20]
            assert args.meta_label_min_confidence == pytest.approx(0.61)
            assert args.disable_meta_dynamic_threshold is True

    def test_parse_data_download_alias_and_sync_flags(self):
        with patch.object(
            sys,
            "argv",
            [
                "main.py",
                "data",
                "download",
                "--symbols",
                "AAPL",
                "--sync-db",
                "--incremental",
                "--batch-size",
                "2500",
            ],
        ):
            args = parse_args()
            assert args.command == "data"
            assert args.data_command == "download"
            assert args.sync_db is True
            assert args.incremental is True
            assert args.batch_size == 2500

    def test_parse_log_level(self):
        """Test parsing log level option."""
        with patch.object(sys, 'argv', [
            'main.py', '--log-level', 'DEBUG', 'trade',
        ]):
            args = parse_args()
            assert args.log_level == "DEBUG"

    def test_parse_log_format(self):
        """Test parsing log format option."""
        with patch.object(sys, 'argv', [
            'main.py', '--log-format', 'text', 'trade',
        ]):
            args = parse_args()
            assert args.log_format == "text"


class TestTradingSystemApp:
    """Tests for TradingSystemApp class."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.app_version = "1.0.0"
        settings.environment = "test"
        return settings

    def test_create_app(self, mock_settings):
        """Test creating trading system app."""
        app = TradingSystemApp(mock_settings)
        assert app.settings == mock_settings
        assert app._running is False

    def test_start_and_stop(self, mock_settings):
        """Test starting and stopping the app."""

        async def run_test():
            app = TradingSystemApp(mock_settings)

            # Start the app in background
            start_task = asyncio.create_task(app.start(mode="paper"))

            # Wait a bit then stop
            await asyncio.sleep(0.1)
            await app.stop()

            # Cancel the start task
            start_task.cancel()
            try:
                await start_task
            except asyncio.CancelledError:
                pass

            return app._running

        running = asyncio.run(run_test())
        assert running is False

    def test_update_system_metrics_with_psutil(self, mock_settings):
        """Test updating system metrics when psutil is available."""
        app = TradingSystemApp(mock_settings)

        # Mock psutil
        with patch.dict('sys.modules', {'psutil': MagicMock()}):
            import sys
            mock_psutil = sys.modules['psutil']
            mock_process = MagicMock()
            mock_process.memory_info.return_value.rss = 1024 * 1024 * 100
            mock_process.cpu_percent.return_value = 25.0
            mock_psutil.Process.return_value = mock_process

            # This should not raise
            app._update_system_metrics()

    def test_update_system_metrics_without_psutil(self, mock_settings):
        """Test updating system metrics when psutil is not available."""
        app = TradingSystemApp(mock_settings)

        # Mock import error
        with patch.dict('sys.modules', {'psutil': None}):
            # This should not raise even without psutil
            app._update_system_metrics()
