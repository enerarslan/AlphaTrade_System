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
