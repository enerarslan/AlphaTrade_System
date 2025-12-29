"""
Unit tests for the data module.

Tests data loading, preprocessing, feature store, and live feed.
"""

import asyncio
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_data_loader_initialization(self):
        """Test DataLoader initializes correctly."""
        from quant_trading_system.data.loader import DataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(tmpdir, cache_size=50, validate=True)

            assert loader.data_dir == Path(tmpdir)
            assert loader.validate is True
            assert loader._cache_size == 50

    def test_data_loader_supported_formats(self):
        """Test supported file formats."""
        from quant_trading_system.data.loader import DataLoader

        expected_formats = {".csv", ".parquet", ".hdf5", ".h5"}
        assert DataLoader.SUPPORTED_FORMATS == expected_formats

    def test_data_loader_get_available_symbols(self):
        """Test getting available symbols from directory."""
        from quant_trading_system.data.loader import DataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "AAPL.parquet").touch()
            (Path(tmpdir) / "GOOGL.csv").touch()
            (Path(tmpdir) / "MSFT.parquet").touch()

            loader = DataLoader(tmpdir)
            symbols = loader.get_available_symbols()

            assert "AAPL" in symbols
            assert "GOOGL" in symbols
            assert "MSFT" in symbols

    def test_data_loader_find_data_file(self):
        """Test finding data files for symbols."""
        from quant_trading_system.data.loader import DataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            test_file = Path(tmpdir) / "AAPL.parquet"
            test_file.touch()

            loader = DataLoader(tmpdir)
            found = loader._find_data_file("AAPL")

            assert found == test_file

    def test_data_loader_file_not_found(self):
        """Test handling of missing data files."""
        from quant_trading_system.core.exceptions import DataNotFoundError
        from quant_trading_system.data.loader import DataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(tmpdir)

            with pytest.raises(DataNotFoundError):
                loader.load_symbol("NONEXISTENT")

    def test_data_loader_standardize_columns(self):
        """Test column name standardization."""
        from quant_trading_system.data.loader import DataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(tmpdir)

            # Create a lazy frame with non-standard column names
            df = pl.DataFrame({
                "date": [datetime.utcnow()],
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.5],
                "Volume": [1000],
            })

            lf = loader._standardize_columns(df.lazy())
            result = lf.collect()

            assert "timestamp" in result.columns
            assert "open" in result.columns
            assert "high" in result.columns
            assert "low" in result.columns
            assert "close" in result.columns
            assert "volume" in result.columns


class TestParquetDataStore:
    """Tests for ParquetDataStore class."""

    def test_parquet_store_initialization(self):
        """Test ParquetDataStore initializes correctly."""
        from quant_trading_system.data.loader import ParquetDataStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ParquetDataStore(tmpdir, compression="zstd")

            assert store.base_dir == Path(tmpdir)
            assert store.compression == "zstd"

    def test_parquet_store_save_and_load(self):
        """Test saving and loading data."""
        from quant_trading_system.data.loader import ParquetDataStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ParquetDataStore(tmpdir)

            # Create test data
            df = pl.DataFrame({
                "timestamp": [datetime.utcnow() - timedelta(hours=i) for i in range(10)],
                "open": [100.0 + i for i in range(10)],
                "high": [101.0 + i for i in range(10)],
                "low": [99.0 + i for i in range(10)],
                "close": [100.5 + i for i in range(10)],
                "volume": [1000 + i * 100 for i in range(10)],
            })

            # Save
            path = store.save_symbol("AAPL", df)
            assert path.exists()

            # Load
            loaded = store.load_symbol("AAPL")
            assert len(loaded) == 10

    def test_parquet_store_list_symbols(self):
        """Test listing available symbols."""
        from quant_trading_system.data.loader import ParquetDataStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ParquetDataStore(tmpdir)

            # Create test data
            df = pl.DataFrame({
                "timestamp": [datetime.utcnow()],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000],
            })

            store.save_symbol("AAPL", df)
            store.save_symbol("GOOGL", df)

            symbols = store.list_symbols()
            assert "AAPL" in symbols
            assert "GOOGL" in symbols


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""

    def test_preprocessor_initialization(self):
        """Test DataPreprocessor initializes correctly."""
        from quant_trading_system.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor(
            outlier_threshold=3.0,
            max_interpolate_gap=5,
            forward_fill_limit=15,
        )

        assert preprocessor.outlier_threshold == 3.0
        assert preprocessor.max_interpolate_gap == 5
        assert preprocessor.forward_fill_limit == 15

    def test_preprocessor_clean_removes_negative_prices(self):
        """Test cleaning removes negative prices."""
        from quant_trading_system.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()

        df = pl.DataFrame({
            "timestamp": [datetime.utcnow() + timedelta(hours=i) for i in range(5)],
            "open": [100.0, 101.0, -50.0, 103.0, 104.0],
            "high": [101.0, 102.0, 51.0, 104.0, 105.0],
            "low": [99.0, 100.0, 49.0, 102.0, 103.0],
            "close": [100.5, 101.5, 50.5, 103.5, 104.5],
            "volume": [1000, 1100, 1200, 1300, 1400],
        })

        cleaned = preprocessor.clean_data(df)

        # Should have removed the row with negative price
        assert len(cleaned) == 4

    def test_preprocessor_clean_removes_invalid_ohlc(self):
        """Test cleaning removes invalid OHLC relationships."""
        from quant_trading_system.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()

        df = pl.DataFrame({
            "timestamp": [datetime.utcnow() + timedelta(hours=i) for i in range(5)],
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 100.0, 104.0, 105.0],  # 3rd row: high < open
            "low": [99.0, 100.0, 99.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [1000, 1100, 1200, 1300, 1400],
        })

        cleaned = preprocessor.clean_data(df)

        # Should have removed the row with invalid OHLC
        assert len(cleaned) == 4

    def test_preprocessor_handles_duplicates(self):
        """Test cleaning removes duplicate timestamps."""
        from quant_trading_system.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()

        ts = datetime.utcnow()
        df = pl.DataFrame({
            "timestamp": [ts, ts, ts + timedelta(hours=1)],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1100, 1200],
        })

        cleaned = preprocessor.clean_data(df)

        # Should have removed duplicates
        assert len(cleaned) == 2


class TestNormalizationMethod:
    """Tests for normalization methods."""

    def test_normalization_method_enum(self):
        """Test NormalizationMethod enum values."""
        from quant_trading_system.data.preprocessor import NormalizationMethod

        assert NormalizationMethod.ZSCORE == "zscore"
        assert NormalizationMethod.MINMAX == "minmax"
        assert NormalizationMethod.ROBUST == "robust"
        assert NormalizationMethod.LOG_RETURN == "log_return"

    def test_normalize_log_return(self):
        """Test log return normalization."""
        from quant_trading_system.data.preprocessor import DataPreprocessor, NormalizationMethod

        preprocessor = DataPreprocessor()

        df = pl.DataFrame({
            "timestamp": [datetime.utcnow() + timedelta(hours=i) for i in range(10)],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        })

        result = preprocessor.normalize(df, ["close"], NormalizationMethod.LOG_RETURN)

        assert "close_norm" in result.columns

    def test_normalize_percent_change(self):
        """Test percent change normalization."""
        from quant_trading_system.data.preprocessor import DataPreprocessor, NormalizationMethod

        preprocessor = DataPreprocessor()

        df = pl.DataFrame({
            "timestamp": [datetime.utcnow() + timedelta(hours=i) for i in range(10)],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        })

        result = preprocessor.normalize(df, ["close"], NormalizationMethod.PERCENT_CHANGE)

        assert "close_norm" in result.columns


class TestDataAligner:
    """Tests for DataAligner class."""

    def test_aligner_initialization(self):
        """Test DataAligner initializes correctly."""
        from quant_trading_system.data.preprocessor import DataAligner

        aligner = DataAligner(
            timeframe_minutes=15,
            include_premarket=True,
            include_postmarket=False,
        )

        assert aligner.timeframe_minutes == 15
        assert aligner.include_premarket is True
        assert aligner.include_postmarket is False

    def test_aligner_is_trading_time(self):
        """Test trading time detection."""
        from quant_trading_system.data.preprocessor import DataAligner

        aligner = DataAligner(timeframe_minutes=15)

        # Regular trading hours (10 AM)
        trading_time = datetime(2024, 1, 15, 10, 0)  # Monday
        assert aligner._is_trading_time(trading_time) is True

        # Weekend
        weekend_time = datetime(2024, 1, 13, 10, 0)  # Saturday
        assert aligner._is_trading_time(weekend_time) is False

        # Before market open
        early_time = datetime(2024, 1, 15, 8, 0)  # Monday
        assert aligner._is_trading_time(early_time) is False


class TestFeatureStore:
    """Tests for FeatureStore class."""

    def test_feature_store_initialization(self):
        """Test FeatureStore initializes correctly."""
        from quant_trading_system.data.feature_store import FeatureStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeatureStore(
                cache_dir=tmpdir,
                memory_cache_size=500,
                default_ttl=1800,
            )

            assert store.cache_dir == Path(tmpdir)
            assert store.default_ttl == 1800

    def test_feature_store_has_default_features(self):
        """Test default features are registered."""
        from quant_trading_system.data.feature_store import FeatureStore

        store = FeatureStore()

        features = store.list_features()

        assert "returns_1" in features
        assert "log_returns_1" in features
        assert "stat_volatility_20_std" in features
        assert "tech_rsi_14_raw" in features

    def test_feature_store_list_categories(self):
        """Test listing feature categories."""
        from quant_trading_system.data.feature_store import FeatureStore

        store = FeatureStore()

        categories = store.registry.list_categories()

        assert "stat" in categories
        assert "tech" in categories

    def test_feature_store_compute_features(self):
        """Test computing features."""
        from quant_trading_system.data.feature_store import FeatureStore

        store = FeatureStore()

        df = pl.DataFrame({
            "timestamp": [datetime.utcnow() + timedelta(minutes=15 * i) for i in range(100)],
            "open": [100.0 + i * 0.1 for i in range(100)],
            "high": [101.0 + i * 0.1 for i in range(100)],
            "low": [99.0 + i * 0.1 for i in range(100)],
            "close": [100.5 + i * 0.1 for i in range(100)],
            "volume": [1000 + i * 10 for i in range(100)],
        })

        result = store.compute_features(df, ["returns_1", "tech_sma_20_raw"])

        assert "returns_1" in result.columns
        assert "tech_sma_20_raw" in result.columns

    def test_feature_store_get_metadata(self):
        """Test getting feature metadata."""
        from quant_trading_system.data.feature_store import FeatureStore

        store = FeatureStore()

        metadata = store.get_feature_metadata("tech_rsi_14_raw")

        assert metadata is not None
        assert metadata["name"] == "tech_rsi_14_raw"
        assert metadata["category"] == "tech"

    def test_feature_store_nonexistent_metadata(self):
        """Test getting metadata for nonexistent feature."""
        from quant_trading_system.data.feature_store import FeatureStore

        store = FeatureStore()

        metadata = store.get_feature_metadata("nonexistent_feature")

        assert metadata is None


class TestFeatureRegistry:
    """Tests for FeatureRegistry class."""

    def test_registry_register_feature(self):
        """Test registering a feature."""
        from quant_trading_system.data.feature_store import FeatureDefinition, FeatureRegistry

        registry = FeatureRegistry()

        feature = FeatureDefinition(
            name="test_feature",
            category="test",
            compute_fn=lambda df: df["close"],
            description="Test feature",
        )

        registry.register(feature)

        assert registry.get("test_feature") is not None
        assert "test" in registry.list_categories()

    def test_registry_get_by_category(self):
        """Test getting features by category."""
        from quant_trading_system.data.feature_store import FeatureDefinition, FeatureRegistry

        registry = FeatureRegistry()

        for i in range(3):
            registry.register(FeatureDefinition(
                name=f"test_feature_{i}",
                category="test",
                compute_fn=lambda df: df["close"],
            ))

        features = registry.get_by_category("test")

        assert len(features) == 3


class TestMemoryCache:
    """Tests for MemoryCache class."""

    def test_memory_cache_set_get(self):
        """Test basic set and get."""
        from quant_trading_system.data.feature_store import MemoryCache

        cache = MemoryCache(max_size=100)

        cache.set("key1", "value1")
        result = cache.get("key1")

        assert result == "value1"

    def test_memory_cache_miss(self):
        """Test cache miss."""
        from quant_trading_system.data.feature_store import MemoryCache

        cache = MemoryCache(max_size=100)

        result = cache.get("nonexistent")

        assert result is None

    def test_memory_cache_invalidate(self):
        """Test cache invalidation."""
        from quant_trading_system.data.feature_store import MemoryCache

        cache = MemoryCache(max_size=100)

        cache.set("key1", "value1")
        cache.invalidate("key1")
        result = cache.get("key1")

        assert result is None

    def test_memory_cache_clear(self):
        """Test clearing cache."""
        from quant_trading_system.data.feature_store import MemoryCache

        cache = MemoryCache(max_size=100)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_memory_cache_eviction(self):
        """Test LRU eviction."""
        from quant_trading_system.data.feature_store import MemoryCache

        cache = MemoryCache(max_size=2)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"


class TestConnectionState:
    """Tests for ConnectionState enum."""

    def test_connection_state_values(self):
        """Test ConnectionState enum values."""
        from quant_trading_system.data.live_feed import ConnectionState

        assert ConnectionState.DISCONNECTED == "disconnected"
        assert ConnectionState.CONNECTING == "connecting"
        assert ConnectionState.CONNECTED == "connected"
        assert ConnectionState.RECONNECTING == "reconnecting"
        assert ConnectionState.ERROR == "error"


class TestMockLiveFeed:
    """Tests for MockLiveFeed class."""

    def test_mock_feed_initialization(self):
        """Test MockLiveFeed initializes correctly."""
        from quant_trading_system.data.live_feed import MockLiveFeed

        feed = MockLiveFeed(
            symbols=["AAPL", "GOOGL"],
            interval_seconds=2.0,
        )

        assert "AAPL" in feed.symbols
        assert "GOOGL" in feed.symbols
        assert feed.interval_seconds == 2.0

    def test_mock_feed_connect_disconnect(self):
        """Test connect and disconnect."""
        from quant_trading_system.data.live_feed import ConnectionState, MockLiveFeed

        async def run_test():
            feed = MockLiveFeed(symbols=["AAPL"])

            await feed.connect()
            assert feed.state == ConnectionState.CONNECTED

            await feed.disconnect()
            assert feed.state == ConnectionState.DISCONNECTED

        asyncio.run(run_test())

    def test_mock_feed_callback(self):
        """Test callback notification."""
        from quant_trading_system.core.data_types import OHLCVBar
        from quant_trading_system.data.live_feed import MockLiveFeed

        received_bars = []

        def callback(bar: OHLCVBar):
            received_bars.append(bar)

        async def run_test():
            feed = MockLiveFeed(
                symbols=["AAPL"],
                interval_seconds=0.1,
            )
            feed.add_callback(callback)

            await feed.connect()
            await asyncio.sleep(0.3)  # Wait for some bars
            await feed.disconnect()

            return len(received_bars)

        count = asyncio.run(run_test())
        assert count > 0


class TestBarAggregator:
    """Tests for BarAggregator class."""

    def test_aggregator_initialization(self):
        """Test BarAggregator initializes correctly."""
        from quant_trading_system.data.live_feed import BarAggregator

        aggregator = BarAggregator(timeframe_minutes=15)

        assert aggregator.timeframe_minutes == 15
        assert aggregator._pending_bars == {}

    def test_aggregator_add_trade(self):
        """Test adding trades to aggregator."""
        from quant_trading_system.data.live_feed import BarAggregator

        aggregator = BarAggregator(timeframe_minutes=15)

        ts = datetime(2024, 1, 15, 10, 5)  # Within a bar
        result = aggregator.add_trade("AAPL", 150.0, 100, ts)

        # First trade doesn't complete a bar
        assert result is None
        assert "AAPL" in aggregator._pending_bars

    def test_aggregator_bar_completion(self):
        """Test bar completion on timeframe boundary."""
        from quant_trading_system.data.live_feed import BarAggregator

        aggregator = BarAggregator(timeframe_minutes=15)

        # Add trade in first bar
        ts1 = datetime(2024, 1, 15, 10, 5)
        aggregator.add_trade("AAPL", 150.0, 100, ts1)

        # Add trade in next bar (should complete first bar)
        ts2 = datetime(2024, 1, 15, 10, 20)
        result = aggregator.add_trade("AAPL", 151.0, 100, ts2)

        assert result is not None
        assert result.symbol == "AAPL"

    def test_aggregator_flush(self):
        """Test flushing pending bars."""
        from quant_trading_system.data.live_feed import BarAggregator

        aggregator = BarAggregator(timeframe_minutes=15)

        ts = datetime(2024, 1, 15, 10, 5)
        aggregator.add_trade("AAPL", 150.0, 100, ts)
        aggregator.add_trade("GOOGL", 2500.0, 50, ts)

        bars = aggregator.flush()

        assert len(bars) == 2
        assert aggregator._pending_bars == {}


class TestCreateLiveFeed:
    """Tests for create_live_feed factory function."""

    def test_create_mock_feed(self):
        """Test creating mock feed."""
        from quant_trading_system.data.live_feed import MockLiveFeed, create_live_feed

        feed = create_live_feed(
            feed_type="mock",
            symbols=["AAPL"],
        )

        assert isinstance(feed, MockLiveFeed)

    def test_create_unknown_feed_raises(self):
        """Test creating unknown feed type raises error."""
        from quant_trading_system.data.live_feed import create_live_feed

        with pytest.raises(ValueError):
            create_live_feed(feed_type="unknown")


class TestStreamingFeatureUpdater:
    """Tests for StreamingFeatureUpdater class."""

    def test_updater_initialization(self):
        """Test StreamingFeatureUpdater initializes correctly."""
        from quant_trading_system.data.feature_store import FeatureStore, StreamingFeatureUpdater

        store = FeatureStore()
        updater = StreamingFeatureUpdater(store)

        assert updater.feature_store is store
        assert updater._buffers == {}

    def test_updater_clear_buffer(self):
        """Test clearing buffers."""
        from quant_trading_system.data.feature_store import FeatureStore, StreamingFeatureUpdater

        store = FeatureStore()
        updater = StreamingFeatureUpdater(store)

        # Add some data to buffer (mock)
        updater._buffers["AAPL"] = pl.DataFrame()
        updater._buffers["GOOGL"] = pl.DataFrame()

        updater.clear_buffer("AAPL")
        assert "AAPL" not in updater._buffers
        assert "GOOGL" in updater._buffers

        updater.clear_buffer()
        assert updater._buffers == {}
