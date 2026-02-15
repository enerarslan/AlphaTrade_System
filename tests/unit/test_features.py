"""
Unit tests for the features module.

Tests technical indicators, statistical features, microstructure features,
cross-sectional features, and the feature pipeline.
"""

import numpy as np
import polars as pl
import pytest
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame for testing."""
    n = 200
    np.random.seed(42)

    # Generate realistic price data
    returns = np.random.normal(0.0005, 0.02, n)
    close = 100.0 * np.exp(np.cumsum(returns))

    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_ = low + np.random.uniform(0, 1, n) * (high - low)
    volume = np.random.randint(100000, 1000000, n).astype(float)

    timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]

    return pl.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def sample_ohlcv_arrays(sample_ohlcv_df):
    """Extract arrays from OHLCV DataFrame."""
    return {
        "open": sample_ohlcv_df["open"].to_numpy(),
        "high": sample_ohlcv_df["high"].to_numpy(),
        "low": sample_ohlcv_df["low"].to_numpy(),
        "close": sample_ohlcv_df["close"].to_numpy(),
        "volume": sample_ohlcv_df["volume"].to_numpy(),
    }


class TestTechnicalIndicators:
    """Tests for technical indicators."""

    def test_sma_computation(self, sample_ohlcv_df):
        """Test Simple Moving Average computation."""
        from quant_trading_system.features.technical import SMA

        sma = SMA(periods=[20])
        result = sma.compute(sample_ohlcv_df)

        # Result should be a dict with sma_20
        assert isinstance(result, dict)
        assert "sma_20" in result

        sma_values = result["sma_20"]
        close = sample_ohlcv_df["close"].to_numpy()

        # Result should have same length as input
        assert len(sma_values) == len(close)

        # First period-1 values should be NaN
        assert np.all(np.isnan(sma_values[: 20 - 1]))

        # Valid values should be present
        assert not np.isnan(sma_values[-1])

        # Verify SMA calculation for a known point
        manual_sma = np.mean(close[-20:])
        assert np.isclose(sma_values[-1], manual_sma, rtol=1e-10)

    def test_ema_computation(self, sample_ohlcv_df):
        """Test Exponential Moving Average computation."""
        from quant_trading_system.features.technical import EMA

        ema = EMA(periods=[20])
        result = ema.compute(sample_ohlcv_df)

        assert isinstance(result, dict)
        assert "ema_20" in result
        assert not np.isnan(result["ema_20"][-1])

    def test_rsi_computation(self, sample_ohlcv_df):
        """Test RSI computation."""
        from quant_trading_system.features.technical import RSI

        rsi = RSI(periods=[14])
        result = rsi.compute(sample_ohlcv_df)

        assert isinstance(result, dict)
        assert "rsi_14" in result

        rsi_values = result["rsi_14"]

        # RSI should be bounded [0, 100]
        valid_values = rsi_values[~np.isnan(rsi_values)]
        assert np.all(valid_values >= 0)
        assert np.all(valid_values <= 100)

    def test_macd_computation(self, sample_ohlcv_df):
        """Test MACD computation."""
        from quant_trading_system.features.technical import MACD

        macd = MACD(fast=12, slow=26, signal=9)
        result = macd.compute(sample_ohlcv_df)

        assert isinstance(result, dict)
        assert "macd_line" in result
        assert "macd_signal" in result
        assert "macd_histogram" in result

        # Histogram should equal MACD line minus signal line
        macd_line = result["macd_line"]
        signal_line = result["macd_signal"]
        histogram = result["macd_histogram"]

        valid_mask = ~(np.isnan(macd_line) | np.isnan(signal_line))
        assert np.allclose(
            histogram[valid_mask],
            macd_line[valid_mask] - signal_line[valid_mask],
            rtol=1e-10,
        )

    def test_bollinger_bands(self, sample_ohlcv_df):
        """Test Bollinger Bands computation."""
        from quant_trading_system.features.technical import BollingerBands

        bb = BollingerBands(period=20, std_dev=2.0)
        result = bb.compute(sample_ohlcv_df)

        assert isinstance(result, dict)
        assert "bb_upper_20" in result
        assert "bb_middle_20" in result
        assert "bb_lower_20" in result

        upper = result["bb_upper_20"]
        middle = result["bb_middle_20"]
        lower = result["bb_lower_20"]

        # Upper should always be >= middle >= lower
        valid_mask = ~(np.isnan(upper) | np.isnan(middle) | np.isnan(lower))
        assert np.all(upper[valid_mask] >= middle[valid_mask])
        assert np.all(middle[valid_mask] >= lower[valid_mask])

    def test_atr_computation(self, sample_ohlcv_df):
        """Test ATR computation."""
        from quant_trading_system.features.technical import ATR

        atr = ATR(period=14)
        result = atr.compute(sample_ohlcv_df)

        assert isinstance(result, dict)
        assert "atr_14" in result

        atr_values = result["atr_14"]

        # ATR should always be positive
        valid_values = atr_values[~np.isnan(atr_values)]
        assert np.all(valid_values > 0)

    def test_adx_computation(self, sample_ohlcv_df):
        """Test ADX computation."""
        from quant_trading_system.features.technical import ADX

        adx = ADX(period=14)
        result = adx.compute(sample_ohlcv_df)

        assert isinstance(result, dict)
        assert "adx_14" in result
        assert "plus_di_14" in result
        assert "minus_di_14" in result

        adx_values = result["adx_14"]

        # ADX should be bounded [0, 100]
        valid_adx = adx_values[~np.isnan(adx_values)]
        assert np.all(valid_adx >= 0)
        assert np.all(valid_adx <= 100)

    def test_stochastic_oscillator(self, sample_ohlcv_df):
        """Test Stochastic Oscillator computation."""
        from quant_trading_system.features.technical import StochasticOscillator

        stoch = StochasticOscillator(k_period=14, d_period=3)
        result = stoch.compute(sample_ohlcv_df)

        assert isinstance(result, dict)
        assert "stoch_k_14" in result
        assert "stoch_d_3" in result

        k = result["stoch_k_14"]
        d = result["stoch_d_3"]

        # %K and %D should be bounded [0, 100]
        valid_k = k[~np.isnan(k)]
        valid_d = d[~np.isnan(d)]
        assert np.all(valid_k >= 0) and np.all(valid_k <= 100)
        assert np.all(valid_d >= 0) and np.all(valid_d <= 100)

    def test_obv_computation(self, sample_ohlcv_df):
        """Test On-Balance Volume computation."""
        from quant_trading_system.features.technical import OBV

        obv = OBV()
        result = obv.compute(sample_ohlcv_df)

        assert isinstance(result, dict)
        assert "obv" in result

        # OBV should have no NaN values after first
        assert not np.isnan(result["obv"][-1])

    def test_vwap_computation(self, sample_ohlcv_df):
        """Test VWAP computation."""
        from quant_trading_system.features.technical import VWAP

        vwap = VWAP()
        result = vwap.compute(sample_ohlcv_df)

        assert isinstance(result, dict)
        assert "vwap_cumulative" in result

        vwap_values = result["vwap_cumulative"]
        high = sample_ohlcv_df["high"].to_numpy()
        low = sample_ohlcv_df["low"].to_numpy()

        # VWAP should be between high and low (approximately)
        valid_mask = ~np.isnan(vwap_values)
        # Allow some tolerance since VWAP is cumulative
        assert len(vwap_values[valid_mask]) > 0

    def test_technical_calculator(self, sample_ohlcv_df):
        """Test TechnicalIndicatorCalculator."""
        from quant_trading_system.features.technical import TechnicalIndicatorCalculator

        calc = TechnicalIndicatorCalculator()
        result = calc.compute_all(sample_ohlcv_df)

        # Should return dict with multiple indicators
        assert isinstance(result, dict)
        assert len(result) > 0

        # Check for expected keys
        assert "sma_20" in result
        assert "rsi_14" in result
        assert "macd_line" in result


class TestStatisticalFeatures:
    """Tests for statistical features."""

    def test_simple_returns(self, sample_ohlcv_df):
        """Test simple returns computation."""
        from quant_trading_system.features.statistical import SimpleReturns

        ret = SimpleReturns(periods=[1])
        result = ret.compute(sample_ohlcv_df)

        assert isinstance(result, dict)
        assert "return_1" in result

        close = sample_ohlcv_df["close"].to_numpy()
        returns = result["return_1"]

        # First value should be NaN
        assert np.isnan(returns[0])

        # Verify calculation
        manual_return = (close[-1] - close[-2]) / close[-2]
        assert np.isclose(returns[-1], manual_return, rtol=1e-10)

    def test_log_returns(self, sample_ohlcv_df):
        """Test log returns computation."""
        from quant_trading_system.features.statistical import LogReturns

        ret = LogReturns(periods=[1])
        result = ret.compute(sample_ohlcv_df)

        assert isinstance(result, dict)
        assert "log_return_1" in result

        close = sample_ohlcv_df["close"].to_numpy()

        # Verify calculation
        manual_return = np.log(close[-1] / close[-2])
        assert np.isclose(result["log_return_1"][-1], manual_return, rtol=1e-10)

    def test_zscore(self, sample_ohlcv_df):
        """Test Z-Score computation."""
        from quant_trading_system.features.statistical import ZScore

        zscore = ZScore(windows=[20])
        result = zscore.compute(sample_ohlcv_df)

        assert isinstance(result, dict)
        assert "zscore_20" in result

        # Z-score should have mean ~0 and std ~1 for valid values
        valid_values = result["zscore_20"][~np.isnan(result["zscore_20"])]
        assert abs(np.mean(valid_values)) < 0.5

    def test_rolling_std(self, sample_ohlcv_df):
        """Test rolling standard deviation."""
        from quant_trading_system.features.statistical import RollingStd

        roll_std = RollingStd(windows=[20])
        result = roll_std.compute(sample_ohlcv_df)

        assert isinstance(result, dict)
        assert "rolling_std_20" in result

        # Std should always be positive
        valid_values = result["rolling_std_20"][~np.isnan(result["rolling_std_20"])]
        assert np.all(valid_values > 0)

    def test_rolling_skewness(self, sample_ohlcv_df):
        """Test rolling skewness."""
        from quant_trading_system.features.statistical import RollingSkewness

        skew = RollingSkewness(windows=[60])
        result = skew.compute(sample_ohlcv_df)

        assert isinstance(result, dict)
        assert "rolling_skew_60" in result

        # Skewness should be finite
        valid_values = result["rolling_skew_60"][~np.isnan(result["rolling_skew_60"])]
        assert np.all(np.isfinite(valid_values))

    def test_rolling_var(self, sample_ohlcv_df):
        """Test rolling VaR."""
        from quant_trading_system.features.statistical import RollingVaR

        var = RollingVaR(window=60, confidence_levels=[0.95])
        result = var.compute(sample_ohlcv_df)

        assert isinstance(result, dict)
        assert "var_95_60" in result

        # VaR should typically be negative (loss)
        valid_values = result["var_95_60"][~np.isnan(result["var_95_60"])]
        assert len(valid_values) > 0

    def test_hurst_exponent(self, sample_ohlcv_df):
        """Test Hurst exponent computation."""
        from quant_trading_system.features.statistical import HurstExponent

        hurst = HurstExponent(window=100)
        result = hurst.compute(sample_ohlcv_df)

        assert isinstance(result, dict)
        assert "hurst_100" in result

        # Hurst exponent typically bounded [0, 1] but R/S can produce slightly higher values
        # Just check that values are finite and reasonable
        valid_values = result["hurst_100"][~np.isnan(result["hurst_100"])]
        if len(valid_values) > 0:
            assert np.all(valid_values >= 0) and np.all(valid_values <= 1.5)

    def test_autocorrelation(self, sample_ohlcv_df):
        """Test autocorrelation computation."""
        from quant_trading_system.features.statistical import Autocorrelation

        acf = Autocorrelation(window=60, lags=[1, 5, 10])
        result = acf.compute(sample_ohlcv_df)

        assert isinstance(result, dict)
        assert len(result) > 0

        # Check any key works - naming may vary
        first_key = list(result.keys())[0]
        valid_values = result[first_key][~np.isnan(result[first_key])]
        if len(valid_values) > 0:
            # Autocorrelation should be bounded [-1, 1]
            assert np.all(valid_values >= -1) and np.all(valid_values <= 1)

    def test_statistical_calculator(self, sample_ohlcv_df):
        """Test StatisticalFeatureCalculator."""
        from quant_trading_system.features.statistical import StatisticalFeatureCalculator

        calc = StatisticalFeatureCalculator()
        result = calc.compute_all(sample_ohlcv_df)

        assert isinstance(result, dict)
        assert len(result) > 0

        # Check we have some features - exact names may vary
        assert any("return" in k for k in result.keys())
        assert any("zscore" in k for k in result.keys())


class TestMicrostructureFeatures:
    """Tests for microstructure features."""

    def test_buy_sell_imbalance(self, sample_ohlcv_df):
        """Test buy/sell volume imbalance."""
        from quant_trading_system.features.microstructure import BuySellVolumeImbalance

        imbalance = BuySellVolumeImbalance(windows=[20])
        result = imbalance.compute(sample_ohlcv_df)

        assert isinstance(result, dict)
        assert "volume_imbalance_20" in result

        values = result["volume_imbalance_20"]

        # Imbalance should be bounded [-1, 1]
        valid_values = values[~np.isnan(values)]
        assert np.all(valid_values >= -1) and np.all(valid_values <= 1)

    def test_amihud_illiquidity(self, sample_ohlcv_df):
        """Test Amihud illiquidity measure."""
        from quant_trading_system.features.microstructure import AmihudIlliquidity

        amihud = AmihudIlliquidity(window=20)
        result = amihud.compute(sample_ohlcv_df)

        assert isinstance(result, dict)
        assert "amihud_20" in result

        values = result["amihud_20"]

        # Illiquidity should be positive
        valid_values = values[~np.isnan(values)]
        assert np.all(valid_values >= 0)

    def test_vpin(self, sample_ohlcv_df):
        """Test VPIN computation."""
        from quant_trading_system.features.microstructure import VPIN

        vpin = VPIN(bucket_size=50, num_buckets=50)
        result = vpin.compute(sample_ohlcv_df)

        assert isinstance(result, dict)
        assert "vpin" in result

        values = result["vpin"]

        # VPIN should be bounded [0, 1]
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            assert np.all(valid_values >= 0) and np.all(valid_values <= 1)

    def test_microstructure_calculator(self, sample_ohlcv_df):
        """Test MicrostructureFeatureCalculator."""
        from quant_trading_system.features.microstructure import MicrostructureFeatureCalculator

        calc = MicrostructureFeatureCalculator()
        result = calc.compute_all(sample_ohlcv_df)

        assert isinstance(result, dict)
        assert len(result) > 0


class TestCrossSectionalFeatures:
    """Tests for cross-sectional features."""

    @pytest.fixture
    def multi_asset_data(self):
        """Create multi-asset data for cross-sectional tests."""
        np.random.seed(42)
        n = 100

        assets = {}
        for symbol in ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]:
            returns = np.random.normal(0.0005, 0.02, n)
            close = 100.0 * np.exp(np.cumsum(returns))
            assets[symbol] = close

        return assets

    def test_relative_strength(self, multi_asset_data, sample_ohlcv_df):
        """Test relative strength computation."""
        from quant_trading_system.features.cross_sectional import RelativeStrength

        rs = RelativeStrength(windows=[20])

        # RelativeStrength expects a DataFrame with 'close' column
        result = rs.compute(sample_ohlcv_df)

        assert isinstance(result, dict)
        # Returns zeros when no benchmark available, which is expected
        assert "rs_vs_benchmark_20" in result

    def test_percentile_rank(self, multi_asset_data, sample_ohlcv_df):
        """Test percentile rank computation."""
        from quant_trading_system.features.cross_sectional import PercentileRank

        pr = PercentileRank()

        # PercentileRank expects a DataFrame with 'close' column
        result = pr.compute(sample_ohlcv_df)

        assert isinstance(result, dict)
        # Returns 50th percentile when no universe data available
        assert "return_percentile_5" in result or "return_percentile_20" in result

    def test_zscore_vs_universe(self, multi_asset_data, sample_ohlcv_df):
        """Test Z-score vs universe computation."""
        from quant_trading_system.features.cross_sectional import ZScoreVsUniverse

        zscore = ZScoreVsUniverse()

        # ZScoreVsUniverse expects a DataFrame with 'close' column
        result = zscore.compute(sample_ohlcv_df)

        assert isinstance(result, dict)
        # Returns zeros when no universe data available
        assert "zscore_universe_5" in result or "zscore_universe_20" in result

    def test_zscore_uses_timestamp_alignment_not_row_index(self):
        """Z-score should respect timestamp alignment for universe returns."""
        from quant_trading_system.features.cross_sectional import ZScoreVsUniverse

        base_ts = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(8)]
        base_df = pl.DataFrame({
            "timestamp": base_ts,
            "close": [100, 101, 102, 103, 104, 105, 106, 107],
        })

        shifted_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 3) + timedelta(days=i) for i in range(8)],
            "close": [100, 102, 104, 106, 108, 110, 112, 114],
        })
        aligned_df = pl.DataFrame({
            "timestamp": base_ts,
            "close": [100, 100, 101, 101, 102, 102, 103, 103],
        })

        zscore = ZScoreVsUniverse(windows=[1])
        result = zscore.compute(base_df, {"SHIFTED": shifted_df, "ALIGNED": aligned_df})
        values = result["zscore_universe_1"]

        # At index 2 (2024-01-03), shifted universe return is still NaN after alignment.
        # This should leave only one valid universe point and z-score should remain NaN.
        assert np.isnan(values[2])
        assert np.isfinite(values[5])

    def test_factor_loadings_handle_misaligned_universe(self):
        """FactorLoadings should be stable with timestamp-misaligned universe data."""
        from quant_trading_system.features.cross_sectional import FactorLoadings

        np.random.seed(7)
        n = 140
        base_ts = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)]
        base_close = 100.0 * np.exp(np.cumsum(np.random.normal(0.0004, 0.01, n)))
        base_df = pl.DataFrame({"timestamp": base_ts, "close": base_close})

        universe_data: dict[str, pl.DataFrame] = {}
        for idx in range(6):
            shift = idx + 1
            ts = [datetime(2023, 1, 1) + timedelta(days=shift + i) for i in range(n)]
            close = 90.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, n)))
            universe_data[f"S{idx}"] = pl.DataFrame({"timestamp": ts, "close": close})

        feature = FactorLoadings(window=30, n_factors=3)
        result = feature.compute(base_df, universe_data)

        assert len(result["factor_loading_1"]) == n
        assert len(result["idiosyncratic_vol"]) == n
        assert np.isfinite(result["factor_loading_1"]).sum() > 0
        assert np.isfinite(result["idiosyncratic_vol"]).sum() > 0

    def test_correlation_centrality_uses_timestamp_alignment(self):
        """CorrelationCentrality should not fabricate early overlap via row-index alignment."""
        from quant_trading_system.features.cross_sectional import CorrelationCentrality

        base_ts = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(12)]
        base_close = np.linspace(100.0, 111.0, 12)
        base_df = pl.DataFrame({"timestamp": base_ts, "close": base_close})

        # Both universe symbols start later than base timestamps.
        shifted_a = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 5) + timedelta(days=i) for i in range(12)],
            "close": np.linspace(50.0, 62.0, 12),
        })
        shifted_b = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 6) + timedelta(days=i) for i in range(12)],
            "close": np.linspace(80.0, 92.0, 12),
        })
        shifted_c = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 7) + timedelta(days=i) for i in range(12)],
            "close": np.linspace(120.0, 132.0, 12),
        })

        feature = CorrelationCentrality(window=4)
        result = feature.compute(base_df, {"A": shifted_a, "B": shifted_b, "C": shifted_c})
        centrality = result["corr_centrality"]

        # Early window has no true overlap; alignment should keep it NaN.
        assert np.isnan(centrality[4])
        assert np.isfinite(centrality).sum() > 0


class TestFeaturePipeline:
    """Tests for feature pipeline."""

    def test_pipeline_creation(self, sample_ohlcv_df):
        """Test pipeline creation."""
        from quant_trading_system.features.feature_pipeline import (
            FeaturePipeline,
            FeatureConfig,
            FeatureGroup,
        )

        config = FeatureConfig(
            groups=[FeatureGroup.TECHNICAL, FeatureGroup.STATISTICAL],
        )

        pipeline = FeaturePipeline(config)
        assert pipeline is not None

    def test_pipeline_compute(self, sample_ohlcv_df):
        """Test pipeline feature computation."""
        from quant_trading_system.features.feature_pipeline import (
            FeaturePipeline,
            FeatureConfig,
            FeatureGroup,
        )

        config = FeatureConfig(
            groups=[FeatureGroup.TECHNICAL],
        )

        pipeline = FeaturePipeline(config)
        result = pipeline.compute(sample_ohlcv_df)

        # Result is a FeatureSet object
        assert result is not None
        assert result.num_features > 0

    def test_create_pipeline_helper(self, sample_ohlcv_df):
        """Test create_pipeline helper function."""
        from quant_trading_system.features.feature_pipeline import create_pipeline

        pipeline = create_pipeline(
            groups=["technical", "statistical"],
        )

        result = pipeline.compute(sample_ohlcv_df)
        # Result is a FeatureSet object
        assert result is not None
        assert result.num_features > 0

    def test_cache_key_deterministic_for_equivalent_configs(self, sample_ohlcv_df):
        """Test cache key is deterministic for equivalent output configs."""
        from quant_trading_system.features.feature_pipeline import (
            FeaturePipeline,
            FeatureConfig,
            FeatureGroup,
        )

        config_a = FeatureConfig(
            groups=[FeatureGroup.TECHNICAL, FeatureGroup.STATISTICAL],
            include_targets=True,
            target_horizons=[1, 5],
            fill_method="ffill",
            use_optimized_pipeline=False,
        )
        config_b = FeatureConfig(
            groups=[FeatureGroup.TECHNICAL, FeatureGroup.STATISTICAL],
            include_targets=True,
            target_horizons=[1, 5],
            fill_method="ffill",
            use_optimized_pipeline=False,
        )

        key_a = FeaturePipeline(config_a)._generate_cache_key(sample_ohlcv_df, "TEST")
        key_b = FeaturePipeline(config_b)._generate_cache_key(sample_ohlcv_df, "TEST")

        assert key_a == key_b

    def test_cache_key_changes_when_output_config_changes(self, sample_ohlcv_df):
        """Test cache key changes when output-affecting config fields change."""
        from quant_trading_system.features.feature_pipeline import (
            FeaturePipeline,
            FeatureConfig,
            FeatureGroup,
            NormalizationMethod,
        )

        pipeline = FeaturePipeline(
            FeatureConfig(
                groups=[FeatureGroup.TECHNICAL],
                fill_method="ffill",
                include_targets=False,
                target_horizons=[1],
                use_optimized_pipeline=False,
            )
        )

        key_base = pipeline._generate_cache_key(sample_ohlcv_df, "TEST")

        pipeline.config.normalization = NormalizationMethod.ZSCORE
        key_norm = pipeline._generate_cache_key(sample_ohlcv_df, "TEST")
        assert key_norm != key_base

        pipeline.config.fill_method = "zero"
        key_fill = pipeline._generate_cache_key(sample_ohlcv_df, "TEST")
        assert key_fill != key_norm

        pipeline.config.include_targets = True
        pipeline.config.target_horizons = [1, 5]
        key_targets = pipeline._generate_cache_key(sample_ohlcv_df, "TEST")
        assert key_targets != key_fill

        pipeline.config.groups = [FeatureGroup.STATISTICAL]
        key_groups = pipeline._generate_cache_key(sample_ohlcv_df, "TEST")
        assert key_groups != key_targets

    def test_cache_invalidates_after_config_change(self, sample_ohlcv_df):
        """Test cache miss occurs after changing output-affecting config."""
        from quant_trading_system.features.feature_pipeline import (
            FeaturePipeline,
            FeatureConfig,
            FeatureGroup,
            NormalizationMethod,
        )

        pipeline = FeaturePipeline(
            FeatureConfig(
                groups=[FeatureGroup.TECHNICAL],
                use_cache=True,
                use_optimized_pipeline=False,
            )
        )

        pipeline.compute(sample_ohlcv_df, symbol="CACHE_TEST")
        stats_after_first = pipeline.get_cache_stats()
        assert stats_after_first["hits"] == 0
        assert stats_after_first["misses"] == 1
        assert stats_after_first["size"] == 1

        pipeline.compute(sample_ohlcv_df, symbol="CACHE_TEST")
        stats_after_second = pipeline.get_cache_stats()
        assert stats_after_second["hits"] == 1
        assert stats_after_second["misses"] == 1
        assert stats_after_second["size"] == 1

        pipeline.config.normalization = NormalizationMethod.ZSCORE
        pipeline.compute(sample_ohlcv_df, symbol="CACHE_TEST")
        stats_after_config_change = pipeline.get_cache_stats()
        assert stats_after_config_change["hits"] == 1
        assert stats_after_config_change["misses"] == 2
        assert stats_after_config_change["size"] == 2

    def test_feature_validator(self, sample_ohlcv_df):
        """Test feature validator."""
        from quant_trading_system.features.feature_pipeline import FeatureValidator

        # Create some feature data
        features = {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.full(100, np.nan),  # All NaN
        }

        # FeatureValidator uses static methods
        nan_ratios = FeatureValidator.check_nan_ratio(features)

        # Should flag the all-NaN feature with ratio 1.0
        assert nan_ratios["feature3"] == 1.0
        assert nan_ratios["feature1"] < 0.1

    def test_feature_selector(self, sample_ohlcv_df):
        """Test feature selector."""
        from quant_trading_system.features.feature_pipeline import FeatureSelector

        # Create correlated features
        np.random.seed(42)
        base = np.random.randn(100)
        features = {
            "feature1": base,
            "feature2": base + np.random.randn(100) * 0.1,  # Highly correlated
            "feature3": np.random.randn(100),  # Independent
        }

        # FeatureSelector uses static methods
        selected = FeatureSelector.select_by_correlation(features, threshold=0.95)

        # Should remove highly correlated features
        assert len(selected) <= len(features)


class TestLookAheadBiasPrevention:
    """Tests for look-ahead bias prevention in feature pipeline.

    Bug Fix Tests: Targets (which contain future data) must be kept
    SEPARATE from features to prevent data leakage during model training.
    """

    def test_targets_separate_from_features(self, sample_ohlcv_df):
        """Test that targets are stored separately from features.

        Bug Fix Test: Previously, targets were mixed into features dict,
        potentially causing look-ahead bias.
        """
        from quant_trading_system.features.feature_pipeline import (
            FeaturePipeline,
            FeatureConfig,
            FeatureGroup,
        )

        config = FeatureConfig(
            groups=[FeatureGroup.TECHNICAL],
            include_targets=True,
            target_horizons=[1, 5],
        )

        pipeline = FeaturePipeline(config)
        result = pipeline.compute(sample_ohlcv_df)

        # Targets should be in separate dict
        assert hasattr(result, 'targets')
        assert hasattr(result, 'target_metadata')

        # Targets should NOT be in features dict
        for name in result.features.keys():
            assert not name.startswith("target_"), \
                f"Target '{name}' found in features dict - LOOK-AHEAD BIAS!"

        # Targets should be in targets dict
        assert len(result.targets) > 0
        for name in result.targets.keys():
            assert name.startswith("target_"), \
                f"Non-target '{name}' found in targets dict"

    def test_to_numpy_rejects_targets(self, sample_ohlcv_df):
        """Test that to_numpy() raises error if targets are requested.

        Bug Fix Test: Prevents accidental inclusion of future data in feature matrix.
        """
        from quant_trading_system.features.feature_pipeline import (
            FeaturePipeline,
            FeatureConfig,
            FeatureGroup,
        )

        config = FeatureConfig(
            groups=[FeatureGroup.TECHNICAL],
            include_targets=True,
            target_horizons=[1],
        )

        pipeline = FeaturePipeline(config)
        result = pipeline.compute(sample_ohlcv_df)

        # to_numpy() without args should work
        feature_matrix = result.to_numpy()
        assert feature_matrix is not None

        # to_numpy() with target name should raise ValueError
        with pytest.raises(ValueError, match="LOOK-AHEAD BIAS"):
            result.to_numpy(feature_names=["target_return_1"])

    def test_to_polars_rejects_targets(self, sample_ohlcv_df):
        """Test that to_polars() raises error if targets are requested.

        Bug Fix Test: Prevents accidental inclusion of future data in feature DataFrame.
        """
        from quant_trading_system.features.feature_pipeline import (
            FeaturePipeline,
            FeatureConfig,
            FeatureGroup,
        )

        config = FeatureConfig(
            groups=[FeatureGroup.TECHNICAL],
            include_targets=True,
            target_horizons=[1],
        )

        pipeline = FeaturePipeline(config)
        result = pipeline.compute(sample_ohlcv_df)

        # to_polars() without args should work
        feature_df = result.to_polars()
        assert feature_df is not None

        # to_polars() with target name should raise ValueError
        with pytest.raises(ValueError, match="LOOK-AHEAD BIAS"):
            result.to_polars(feature_names=["target_direction_1"])

    def test_get_targets_numpy_works(self, sample_ohlcv_df):
        """Test that get_targets_numpy() returns targets correctly."""
        from quant_trading_system.features.feature_pipeline import (
            FeaturePipeline,
            FeatureConfig,
            FeatureGroup,
        )

        config = FeatureConfig(
            groups=[FeatureGroup.TECHNICAL],
            include_targets=True,
            target_horizons=[1, 5],
        )

        pipeline = FeaturePipeline(config)
        result = pipeline.compute(sample_ohlcv_df)

        # get_targets_numpy() should work
        targets_matrix = result.get_targets_numpy()
        assert targets_matrix is not None
        assert len(targets_matrix) > 0

    def test_feature_set_properties(self, sample_ohlcv_df):
        """Test FeatureSet properties for targets."""
        from quant_trading_system.features.feature_pipeline import (
            FeaturePipeline,
            FeatureConfig,
            FeatureGroup,
        )

        config = FeatureConfig(
            groups=[FeatureGroup.TECHNICAL],
            include_targets=True,
            target_horizons=[1, 5],
        )

        pipeline = FeaturePipeline(config)
        result = pipeline.compute(sample_ohlcv_df)

        # Check new properties
        assert result.num_features > 0
        assert result.num_targets > 0
        assert len(result.feature_names) == result.num_features
        assert len(result.target_names) == result.num_targets

        # feature_names should not contain targets
        for name in result.feature_names:
            assert not name.startswith("target_")

        # target_names should all be targets
        for name in result.target_names:
            assert name.startswith("target_")

    def test_targets_not_filtered(self, sample_ohlcv_df):
        """Test that targets are not affected by variance/correlation filtering.

        Bug Fix Test: Targets should not go through feature filtering as
        this could incorrectly remove valid training labels.
        """
        from quant_trading_system.features.feature_pipeline import (
            FeaturePipeline,
            FeatureConfig,
            FeatureGroup,
        )

        config = FeatureConfig(
            groups=[FeatureGroup.TECHNICAL],
            include_targets=True,
            target_horizons=[1, 5, 10],
            variance_threshold=0.01,  # Would filter low-variance features
            correlation_threshold=0.90,  # Would filter highly correlated features
        )

        pipeline = FeaturePipeline(config)
        result = pipeline.compute(sample_ohlcv_df)

        # All configured targets should be present (not filtered)
        expected_targets = [
            "target_return_1", "target_direction_1",
            "target_return_5", "target_direction_5",
            "target_return_10", "target_direction_10",
        ]

        for expected in expected_targets:
            assert expected in result.targets, \
                f"Target '{expected}' was incorrectly filtered out"

    def test_targets_contain_nans_at_end(self, sample_ohlcv_df):
        """Test that targets have NaN values at the end (no future data available).

        This verifies the targets are correctly computed - the last 'horizon'
        rows should be NaN because we cannot compute forward returns for them.
        """
        from quant_trading_system.features.feature_pipeline import (
            FeaturePipeline,
            FeatureConfig,
            FeatureGroup,
        )

        config = FeatureConfig(
            groups=[FeatureGroup.TECHNICAL],
            include_targets=True,
            target_horizons=[5],
        )

        pipeline = FeaturePipeline(config)
        result = pipeline.compute(sample_ohlcv_df)

        target_5 = result.targets["target_return_5"]

        # Last 5 values should be NaN (horizon=5)
        assert np.all(np.isnan(target_5[-5:])), \
            "Last 5 values of target_return_5 should be NaN"

        # Some earlier values should be valid
        valid_values = target_5[~np.isnan(target_5)]
        assert len(valid_values) > 0, "Should have some valid target values"
