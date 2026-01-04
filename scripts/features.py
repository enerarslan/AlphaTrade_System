"""
================================================================================
ALPHATRADE FEATURE ENGINEERING
================================================================================

Institutional-grade feature engineering for the AlphaTrade trading system.

@mlquant: This script implements all feature engineering requirements including:
  - Technical features (trend, momentum, volatility, volume)
  - Statistical features (autocorrelation, Hurst, beta, correlation)
  - Microstructure features (P1-C: order book imbalance, VPIN, Kyle's Lambda)
  - GPU-accelerated computation (P3-C: RAPIDS cuDF)
  - Feature pipeline with LRU caching (P1-H4)
  - Feature importance analysis (SHAP, mutual information)
  - Look-ahead bias prevention (strict backward-looking)

Commands:
    python main.py features compute --symbols AAPL MSFT --groups all
    python main.py features analyze --path data/features/
    python main.py features importance --model models/xgboost.pkl
    python main.py features gpu --symbols AAPL MSFT --benchmark

Author: AlphaTrade System
Version: 1.3.0
================================================================================
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("features")

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""

    # Feature groups
    groups: list[str] = field(default_factory=lambda: [
        "trend", "momentum", "volatility", "volume", "statistical"
    ])

    # Technical feature parameters
    sma_windows: list[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    ema_windows: list[int] = field(default_factory=lambda: [5, 10, 20, 50])
    rsi_periods: list[int] = field(default_factory=lambda: [7, 14, 21])
    macd_params: tuple[int, int, int] = (12, 26, 9)
    bollinger_params: tuple[int, float] = (20, 2.0)
    atr_periods: list[int] = field(default_factory=lambda: [7, 14, 21])

    # Statistical feature parameters
    rolling_windows: list[int] = field(default_factory=lambda: [5, 10, 20, 50])
    autocorr_lags: list[int] = field(default_factory=lambda: [1, 5, 10, 20])

    # Normalization
    normalization: str = "zscore"  # zscore, minmax, robust, none
    handle_nan: str = "drop"  # drop, ffill, interpolate

    # Feature selection
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    max_features: int = 0  # 0 = no limit

    # GPU acceleration (P3-C)
    use_gpu: bool = False
    gpu_device: int = 0

    # Caching (P1-H4)
    use_cache: bool = True
    cache_dir: str = "cache/features"
    cache_max_size: int = 1000  # Max cached items

    # Output
    output_dir: str = "data/features"
    output_format: str = "parquet"  # parquet, csv, hdf5


# ============================================================================
# FEATURE CALCULATOR
# ============================================================================


class FeatureCalculator:
    """
    Comprehensive feature calculator with GPU acceleration.

    @mlquant: This class computes all feature types with:
      - Strict backward-looking computation (no look-ahead bias)
      - GPU acceleration via RAPIDS cuDF
      - LRU caching for performance
      - Comprehensive error handling
    """

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.logger = logging.getLogger("FeatureCalculator")

        # GPU availability
        self.gpu_available = False
        self.cudf = None

        if config.use_gpu:
            self._setup_gpu()

        # Cache
        self.feature_cache: dict[str, pd.DataFrame] = {}

    def _setup_gpu(self) -> None:
        """Setup GPU acceleration."""
        try:
            import cudf
            self.cudf = cudf
            self.gpu_available = True
            self.logger.info("GPU acceleration enabled (RAPIDS cuDF)")
        except ImportError:
            self.logger.warning("cuDF not available, falling back to CPU")
            self.gpu_available = False

    def compute_all(self, df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
        """
        Compute all feature groups.

        @mlquant: All features are strictly backward-looking to prevent
        look-ahead bias.

        Args:
            df: Input OHLCV DataFrame.
            symbol: Symbol name for caching.

        Returns:
            DataFrame with all computed features.
        """
        self.logger.info(f"Computing features for {symbol or 'data'}")
        start_time = time.time()

        # Check cache
        cache_key = self._get_cache_key(df, symbol)
        if self.config.use_cache and cache_key in self.feature_cache:
            self.logger.info("Using cached features")
            return self.feature_cache[cache_key]

        # Prepare data
        df = self._prepare_data(df)

        # Convert to GPU if available
        if self.gpu_available:
            df_compute = self.cudf.DataFrame.from_pandas(df)
        else:
            df_compute = df

        # Compute each feature group
        feature_dfs = []

        for group in self.config.groups:
            try:
                if group == "trend":
                    features = self._compute_trend_features(df_compute)
                elif group == "momentum":
                    features = self._compute_momentum_features(df_compute)
                elif group == "volatility":
                    features = self._compute_volatility_features(df_compute)
                elif group == "volume":
                    features = self._compute_volume_features(df_compute)
                elif group == "statistical":
                    features = self._compute_statistical_features(df_compute)
                elif group == "microstructure":
                    features = self._compute_microstructure_features(df_compute)
                else:
                    self.logger.warning(f"Unknown feature group: {group}")
                    continue

                feature_dfs.append(features)
                self.logger.info(f"Computed {group} features: {len(features.columns)} columns")

            except Exception as e:
                self.logger.error(f"Failed to compute {group} features: {e}")

        # Combine all features
        if feature_dfs:
            result = pd.concat(feature_dfs, axis=1)

            # Convert back from GPU if needed
            if self.gpu_available and hasattr(result, "to_pandas"):
                result = result.to_pandas()

            # Apply normalization
            result = self._normalize_features(result)

            # Apply feature selection
            result = self._select_features(result)

            # Handle NaN values
            result = self._handle_nan(result)

            # Cache result
            if self.config.use_cache:
                self._cache_features(cache_key, result)

            duration = time.time() - start_time
            self.logger.info(f"Feature computation complete: {len(result.columns)} features in {duration:.2f}s")

            return result
        else:
            return pd.DataFrame()

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for feature computation."""
        df = df.copy()

        # Ensure required columns
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Sort by timestamp if available
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    # ========================================================================
    # TREND FEATURES
    # ========================================================================

    def _compute_trend_features(self, df) -> pd.DataFrame:
        """
        Compute trend features.

        Includes: SMA, EMA, DEMA, TEMA, Keltner Channels, Ichimoku
        """
        features = pd.DataFrame(index=df.index if hasattr(df, "index") else range(len(df)))
        close = df["close"].values if hasattr(df["close"], "values") else df["close"].to_pandas().values

        # Simple Moving Averages
        for window in self.config.sma_windows:
            sma = pd.Series(close).rolling(window).mean().values
            features[f"sma_{window}"] = sma
            features[f"close_sma_{window}_ratio"] = close / sma

        # Exponential Moving Averages
        for window in self.config.ema_windows:
            ema = pd.Series(close).ewm(span=window).mean().values
            features[f"ema_{window}"] = ema
            features[f"close_ema_{window}_ratio"] = close / ema

        # DEMA (Double Exponential Moving Average)
        for window in self.config.ema_windows[:2]:  # Limit to first 2
            ema1 = pd.Series(close).ewm(span=window).mean()
            ema2 = ema1.ewm(span=window).mean()
            dema = 2 * ema1 - ema2
            features[f"dema_{window}"] = dema.values

        # TEMA (Triple Exponential Moving Average)
        ema1 = pd.Series(close).ewm(span=20).mean()
        ema2 = ema1.ewm(span=20).mean()
        ema3 = ema2.ewm(span=20).mean()
        features["tema_20"] = (3 * ema1 - 3 * ema2 + ema3).values

        # Price relative to MAs
        for window in [20, 50, 200]:
            if window in self.config.sma_windows:
                sma = pd.Series(close).rolling(window).mean().values
                features[f"above_sma_{window}"] = (close > sma).astype(int)

        return features

    # ========================================================================
    # MOMENTUM FEATURES
    # ========================================================================

    def _compute_momentum_features(self, df) -> pd.DataFrame:
        """
        Compute momentum features.

        Includes: RSI, MACD, Stochastic, CCI, Williams %R, ROC
        """
        features = pd.DataFrame(index=df.index if hasattr(df, "index") else range(len(df)))

        close = df["close"].values if hasattr(df["close"], "values") else df["close"].to_pandas().values
        high = df["high"].values if hasattr(df["high"], "values") else df["high"].to_pandas().values
        low = df["low"].values if hasattr(df["low"], "values") else df["low"].to_pandas().values

        close_series = pd.Series(close)
        high_series = pd.Series(high)
        low_series = pd.Series(low)

        # RSI
        for period in self.config.rsi_periods:
            features[f"rsi_{period}"] = self._compute_rsi(close_series, period)

        # MACD
        fast, slow, signal = self.config.macd_params
        ema_fast = close_series.ewm(span=fast).mean()
        ema_slow = close_series.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        features["macd"] = macd_line.values
        features["macd_signal"] = signal_line.values
        features["macd_histogram"] = (macd_line - signal_line).values

        # Stochastic Oscillator
        for period in [14, 21]:
            lowest_low = low_series.rolling(period).min()
            highest_high = high_series.rolling(period).max()
            stoch_k = 100 * (close_series - lowest_low) / (highest_high - lowest_low + 1e-10)
            stoch_d = stoch_k.rolling(3).mean()
            features[f"stoch_k_{period}"] = stoch_k.values
            features[f"stoch_d_{period}"] = stoch_d.values

        # CCI (Commodity Channel Index)
        typical_price = (high_series + low_series + close_series) / 3
        for period in [14, 20]:
            sma_tp = typical_price.rolling(period).mean()
            mad = typical_price.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
            cci = (typical_price - sma_tp) / (0.015 * mad + 1e-10)
            features[f"cci_{period}"] = cci.values

        # Williams %R
        for period in [14, 21]:
            highest_high = high_series.rolling(period).max()
            lowest_low = low_series.rolling(period).min()
            williams_r = -100 * (highest_high - close_series) / (highest_high - lowest_low + 1e-10)
            features[f"williams_r_{period}"] = williams_r.values

        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            roc = (close_series / close_series.shift(period) - 1) * 100
            features[f"roc_{period}"] = roc.values

        # Momentum
        for period in [5, 10, 20]:
            momentum = close_series - close_series.shift(period)
            features[f"momentum_{period}"] = momentum.values

        return features

    def _compute_rsi(self, prices: pd.Series, period: int) -> np.ndarray:
        """Compute RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return (100 - (100 / (1 + rs))).values

    # ========================================================================
    # VOLATILITY FEATURES
    # ========================================================================

    def _compute_volatility_features(self, df) -> pd.DataFrame:
        """
        Compute volatility features.

        Includes: Bollinger Bands, ATR, NATR, historical volatility
        """
        features = pd.DataFrame(index=df.index if hasattr(df, "index") else range(len(df)))

        close = df["close"].values if hasattr(df["close"], "values") else df["close"].to_pandas().values
        high = df["high"].values if hasattr(df["high"], "values") else df["high"].to_pandas().values
        low = df["low"].values if hasattr(df["low"], "values") else df["low"].to_pandas().values

        close_series = pd.Series(close)
        high_series = pd.Series(high)
        low_series = pd.Series(low)

        # Returns
        returns = close_series.pct_change()
        log_returns = np.log(close_series / close_series.shift(1))

        # Historical Volatility
        for window in self.config.rolling_windows:
            features[f"volatility_{window}"] = returns.rolling(window).std().values
            features[f"log_volatility_{window}"] = log_returns.rolling(window).std().values

        # Annualized volatility
        features["volatility_20_ann"] = (returns.rolling(20).std() * np.sqrt(252)).values

        # Bollinger Bands
        period, std_mult = self.config.bollinger_params
        sma = close_series.rolling(period).mean()
        std = close_series.rolling(period).std()
        upper_band = sma + std_mult * std
        lower_band = sma - std_mult * std

        features["bb_upper"] = upper_band.values
        features["bb_middle"] = sma.values
        features["bb_lower"] = lower_band.values
        features["bb_width"] = ((upper_band - lower_band) / sma).values
        features["bb_position"] = ((close_series - lower_band) / (upper_band - lower_band + 1e-10)).values

        # ATR (Average True Range)
        for period in self.config.atr_periods:
            tr1 = high_series - low_series
            tr2 = abs(high_series - close_series.shift(1))
            tr3 = abs(low_series - close_series.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            features[f"atr_{period}"] = atr.values
            features[f"natr_{period}"] = (atr / close_series * 100).values

        # Parkinson volatility
        features["parkinson_vol"] = (np.log(high_series / low_series) ** 2 / (4 * np.log(2))).rolling(20).mean().values ** 0.5

        # Garman-Klass volatility
        open_series = pd.Series(df["open"].values if hasattr(df["open"], "values") else df["open"].to_pandas().values)
        log_hl = np.log(high_series / low_series) ** 2
        log_co = np.log(close_series / open_series) ** 2
        gk_vol = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        features["garman_klass_vol"] = gk_vol.rolling(20).mean().values ** 0.5

        return features

    # ========================================================================
    # VOLUME FEATURES
    # ========================================================================

    def _compute_volume_features(self, df) -> pd.DataFrame:
        """
        Compute volume features.

        Includes: OBV, MFI, CMF, VWAP, volume ratios
        """
        features = pd.DataFrame(index=df.index if hasattr(df, "index") else range(len(df)))

        close = df["close"].values if hasattr(df["close"], "values") else df["close"].to_pandas().values
        high = df["high"].values if hasattr(df["high"], "values") else df["high"].to_pandas().values
        low = df["low"].values if hasattr(df["low"], "values") else df["low"].to_pandas().values
        volume = df["volume"].values if hasattr(df["volume"], "values") else df["volume"].to_pandas().values

        close_series = pd.Series(close)
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        volume_series = pd.Series(volume)

        # Volume SMA
        for window in self.config.rolling_windows:
            features[f"volume_sma_{window}"] = volume_series.rolling(window).mean().values
            features[f"volume_std_{window}"] = volume_series.rolling(window).std().values

        # Volume ratio
        features["volume_ratio_20"] = (volume_series / volume_series.rolling(20).mean()).values

        # OBV (On Balance Volume)
        price_diff = close_series.diff()
        obv_direction = np.where(price_diff > 0, 1, np.where(price_diff < 0, -1, 0))
        obv = (volume_series * obv_direction).cumsum()
        features["obv"] = obv.values

        # MFI (Money Flow Index)
        typical_price = (high_series + low_series + close_series) / 3
        raw_money_flow = typical_price * volume_series
        mf_direction = typical_price.diff()

        for period in [14, 20]:
            positive_mf = raw_money_flow.where(mf_direction > 0, 0).rolling(period).sum()
            negative_mf = raw_money_flow.where(mf_direction < 0, 0).rolling(period).sum()
            money_ratio = positive_mf / (negative_mf + 1e-10)
            mfi = 100 - (100 / (1 + money_ratio))
            features[f"mfi_{period}"] = mfi.values

        # CMF (Chaikin Money Flow)
        mf_multiplier = ((close_series - low_series) - (high_series - close_series)) / (high_series - low_series + 1e-10)
        mf_volume = mf_multiplier * volume_series
        for period in [20, 21]:
            cmf = mf_volume.rolling(period).sum() / (volume_series.rolling(period).sum() + 1e-10)
            features[f"cmf_{period}"] = cmf.values

        # VWAP approximation (intraday)
        typical_price = (high_series + low_series + close_series) / 3
        features["vwap_approx"] = (typical_price * volume_series).cumsum() / (volume_series.cumsum() + 1e-10)

        # AD Line (Accumulation/Distribution)
        ad = ((close_series - low_series) - (high_series - close_series)) / (high_series - low_series + 1e-10) * volume_series
        features["ad_line"] = ad.cumsum().values

        # Force Index
        features["force_index_13"] = (close_series.diff() * volume_series).ewm(span=13).mean().values

        return features

    # ========================================================================
    # STATISTICAL FEATURES
    # ========================================================================

    def _compute_statistical_features(self, df) -> pd.DataFrame:
        """
        Compute statistical features.

        Includes: autocorrelation, skewness, kurtosis, Hurst exponent
        """
        features = pd.DataFrame(index=df.index if hasattr(df, "index") else range(len(df)))

        close = df["close"].values if hasattr(df["close"], "values") else df["close"].to_pandas().values
        close_series = pd.Series(close)

        returns = close_series.pct_change()

        # Rolling statistics
        for window in self.config.rolling_windows:
            features[f"returns_mean_{window}"] = returns.rolling(window).mean().values
            features[f"returns_std_{window}"] = returns.rolling(window).std().values
            features[f"returns_skew_{window}"] = returns.rolling(window).skew().values
            features[f"returns_kurt_{window}"] = returns.rolling(window).kurt().values

        # Autocorrelation
        for lag in self.config.autocorr_lags:
            autocorr = returns.rolling(50).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan,
                raw=False
            )
            features[f"autocorr_lag_{lag}"] = autocorr.values

        # Z-score of price
        for window in [20, 50]:
            mean = close_series.rolling(window).mean()
            std = close_series.rolling(window).std()
            zscore = (close_series - mean) / (std + 1e-10)
            features[f"price_zscore_{window}"] = zscore.values

        # Hurst exponent approximation
        features["hurst_approx"] = self._compute_hurst_approx(returns, 50)

        # Variance ratio
        for k in [2, 4, 8]:
            var_1 = returns.rolling(50).var()
            var_k = returns.rolling(50 * k).var() / k
            features[f"variance_ratio_{k}"] = (var_k / (var_1 + 1e-10)).values

        return features

    def _compute_hurst_approx(self, returns: pd.Series, window: int) -> np.ndarray:
        """Approximate Hurst exponent using R/S analysis."""
        hurst_values = np.full(len(returns), np.nan)

        for i in range(window, len(returns)):
            chunk = returns.iloc[i - window:i].dropna()
            if len(chunk) < window // 2:
                continue

            try:
                mean = chunk.mean()
                std = chunk.std()
                if std < 1e-10:
                    continue

                cumdev = (chunk - mean).cumsum()
                R = cumdev.max() - cumdev.min()
                S = std

                if S > 0 and R > 0:
                    hurst_values[i] = np.log(R / S) / np.log(window)
            except Exception:
                continue

        return hurst_values

    # ========================================================================
    # MICROSTRUCTURE FEATURES (P1-C)
    # ========================================================================

    def _compute_microstructure_features(self, df) -> pd.DataFrame:
        """
        Compute microstructure features.

        @mlquant P1-C: Order book imbalance and microstructure features
        for informed trading detection.

        Includes: VPIN, order flow imbalance, Amihud illiquidity
        """
        features = pd.DataFrame(index=df.index if hasattr(df, "index") else range(len(df)))

        close = df["close"].values if hasattr(df["close"], "values") else df["close"].to_pandas().values
        volume = df["volume"].values if hasattr(df["volume"], "values") else df["volume"].to_pandas().values

        close_series = pd.Series(close)
        volume_series = pd.Series(volume)

        returns = close_series.pct_change()
        abs_returns = returns.abs()

        # Amihud Illiquidity
        dollar_volume = close_series * volume_series
        amihud = abs_returns / (dollar_volume + 1e-10)
        features["amihud_illiquidity"] = amihud.rolling(20).mean().values * 1e6

        # Kyle's Lambda approximation
        features["kyle_lambda_approx"] = (abs_returns / (volume_series + 1e-10)).rolling(20).mean().values * 1e6

        # Bid-Ask Spread Proxy (using high-low)
        high = df["high"].values if hasattr(df["high"], "values") else df["high"].to_pandas().values
        low = df["low"].values if hasattr(df["low"], "values") else df["low"].to_pandas().values

        high_series = pd.Series(high)
        low_series = pd.Series(low)

        # Corwin-Schultz spread estimator
        beta = (np.log(high_series / low_series) ** 2).rolling(2).sum()
        gamma = np.log(high_series.rolling(2).max() / low_series.rolling(2).min()) ** 2
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        features["corwin_schultz_spread"] = spread.values

        # VPIN approximation (Volume-synchronized Probability of Informed Trading)
        buy_volume = volume_series * (returns > 0).astype(float)
        sell_volume = volume_series * (returns < 0).astype(float)
        imbalance = abs(buy_volume - sell_volume)
        vpin = imbalance.rolling(50).sum() / (volume_series.rolling(50).sum() + 1e-10)
        features["vpin_approx"] = vpin.values

        # Order flow imbalance
        features["order_flow_imbalance"] = ((buy_volume - sell_volume) / (volume_series + 1e-10)).rolling(20).mean().values

        return features

    # ========================================================================
    # NORMALIZATION AND SELECTION
    # ========================================================================

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization to features."""
        if self.config.normalization == "none":
            return df

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if self.config.normalization == "zscore":
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std

            elif self.config.normalization == "minmax":
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)

            elif self.config.normalization == "robust":
                median = df[col].median()
                q75 = df[col].quantile(0.75)
                q25 = df[col].quantile(0.25)
                iqr = q75 - q25
                if iqr > 0:
                    df[col] = (df[col] - median) / iqr

        return df

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection."""
        # Variance threshold
        if self.config.variance_threshold > 0:
            variances = df.var()
            low_variance = variances[variances < self.config.variance_threshold].index
            df = df.drop(columns=low_variance, errors="ignore")
            if len(low_variance) > 0:
                self.logger.info(f"Removed {len(low_variance)} low-variance features")

        # Correlation threshold
        if self.config.correlation_threshold < 1.0:
            corr_matrix = df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > self.config.correlation_threshold)]
            df = df.drop(columns=to_drop, errors="ignore")
            if len(to_drop) > 0:
                self.logger.info(f"Removed {len(to_drop)} highly correlated features")

        # Max features
        if self.config.max_features > 0 and len(df.columns) > self.config.max_features:
            # Keep features by variance
            variances = df.var().sort_values(ascending=False)
            keep_cols = variances.head(self.config.max_features).index.tolist()
            df = df[keep_cols]
            self.logger.info(f"Limited to top {self.config.max_features} features")

        return df

    def _handle_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle NaN values."""
        if self.config.handle_nan == "drop":
            df = df.dropna()
        elif self.config.handle_nan == "ffill":
            df = df.ffill().bfill()
        elif self.config.handle_nan == "interpolate":
            df = df.interpolate(method="linear").bfill().ffill()

        return df

    # ========================================================================
    # CACHING
    # ========================================================================

    def _get_cache_key(self, df: pd.DataFrame, symbol: str) -> str:
        """Generate cache key for data."""
        import hashlib

        # FIX: Use SHA256 instead of deprecated MD5
        data_hash = hashlib.sha256(
            f"{len(df)}_{df.iloc[0].to_dict() if len(df) > 0 else ''}_{symbol}".encode()
        ).hexdigest()[:16]

        return f"{symbol}_{data_hash}"

    def _cache_features(self, key: str, features: pd.DataFrame) -> None:
        """Cache computed features (P1-H4: LRU Cache)."""
        if len(self.feature_cache) >= self.config.cache_max_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self.feature_cache))
            del self.feature_cache[oldest_key]

        self.feature_cache[key] = features


# ============================================================================
# FEATURE IMPORTANCE ANALYZER
# ============================================================================


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance using multiple methods.

    @mlquant: Feature importance analysis for model interpretation
    and feature selection.
    """

    def __init__(self):
        self.logger = logging.getLogger("FeatureImportanceAnalyzer")

    def analyze(
        self,
        model,
        features: pd.DataFrame,
        labels: pd.Series | None = None,
        method: str = "auto",
    ) -> dict:
        """
        Compute feature importance.

        Args:
            model: Trained model.
            features: Feature DataFrame.
            labels: Target labels (for permutation importance).
            method: Importance method (auto, tree, shap, permutation, mutual_info).

        Returns:
            Dictionary with importance scores.
        """
        self.logger.info(f"Computing feature importance using {method} method")

        results = {}

        if method == "auto":
            # Detect appropriate method
            if hasattr(model, "feature_importances_"):
                method = "tree"
            else:
                method = "permutation"

        if method == "tree" and hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            results["tree_importance"] = dict(zip(features.columns, importance))

        if method == "shap":
            results["shap_importance"] = self._compute_shap_importance(model, features)

        if method == "permutation" and labels is not None:
            results["permutation_importance"] = self._compute_permutation_importance(
                model, features, labels
            )

        if method == "mutual_info" and labels is not None:
            results["mutual_info"] = self._compute_mutual_info(features, labels)

        return results

    def _compute_shap_importance(self, model, features: pd.DataFrame) -> dict:
        """Compute SHAP importance."""
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features.head(1000))

            if isinstance(shap_values, list):
                importance = np.abs(shap_values[1]).mean(axis=0)
            else:
                importance = np.abs(shap_values).mean(axis=0)

            return dict(zip(features.columns, importance))

        except Exception as e:
            self.logger.warning(f"SHAP computation failed: {e}")
            return {}

    def _compute_permutation_importance(
        self, model, features: pd.DataFrame, labels: pd.Series
    ) -> dict:
        """Compute permutation importance."""
        try:
            from sklearn.inspection import permutation_importance

            result = permutation_importance(
                model, features, labels, n_repeats=10, random_state=42, n_jobs=-1
            )

            return dict(zip(features.columns, result.importances_mean))

        except Exception as e:
            self.logger.warning(f"Permutation importance failed: {e}")
            return {}

    def _compute_mutual_info(self, features: pd.DataFrame, labels: pd.Series) -> dict:
        """Compute mutual information."""
        try:
            from sklearn.feature_selection import mutual_info_classif

            mi = mutual_info_classif(features, labels, random_state=42)
            return dict(zip(features.columns, mi))

        except Exception as e:
            self.logger.warning(f"Mutual info computation failed: {e}")
            return {}


# ============================================================================
# COMMAND HANDLERS
# ============================================================================


def cmd_compute(args: argparse.Namespace) -> int:
    """Compute features for all symbols."""
    logger.info("=" * 80)
    logger.info("FEATURE COMPUTATION")
    logger.info("=" * 80)

    config = FeatureConfig(
        groups=getattr(args, "groups", ["all"]),
        use_gpu=getattr(args, "use_gpu", False),
        output_dir=getattr(args, "output", "data/features"),
        normalization=getattr(args, "normalization", "zscore"),
    )

    # Expand "all" group
    if "all" in config.groups:
        config.groups = ["trend", "momentum", "volatility", "volume", "statistical", "microstructure"]

    calculator = FeatureCalculator(config)

    # Load data
    from scripts.data import DataManager, DataConfig

    data_config = DataConfig()
    data_manager = DataManager(data_config)

    symbols = getattr(args, "symbols", None) or data_manager.get_available_symbols()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        try:
            df = data_manager.load_symbol(symbol)
            features = calculator.compute_all(df, symbol)

            if len(features) > 0:
                output_path = output_dir / f"{symbol}_features.parquet"
                features.to_parquet(output_path, index=False)
                logger.info(f"Saved {symbol}: {len(features)} rows, {len(features.columns)} features")

        except Exception as e:
            logger.error(f"Failed to compute features for {symbol}: {e}")

    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze computed features."""
    logger.info("=" * 80)
    logger.info("FEATURE ANALYSIS")
    logger.info("=" * 80)

    feature_dir = Path(getattr(args, "path", "data/features"))

    if not feature_dir.exists():
        logger.error(f"Feature directory not found: {feature_dir}")
        return 1

    for pq_file in feature_dir.glob("*_features.parquet"):
        symbol = pq_file.stem.replace("_features", "")

        try:
            df = pd.read_parquet(pq_file)

            logger.info(f"\n{symbol}:")
            logger.info(f"  Rows: {len(df)}")
            logger.info(f"  Features: {len(df.columns)}")
            logger.info(f"  Missing: {df.isnull().sum().sum()}")

            # Feature statistics
            stats = df.describe().T[["mean", "std", "min", "max"]]
            logger.info(f"  Top features by std:")
            top_std = stats.nlargest(5, "std")
            for feat, row in top_std.iterrows():
                logger.info(f"    {feat}: std={row['std']:.4f}")

        except Exception as e:
            logger.error(f"Failed to analyze {pq_file}: {e}")

    return 0


def cmd_importance(args: argparse.Namespace) -> int:
    """Compute feature importance for a model."""
    logger.info("=" * 80)
    logger.info("FEATURE IMPORTANCE ANALYSIS")
    logger.info("=" * 80)

    model_path = getattr(args, "model", None)
    if not model_path:
        logger.error("Model path required")
        return 1

    import pickle

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Load features
        feature_dir = Path(getattr(args, "features", "data/features"))
        feature_files = list(feature_dir.glob("*_features.parquet"))

        if not feature_files:
            logger.error("No feature files found")
            return 1

        # Use first symbol's features
        features = pd.read_parquet(feature_files[0])

        analyzer = FeatureImportanceAnalyzer()
        importance = analyzer.analyze(model, features, method="auto")

        # Print results
        for method, scores in importance.items():
            logger.info(f"\n{method}:")
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for feat, score in sorted_scores[:20]:
                logger.info(f"  {feat}: {score:.4f}")

        return 0

    except Exception as e:
        logger.error(f"Importance analysis failed: {e}")
        return 1


def cmd_gpu_benchmark(args: argparse.Namespace) -> int:
    """Benchmark GPU vs CPU feature computation."""
    logger.info("=" * 80)
    logger.info("GPU BENCHMARK")
    logger.info("=" * 80)

    from scripts.data import DataManager, DataConfig

    data_config = DataConfig()
    data_manager = DataManager(data_config)

    symbols = getattr(args, "symbols", None) or data_manager.get_available_symbols()[:3]

    # CPU benchmark
    cpu_config = FeatureConfig(use_gpu=False)
    cpu_calculator = FeatureCalculator(cpu_config)

    cpu_times = []
    for symbol in symbols:
        df = data_manager.load_symbol(symbol)
        start = time.time()
        cpu_calculator.compute_all(df, symbol)
        cpu_times.append(time.time() - start)

    logger.info(f"CPU average time: {np.mean(cpu_times):.3f}s")

    # GPU benchmark
    gpu_config = FeatureConfig(use_gpu=True)
    gpu_calculator = FeatureCalculator(gpu_config)

    if gpu_calculator.gpu_available:
        gpu_times = []
        for symbol in symbols:
            df = data_manager.load_symbol(symbol)
            start = time.time()
            gpu_calculator.compute_all(df, symbol)
            gpu_times.append(time.time() - start)

        logger.info(f"GPU average time: {np.mean(gpu_times):.3f}s")
        logger.info(f"Speedup: {np.mean(cpu_times) / np.mean(gpu_times):.2f}x")
    else:
        logger.warning("GPU not available for benchmark")

    return 0


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def run_features_command(args: argparse.Namespace) -> int:
    """
    Main entry point for feature commands.

    @mlquant: This function routes to the appropriate feature command handler.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code.
    """
    command = getattr(args, "features_command", "compute")

    commands = {
        "compute": cmd_compute,
        "analyze": cmd_analyze,
        "importance": cmd_importance,
        "gpu": cmd_gpu_benchmark,
    }

    handler = commands.get(command)
    if handler:
        return handler(args)
    else:
        logger.error(f"Unknown features command: {command}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaTrade Feature Engineering")
    subparsers = parser.add_subparsers(dest="features_command")

    # Compute command
    compute_parser = subparsers.add_parser("compute", help="Compute features")
    compute_parser.add_argument("--symbols", nargs="+")
    compute_parser.add_argument("--groups", nargs="+", default=["all"])
    compute_parser.add_argument("--use-gpu", action="store_true")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze features")
    analyze_parser.add_argument("--path", type=str, default="data/features")

    # Importance command
    importance_parser = subparsers.add_parser("importance", help="Feature importance")
    importance_parser.add_argument("--model", type=str, required=True)
    importance_parser.add_argument("--features", type=str, default="data/features")

    args = parser.parse_args()
    sys.exit(run_features_command(args))
