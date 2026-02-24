"""
Alternative Data Framework.

P2-A Enhancement: Framework for integrating alternative data sources:
- Sentiment data (news, social media)
- Web traffic and app analytics
- Satellite/geolocation data
- Credit card transaction proxies
- Supply chain signals

Expected Impact: +10-20 bps from orthogonal alpha sources.

Author: AlphaTrade System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quant_trading_system.core.events import (
    Event,
    EventBus,
    EventPriority,
    EventType,
    get_event_bus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Alternative Data Types
# =============================================================================


class AltDataType(str, Enum):
    """Types of alternative data."""

    SENTIMENT = "sentiment"
    WEB_TRAFFIC = "web_traffic"
    APP_DOWNLOADS = "app_downloads"
    SOCIAL_MEDIA = "social_media"
    NEWS = "news"
    SATELLITE = "satellite"
    CREDIT_CARD = "credit_card"
    SUPPLY_CHAIN = "supply_chain"
    WEATHER = "weather"
    GOVERNMENT = "government"
    CUSTOM = "custom"


class SentimentScore(str, Enum):
    """Sentiment classification."""

    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class AltDataPoint:
    """Single alternative data observation.

    Note: value is float (not Decimal) because alternative data values are
    normalized scores (sentiment, confidence, percentage changes), not monetary
    values. For monetary values, use the core data_types module with Decimal.
    """

    symbol: str
    data_type: AltDataType
    timestamp: datetime
    value: float
    confidence: float = 1.0  # Data quality confidence [0, 1]
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """FIX: Validate alternative data values are within reasonable bounds."""
        # Confidence must be between 0 and 1
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

        # Values should be bounded (most alt data is normalized or percentage-based)
        # Allow range of -1000 to 1000 for flexibility (e.g., percentage changes)
        if not -1000.0 <= self.value <= 1000.0:
            logger.warning(
                f"AltDataPoint value {self.value} is outside typical bounds [-1000, 1000] "
                f"for {self.data_type.value}. Ensure this is not a monetary value."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "data_type": self.data_type.value,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class SentimentData:
    """Sentiment data observation."""

    symbol: str
    timestamp: datetime
    score: float  # -1 (very bearish) to +1 (very bullish)
    magnitude: float  # Strength of sentiment (0 to 1)
    volume: int  # Number of mentions/articles
    source: str  # news, twitter, reddit, etc.
    classification: SentimentScore = SentimentScore.NEUTRAL

    @classmethod
    def from_score(cls, symbol: str, score: float, **kwargs) -> "SentimentData":
        """Create from numeric score."""
        if score < -0.6:
            classification = SentimentScore.VERY_BEARISH
        elif score < -0.2:
            classification = SentimentScore.BEARISH
        elif score < 0.2:
            classification = SentimentScore.NEUTRAL
        elif score < 0.6:
            classification = SentimentScore.BULLISH
        else:
            classification = SentimentScore.VERY_BULLISH

        return cls(
            symbol=symbol,
            score=score,
            classification=classification,
            timestamp=kwargs.get("timestamp", datetime.now(timezone.utc)),
            magnitude=kwargs.get("magnitude", abs(score)),
            volume=kwargs.get("volume", 0),
            source=kwargs.get("source", "unknown"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "score": self.score,
            "magnitude": self.magnitude,
            "volume": self.volume,
            "source": self.source,
            "classification": self.classification.value,
        }


@dataclass
class WebTrafficData:
    """Web traffic/app analytics data."""

    symbol: str
    timestamp: datetime
    visits: int
    unique_visitors: int
    page_views: int
    bounce_rate: float
    avg_session_duration: float  # seconds
    yoy_growth: float  # Year-over-year growth
    source: str = "similarweb"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "visits": self.visits,
            "unique_visitors": self.unique_visitors,
            "page_views": self.page_views,
            "bounce_rate": self.bounce_rate,
            "avg_session_duration": self.avg_session_duration,
            "yoy_growth": self.yoy_growth,
            "source": self.source,
        }


class AltDataConfig(BaseModel):
    """Configuration for alternative data framework."""

    # Data freshness
    max_staleness_hours: int = Field(default=24, description="Max hours before data considered stale")

    # Sentiment settings
    sentiment_lookback_days: int = Field(default=7, description="Days of sentiment history")
    sentiment_decay: float = Field(default=0.9, description="Exponential decay for older sentiment")
    min_sentiment_volume: int = Field(default=5, description="Minimum mentions for valid signal")

    # Quality filters
    min_confidence: float = Field(default=0.5, description="Minimum confidence threshold")
    outlier_std_threshold: float = Field(default=3.0, description="Outlier detection threshold")

    # Caching
    cache_ttl_seconds: int = Field(default=300, description="Cache TTL")


# =============================================================================
# Alternative Data Providers (Abstract)
# =============================================================================


class AltDataProvider(ABC):
    """Abstract base class for alternative data providers."""

    def __init__(
        self,
        name: str,
        data_type: AltDataType,
        config: AltDataConfig | None = None,
    ):
        """Initialize provider.

        Args:
            name: Provider name
            data_type: Type of data provided
            config: Configuration
        """
        self.name = name
        self.data_type = data_type
        self.config = config or AltDataConfig()

        self._lock = threading.RLock()
        self._cache: dict[str, tuple[datetime, Any]] = {}
        self._connected = False

    @abstractmethod
    async def connect(self) -> None:
        """Connect to data source."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from data source."""
        pass

    @abstractmethod
    async def fetch_data(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[AltDataPoint]:
        """Fetch data for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of data points
        """
        pass

    def _get_cached(self, key: str) -> Any | None:
        """Get cached value if not stale."""
        with self._lock:
            if key in self._cache:
                cached_time, value = self._cache[key]
                if (datetime.now(timezone.utc) - cached_time).total_seconds() < self.config.cache_ttl_seconds:
                    return value
        return None

    def _set_cached(self, key: str, value: Any) -> None:
        """Set cached value."""
        with self._lock:
            self._cache[key] = (datetime.now(timezone.utc), value)


# =============================================================================
# Sentiment Provider Implementation
# =============================================================================


class SentimentProvider(AltDataProvider):
    """
    Sentiment data provider.

    Aggregates sentiment from multiple sources:
    - Financial news (Reuters, Bloomberg, etc.)
    - Social media (Twitter, StockTwits)
    - Reddit (wallstreetbets, investing)
    - Analyst reports
    """

    def __init__(
        self,
        config: AltDataConfig | None = None,
        api_key: str | None = None,
    ):
        """Initialize sentiment provider.

        Args:
            config: Configuration
            api_key: API key for sentiment service
        """
        super().__init__("SentimentProvider", AltDataType.SENTIMENT, config)
        self.api_key = api_key
        self._sentiment_history: dict[str, list[SentimentData]] = {}

    async def connect(self) -> None:
        """Connect to sentiment API."""
        # In production, connect to actual sentiment API (e.g., Refinitiv, Bloomberg)
        self._connected = True
        logger.info("Sentiment provider connected")

    async def disconnect(self) -> None:
        """Disconnect from sentiment API."""
        self._connected = False
        logger.info("Sentiment provider disconnected")

    async def fetch_data(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[AltDataPoint]:
        """Fetch sentiment data for a symbol."""
        # Check cache
        cache_key = f"sentiment_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # In production, this would call actual sentiment API
        # For now, return mock data
        data = self._generate_mock_sentiment(symbol, start_date, end_date)

        self._set_cached(cache_key, data)
        return data

    async def get_sentiment(self, symbol: str) -> SentimentData | None:
        """Get current aggregate sentiment for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Aggregated sentiment data
        """
        data_points = await self.fetch_data(symbol)
        if not data_points:
            return None

        # Aggregate with decay weighting
        now = datetime.now(timezone.utc)
        weighted_score = 0.0
        total_weight = 0.0
        total_volume = 0

        for dp in data_points:
            days_ago = (now - dp.timestamp).total_seconds() / 86400
            weight = self.config.sentiment_decay ** days_ago * dp.confidence

            weighted_score += dp.value * weight
            total_weight += weight
            total_volume += dp.metadata.get("volume", 1)

        if total_weight < 0.01:
            return None

        avg_score = weighted_score / total_weight

        return SentimentData.from_score(
            symbol=symbol,
            score=avg_score,
            timestamp=now,
            magnitude=min(abs(avg_score) * 1.5, 1.0),
            volume=total_volume,
            source="aggregated",
        )

    def _generate_mock_sentiment(
        self,
        symbol: str,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> list[AltDataPoint]:
        """Generate mock sentiment data for testing."""
        end = end_date or datetime.now(timezone.utc)
        start = start_date or (end - timedelta(days=self.config.sentiment_lookback_days))

        data_points = []
        current = start

        # Symbol-specific sentiment bias (mock)
        symbol_bias = {
            "AAPL": 0.2,
            "MSFT": 0.15,
            "TSLA": 0.3,
            "NVDA": 0.25,
            "META": -0.1,
        }.get(symbol, 0.0)

        while current <= end:
            # Generate random sentiment with symbol bias
            base_score = np.random.normal(symbol_bias, 0.3)
            score = np.clip(base_score, -1.0, 1.0)

            data_points.append(AltDataPoint(
                symbol=symbol,
                data_type=AltDataType.SENTIMENT,
                timestamp=current,
                value=score,
                confidence=np.random.uniform(0.6, 1.0),
                source="mock",
                metadata={
                    "volume": np.random.randint(10, 100),
                    "news_count": np.random.randint(1, 20),
                },
            ))

            current += timedelta(hours=np.random.randint(4, 12))

        return data_points


# =============================================================================
# Web Traffic Provider
# =============================================================================


class WebTrafficProvider(AltDataProvider):
    """
    Web traffic and app analytics provider.

    Tracks:
    - Website visits (SimilarWeb-style data)
    - App downloads and rankings
    - User engagement metrics
    """

    def __init__(
        self,
        config: AltDataConfig | None = None,
    ):
        super().__init__("WebTrafficProvider", AltDataType.WEB_TRAFFIC, config)
        self._company_domains: dict[str, str] = {
            "AAPL": "apple.com",
            "MSFT": "microsoft.com",
            "GOOGL": "google.com",
            "AMZN": "amazon.com",
            "META": "meta.com",
            "TSLA": "tesla.com",
            "NVDA": "nvidia.com",
            "NFLX": "netflix.com",
        }

    async def connect(self) -> None:
        """Connect to web traffic API."""
        self._connected = True
        logger.info("Web traffic provider connected")

    async def disconnect(self) -> None:
        """Disconnect from web traffic API."""
        self._connected = False

    async def fetch_data(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[AltDataPoint]:
        """Fetch web traffic data."""
        cache_key = f"webtraffic_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        data = self._generate_mock_traffic(symbol)
        self._set_cached(cache_key, data)
        return data

    async def get_traffic_signal(self, symbol: str) -> float | None:
        """Get web traffic signal (-1 to +1).

        Compares recent traffic to historical baseline.

        Args:
            symbol: Stock symbol

        Returns:
            Traffic signal or None
        """
        data = await self.fetch_data(symbol)
        if not data or len(data) < 2:
            return None

        # Compare most recent to average
        recent = data[-1].value
        baseline = np.mean([d.value for d in data[:-1]])

        if baseline == 0:
            return 0.0

        change_pct = (recent - baseline) / baseline

        # Convert to -1 to +1 scale
        return np.clip(change_pct / 0.5, -1.0, 1.0)

    def _generate_mock_traffic(self, symbol: str) -> list[AltDataPoint]:
        """Generate mock web traffic data."""
        data_points = []
        now = datetime.now(timezone.utc)

        # Base traffic levels (mock)
        base_traffic = {
            "AAPL": 50_000_000,
            "MSFT": 40_000_000,
            "GOOGL": 100_000_000,
            "AMZN": 80_000_000,
        }.get(symbol, 10_000_000)

        for days_ago in range(30, 0, -1):
            timestamp = now - timedelta(days=days_ago)
            # Random walk traffic
            traffic = base_traffic * np.random.uniform(0.8, 1.2)

            data_points.append(AltDataPoint(
                symbol=symbol,
                data_type=AltDataType.WEB_TRAFFIC,
                timestamp=timestamp,
                value=traffic,
                confidence=0.9,
                source="mock",
                metadata={
                    "unique_visitors": int(traffic * 0.6),
                    "bounce_rate": np.random.uniform(0.3, 0.6),
                },
            ))

        return data_points


# =============================================================================
# Satellite Data Provider
# =============================================================================


@dataclass
class SatelliteData:
    """Satellite imagery derived data."""

    symbol: str
    timestamp: datetime
    location_type: str  # retail_parking, factory, port, warehouse
    activity_index: float  # Normalized activity level (0-1)
    change_vs_baseline: float  # % change vs historical baseline
    confidence: float  # Detection confidence
    source: str = "satellite"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "location_type": self.location_type,
            "activity_index": self.activity_index,
            "change_vs_baseline": self.change_vs_baseline,
            "confidence": self.confidence,
            "source": self.source,
        }


class SatelliteProvider(AltDataProvider):
    """
    Satellite imagery data provider.

    Analyzes satellite imagery for:
    - Retail parking lot traffic (consumer spending proxy)
    - Factory activity levels (production proxy)
    - Port and shipping activity (supply chain proxy)
    - Construction progress (capex tracking)

    Data sourced from providers like:
    - Planet Labs, Orbital Insight, RS Metrics, Descartes Labs
    """

    def __init__(
        self,
        config: AltDataConfig | None = None,
        api_key: str | None = None,
    ):
        """Initialize satellite provider.

        Args:
            config: Configuration
            api_key: API key for satellite data service
        """
        super().__init__("SatelliteProvider", AltDataType.SATELLITE, config)
        self.api_key = api_key

        # Company to location mappings (mock data)
        self._company_locations: dict[str, list[dict[str, Any]]] = {
            "WMT": [
                {"type": "retail_parking", "name": "Walmart Supercenter", "baseline_cars": 500},
                {"type": "warehouse", "name": "Distribution Center", "baseline_trucks": 50},
            ],
            "TGT": [
                {"type": "retail_parking", "name": "Target Store", "baseline_cars": 300},
            ],
            "AMZN": [
                {"type": "warehouse", "name": "Fulfillment Center", "baseline_trucks": 200},
                {"type": "warehouse", "name": "Sortation Center", "baseline_trucks": 100},
            ],
            "TSLA": [
                {"type": "factory", "name": "Gigafactory", "baseline_vehicles": 1000},
            ],
            "HD": [
                {"type": "retail_parking", "name": "Home Depot Store", "baseline_cars": 400},
            ],
        }

    async def connect(self) -> None:
        """Connect to satellite data API."""
        # In production, connect to actual satellite data provider
        self._connected = True
        logger.info("Satellite data provider connected")

    async def disconnect(self) -> None:
        """Disconnect from satellite data API."""
        self._connected = False
        logger.info("Satellite data provider disconnected")

    async def fetch_data(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[AltDataPoint]:
        """Fetch satellite data for a symbol."""
        cache_key = f"satellite_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        data = self._generate_mock_satellite_data(symbol, start_date, end_date)
        self._set_cached(cache_key, data)
        return data

    async def get_activity_signal(self, symbol: str) -> float | None:
        """Get satellite-derived activity signal.

        Returns activity signal normalized to (-1, +1):
        - Positive: Higher than baseline activity
        - Negative: Lower than baseline activity

        Args:
            symbol: Stock symbol

        Returns:
            Activity signal or None
        """
        data = await self.fetch_data(symbol)
        if not data:
            return None

        # Get most recent data points
        recent = data[-min(5, len(data)):]

        # Average change vs baseline
        avg_change = np.mean([d.metadata.get("change_vs_baseline", 0) for d in recent])

        # Normalize to -1 to +1 (assume +/- 30% is full scale)
        return np.clip(avg_change / 0.3, -1.0, 1.0)

    async def get_parking_lot_data(self, symbol: str) -> SatelliteData | None:
        """Get parking lot traffic data for retail companies.

        Args:
            symbol: Stock symbol

        Returns:
            Latest parking lot data or None
        """
        data = await self.fetch_data(symbol)
        if not data:
            return None

        # Filter for parking lot data
        parking_data = [d for d in data if d.metadata.get("location_type") == "retail_parking"]
        if not parking_data:
            return None

        latest = parking_data[-1]
        return SatelliteData(
            symbol=symbol,
            timestamp=latest.timestamp,
            location_type="retail_parking",
            activity_index=latest.value,
            change_vs_baseline=latest.metadata.get("change_vs_baseline", 0),
            confidence=latest.confidence,
        )

    def _generate_mock_satellite_data(
        self,
        symbol: str,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> list[AltDataPoint]:
        """Generate mock satellite data for testing."""
        end = end_date or datetime.now(timezone.utc)
        start = start_date or (end - timedelta(days=30))

        locations = self._company_locations.get(symbol, [
            {"type": "retail_parking", "name": "Default Location", "baseline_cars": 200}
        ])

        data_points = []
        current = start

        while current <= end:
            for location in locations:
                # Generate random activity with some trend
                base_activity = np.random.uniform(0.7, 1.3)
                # Add day-of-week effect
                day_effect = 1.2 if current.weekday() in [4, 5] else 1.0  # Weekend boost
                activity = base_activity * day_effect

                change_vs_baseline = (activity - 1.0)

                data_points.append(AltDataPoint(
                    symbol=symbol,
                    data_type=AltDataType.SATELLITE,
                    timestamp=current,
                    value=min(activity, 2.0),  # Cap at 2x baseline
                    confidence=np.random.uniform(0.8, 0.98),
                    source="mock_satellite",
                    metadata={
                        "location_type": location["type"],
                        "location_name": location["name"],
                        "change_vs_baseline": change_vs_baseline,
                        "image_quality": np.random.choice(["high", "medium"]),
                    },
                ))

            current += timedelta(days=1)

        return data_points


# =============================================================================
# Credit Card Data Provider
# =============================================================================


@dataclass
class CreditCardData:
    """Credit card transaction data."""

    symbol: str
    timestamp: datetime
    transaction_count: int
    transaction_volume: float  # USD
    avg_ticket_size: float
    yoy_growth: float  # Year-over-year growth %
    wow_growth: float  # Week-over-week growth %
    market_share: float  # Share of category spending
    source: str = "credit_card"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "transaction_count": self.transaction_count,
            "transaction_volume": self.transaction_volume,
            "avg_ticket_size": self.avg_ticket_size,
            "yoy_growth": self.yoy_growth,
            "wow_growth": self.wow_growth,
            "market_share": self.market_share,
            "source": self.source,
        }


class CreditCardProvider(AltDataProvider):
    """
    Credit card transaction data provider.

    Provides spending signals from:
    - Transaction counts and volumes
    - Average ticket sizes
    - Year-over-year and week-over-week growth
    - Market share changes

    Data sourced from providers like:
    - Second Measure, Earnest Research, Bloomberg, M Science
    """

    def __init__(
        self,
        config: AltDataConfig | None = None,
        api_key: str | None = None,
    ):
        """Initialize credit card provider.

        Args:
            config: Configuration
            api_key: API key for credit card data service
        """
        super().__init__("CreditCardProvider", AltDataType.CREDIT_CARD, config)
        self.api_key = api_key

        # Company spending baselines (mock data)
        self._company_baselines: dict[str, dict[str, Any]] = {
            "AMZN": {"category": "e_commerce", "weekly_volume_mm": 5000, "avg_ticket": 85},
            "WMT": {"category": "retail", "weekly_volume_mm": 8000, "avg_ticket": 65},
            "COST": {"category": "retail", "weekly_volume_mm": 2500, "avg_ticket": 150},
            "TGT": {"category": "retail", "weekly_volume_mm": 1500, "avg_ticket": 55},
            "HD": {"category": "home_improvement", "weekly_volume_mm": 2000, "avg_ticket": 120},
            "LOW": {"category": "home_improvement", "weekly_volume_mm": 1500, "avg_ticket": 100},
            "SBUX": {"category": "restaurant", "weekly_volume_mm": 800, "avg_ticket": 8},
            "MCD": {"category": "restaurant", "weekly_volume_mm": 1200, "avg_ticket": 12},
            "CMG": {"category": "restaurant", "weekly_volume_mm": 400, "avg_ticket": 18},
            "DPZ": {"category": "restaurant", "weekly_volume_mm": 350, "avg_ticket": 25},
        }

    async def connect(self) -> None:
        """Connect to credit card data API."""
        self._connected = True
        logger.info("Credit card data provider connected")

    async def disconnect(self) -> None:
        """Disconnect from credit card data API."""
        self._connected = False
        logger.info("Credit card data provider disconnected")

    async def fetch_data(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[AltDataPoint]:
        """Fetch credit card data for a symbol."""
        cache_key = f"creditcard_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        data = self._generate_mock_cc_data(symbol, start_date, end_date)
        self._set_cached(cache_key, data)
        return data

    async def get_spending_signal(self, symbol: str) -> float | None:
        """Get spending signal normalized to (-1, +1).

        Combines multiple metrics:
        - YoY growth (40% weight)
        - WoW growth momentum (30% weight)
        - Market share change (30% weight)

        Args:
            symbol: Stock symbol

        Returns:
            Spending signal or None
        """
        data = await self.fetch_data(symbol)
        if not data or len(data) < 2:
            return None

        # Get recent data
        recent = data[-1]

        yoy_growth = recent.metadata.get("yoy_growth", 0)
        wow_growth = recent.metadata.get("wow_growth", 0)
        market_share_change = recent.metadata.get("market_share_change", 0)

        # Normalize each component
        # YoY: 10% growth = 0.5 signal
        yoy_signal = np.clip(yoy_growth / 0.2, -1.0, 1.0)

        # WoW: 5% = 0.5 signal
        wow_signal = np.clip(wow_growth / 0.1, -1.0, 1.0)

        # Market share change: 1% = 0.5 signal
        share_signal = np.clip(market_share_change / 0.02, -1.0, 1.0)

        # Weighted combination
        composite = 0.4 * yoy_signal + 0.3 * wow_signal + 0.3 * share_signal

        return np.clip(composite, -1.0, 1.0)

    async def get_transaction_data(self, symbol: str) -> CreditCardData | None:
        """Get latest credit card transaction data.

        Args:
            symbol: Stock symbol

        Returns:
            Credit card data or None
        """
        data = await self.fetch_data(symbol)
        if not data:
            return None

        latest = data[-1]
        return CreditCardData(
            symbol=symbol,
            timestamp=latest.timestamp,
            transaction_count=latest.metadata.get("transaction_count", 0),
            transaction_volume=latest.value,
            avg_ticket_size=latest.metadata.get("avg_ticket", 0),
            yoy_growth=latest.metadata.get("yoy_growth", 0),
            wow_growth=latest.metadata.get("wow_growth", 0),
            market_share=latest.metadata.get("market_share", 0),
        )

    def _generate_mock_cc_data(
        self,
        symbol: str,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> list[AltDataPoint]:
        """Generate mock credit card data for testing."""
        end = end_date or datetime.now(timezone.utc)
        start = start_date or (end - timedelta(weeks=12))

        baseline = self._company_baselines.get(symbol, {
            "category": "other",
            "weekly_volume_mm": 500,
            "avg_ticket": 50,
        })

        data_points = []
        current = start

        # Generate weekly data points
        week_num = 0
        cumulative_growth = 0.0

        while current <= end:
            # Random walk for spending trends
            weekly_change = np.random.normal(0.01, 0.03)  # 1% avg growth, 3% volatility
            cumulative_growth += weekly_change

            # Seasonal effects
            month = current.month
            if month in [11, 12]:  # Holiday season
                seasonal_factor = 1.3
            elif month in [1, 2]:  # Post-holiday
                seasonal_factor = 0.9
            else:
                seasonal_factor = 1.0

            volume = baseline["weekly_volume_mm"] * (1 + cumulative_growth) * seasonal_factor
            volume *= np.random.uniform(0.95, 1.05)  # Add noise

            # Calculate metrics
            yoy_growth = cumulative_growth + np.random.uniform(-0.02, 0.02)
            wow_growth = weekly_change
            market_share = 0.15 + np.random.uniform(-0.01, 0.01)

            data_points.append(AltDataPoint(
                symbol=symbol,
                data_type=AltDataType.CREDIT_CARD,
                timestamp=current,
                value=volume,
                confidence=0.95,
                source="mock_cc_panel",
                metadata={
                    "category": baseline["category"],
                    "transaction_count": int(volume * 1_000_000 / baseline["avg_ticket"]),
                    "avg_ticket": baseline["avg_ticket"] * np.random.uniform(0.95, 1.05),
                    "yoy_growth": yoy_growth,
                    "wow_growth": wow_growth,
                    "market_share": market_share,
                    "market_share_change": wow_growth * 0.1,
                    "panel_size": 5_000_000,  # Mock panel size
                },
            ))

            current += timedelta(weeks=1)
            week_num += 1

        return data_points


# =============================================================================
# Supply Chain Data Provider
# =============================================================================


@dataclass
class SupplyChainData:
    """Supply chain and logistics data."""

    symbol: str
    timestamp: datetime
    inventory_days: float  # Days of inventory on hand
    lead_time_days: float  # Supplier lead time
    shipping_volume: int  # Container/truck shipments
    shipping_cost_index: float  # Relative shipping cost
    supplier_health: float  # Supplier financial health (0-1)
    disruption_risk: float  # Supply disruption risk (0-1)
    source: str = "supply_chain"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "inventory_days": self.inventory_days,
            "lead_time_days": self.lead_time_days,
            "shipping_volume": self.shipping_volume,
            "shipping_cost_index": self.shipping_cost_index,
            "supplier_health": self.supplier_health,
            "disruption_risk": self.disruption_risk,
            "source": self.source,
        }


class SupplyChainProvider(AltDataProvider):
    """
    Supply chain and logistics data provider.

    Tracks:
    - Shipping and freight activity
    - Inventory levels and turns
    - Supplier health metrics
    - Port congestion and delays
    - Component availability

    Data sourced from:
    - FreightWaves, project44, FourKites
    - AIS ship tracking data
    - Customs and port authority data
    """

    def __init__(
        self,
        config: AltDataConfig | None = None,
        api_key: str | None = None,
    ):
        """Initialize supply chain provider.

        Args:
            config: Configuration
            api_key: API key for supply chain data service
        """
        super().__init__("SupplyChainProvider", AltDataType.SUPPLY_CHAIN, config)
        self.api_key = api_key

        # Company supply chain profiles (mock)
        self._company_profiles: dict[str, dict[str, Any]] = {
            "AAPL": {
                "industry": "consumer_electronics",
                "avg_inventory_days": 9,
                "primary_suppliers": ["TSM", "HON", "QRVO"],
                "key_ports": ["Shanghai", "Long Beach"],
            },
            "TSLA": {
                "industry": "automotive",
                "avg_inventory_days": 4,
                "primary_suppliers": ["PCRFY", "ALB", "LG Chem"],
                "key_ports": ["Shanghai", "Fremont"],
            },
            "NKE": {
                "industry": "apparel",
                "avg_inventory_days": 85,
                "primary_suppliers": ["Vietnam Mfg", "China Mfg"],
                "key_ports": ["Los Angeles", "Rotterdam"],
            },
            "WMT": {
                "industry": "retail",
                "avg_inventory_days": 45,
                "primary_suppliers": ["Various"],
                "key_ports": ["Long Beach", "Savannah", "Houston"],
            },
            "HD": {
                "industry": "retail",
                "avg_inventory_days": 80,
                "primary_suppliers": ["Various"],
                "key_ports": ["Long Beach", "Newark"],
            },
        }

    async def connect(self) -> None:
        """Connect to supply chain data API."""
        self._connected = True
        logger.info("Supply chain data provider connected")

    async def disconnect(self) -> None:
        """Disconnect from supply chain data API."""
        self._connected = False
        logger.info("Supply chain data provider disconnected")

    async def fetch_data(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[AltDataPoint]:
        """Fetch supply chain data for a symbol."""
        cache_key = f"supplychain_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        data = self._generate_mock_sc_data(symbol, start_date, end_date)
        self._set_cached(cache_key, data)
        return data

    async def get_supply_chain_signal(self, symbol: str) -> float | None:
        """Get supply chain health signal normalized to (-1, +1).

        Positive: Good supply chain health, low risk
        Negative: Supply chain stress, elevated risk

        Considers:
        - Inventory adequacy
        - Shipping activity trends
        - Lead time changes
        - Disruption risk

        Args:
            symbol: Stock symbol

        Returns:
            Supply chain signal or None
        """
        data = await self.fetch_data(symbol)
        if not data or len(data) < 2:
            return None

        recent = data[-5:] if len(data) >= 5 else data

        # Average metrics from recent data
        avg_inventory = np.mean([d.metadata.get("inventory_index", 1.0) for d in recent])
        avg_shipping = np.mean([d.value for d in recent])
        avg_disruption = np.mean([d.metadata.get("disruption_risk", 0.3) for d in recent])

        # Calculate signal components
        # Inventory: 1.0 is baseline, <0.8 is low (negative), >1.2 is high (slightly negative due to carrying cost)
        if avg_inventory < 0.8:
            inventory_signal = (avg_inventory - 0.8) / 0.4  # Negative when low
        elif avg_inventory > 1.2:
            inventory_signal = -(avg_inventory - 1.2) / 0.4  # Slightly negative when too high
        else:
            inventory_signal = 0.2  # Goldilocks zone

        # Shipping activity: positive when above baseline
        shipping_signal = np.clip((avg_shipping - 1.0) / 0.3, -1.0, 1.0)

        # Disruption risk: negative when high
        disruption_signal = -2 * (avg_disruption - 0.3)  # 0.3 is baseline

        # Weighted combination
        composite = 0.3 * inventory_signal + 0.4 * shipping_signal + 0.3 * disruption_signal

        return np.clip(composite, -1.0, 1.0)

    async def get_disruption_alert(self, symbol: str) -> dict[str, Any] | None:
        """Get supply chain disruption alert if risk is elevated.

        Args:
            symbol: Stock symbol

        Returns:
            Alert details or None if no alert
        """
        data = await self.fetch_data(symbol)
        if not data:
            return None

        latest = data[-1]
        disruption_risk = latest.metadata.get("disruption_risk", 0)

        if disruption_risk > 0.6:
            return {
                "symbol": symbol,
                "severity": "high" if disruption_risk > 0.8 else "medium",
                "disruption_risk": disruption_risk,
                "timestamp": latest.timestamp.isoformat(),
                "factors": latest.metadata.get("risk_factors", []),
                "affected_suppliers": latest.metadata.get("affected_suppliers", []),
            }

        return None

    async def get_logistics_data(self, symbol: str) -> SupplyChainData | None:
        """Get latest supply chain logistics data.

        Args:
            symbol: Stock symbol

        Returns:
            Supply chain data or None
        """
        data = await self.fetch_data(symbol)
        if not data:
            return None

        latest = data[-1]
        profile = self._company_profiles.get(symbol, {})

        return SupplyChainData(
            symbol=symbol,
            timestamp=latest.timestamp,
            inventory_days=latest.metadata.get("inventory_days", profile.get("avg_inventory_days", 30)),
            lead_time_days=latest.metadata.get("lead_time_days", 14),
            shipping_volume=int(latest.value * 1000),
            shipping_cost_index=latest.metadata.get("shipping_cost_index", 1.0),
            supplier_health=latest.metadata.get("supplier_health", 0.8),
            disruption_risk=latest.metadata.get("disruption_risk", 0.2),
        )

    def _generate_mock_sc_data(
        self,
        symbol: str,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> list[AltDataPoint]:
        """Generate mock supply chain data for testing."""
        end = end_date or datetime.now(timezone.utc)
        start = start_date or (end - timedelta(weeks=12))

        profile = self._company_profiles.get(symbol, {
            "industry": "general",
            "avg_inventory_days": 30,
            "primary_suppliers": [],
            "key_ports": [],
        })

        data_points = []
        current = start

        # Initialize baseline values
        inventory_days = profile["avg_inventory_days"]
        base_shipping = 1.0
        disruption_risk = 0.2

        while current <= end:
            # Random walk for supply chain metrics
            inventory_days += np.random.normal(0, 2)
            inventory_days = max(1, inventory_days)  # Can't have negative inventory

            # Shipping activity with some correlation to economic conditions
            shipping_change = np.random.normal(0.01, 0.05)
            base_shipping = np.clip(base_shipping * (1 + shipping_change), 0.5, 2.0)

            # Disruption risk with occasional spikes
            if np.random.random() < 0.05:  # 5% chance of disruption event
                disruption_risk = min(disruption_risk + 0.3, 1.0)
            else:
                disruption_risk = max(0.1, disruption_risk * 0.95)  # Decay back to normal

            # Lead time correlates with disruption risk
            lead_time = 14 * (1 + disruption_risk * 2)

            risk_factors = []
            if disruption_risk > 0.5:
                risk_factors = np.random.choice(
                    ["port_congestion", "supplier_shutdown", "shipping_delays",
                     "component_shortage", "weather", "labor_dispute"],
                    size=min(3, int(disruption_risk * 5)),
                    replace=False
                ).tolist()

            data_points.append(AltDataPoint(
                symbol=symbol,
                data_type=AltDataType.SUPPLY_CHAIN,
                timestamp=current,
                value=base_shipping,  # Shipping activity index
                confidence=0.9,
                source="mock_sc_tracker",
                metadata={
                    "industry": profile["industry"],
                    "inventory_days": inventory_days,
                    "inventory_index": inventory_days / profile["avg_inventory_days"],
                    "lead_time_days": lead_time,
                    "shipping_cost_index": 1.0 + disruption_risk * 0.5,
                    "supplier_health": 1.0 - disruption_risk * 0.3,
                    "disruption_risk": disruption_risk,
                    "risk_factors": risk_factors,
                    "key_ports": profile.get("key_ports", []),
                    "affected_suppliers": [] if disruption_risk < 0.6 else profile.get("primary_suppliers", [])[:2],
                },
            ))

            current += timedelta(days=1)

        return data_points


# =============================================================================
# Alternative Data Aggregator
# =============================================================================


class AltDataAggregator:
    """
    P2-A Enhancement: Aggregates multiple alternative data sources.

    Combines signals from:
    - Sentiment providers
    - Web traffic providers
    - Custom data sources

    Outputs normalized signals for use in alpha models.
    """

    def __init__(
        self,
        config: AltDataConfig | None = None,
        event_bus: EventBus | None = None,
    ):
        """Initialize aggregator.

        Args:
            config: Configuration
            event_bus: Event bus for publishing signals
        """
        self.config = config or AltDataConfig()
        self.event_bus = event_bus or get_event_bus()

        self._providers: dict[AltDataType, AltDataProvider] = {}
        self._lock = threading.RLock()
        self._signal_history: dict[str, list[tuple[datetime, float]]] = {}

    def register_provider(self, provider: AltDataProvider) -> None:
        """Register a data provider.

        Args:
            provider: Alternative data provider
        """
        with self._lock:
            self._providers[provider.data_type] = provider
            logger.info(f"Registered alt data provider: {provider.name}")

    async def connect_all(self) -> None:
        """Connect all registered providers."""
        for provider in self._providers.values():
            try:
                await provider.connect()
            except Exception as e:
                logger.error(f"Failed to connect {provider.name}: {e}")

    async def disconnect_all(self) -> None:
        """Disconnect all providers."""
        for provider in self._providers.values():
            try:
                await provider.disconnect()
            except Exception as e:
                logger.error(f"Failed to disconnect {provider.name}: {e}")

    async def get_composite_signal(
        self,
        symbol: str,
        weights: dict[AltDataType, float] | None = None,
    ) -> float | None:
        """Get composite alternative data signal.

        Combines all available signals with optional weighting.

        Args:
            symbol: Stock symbol
            weights: Optional weights per data type

        Returns:
            Composite signal (-1 to +1) or None
        """
        signals: dict[AltDataType, float] = {}

        # Collect signals from each provider
        for data_type, provider in self._providers.items():
            try:
                if data_type == AltDataType.SENTIMENT:
                    sentiment = await provider.get_sentiment(symbol)
                    if sentiment:
                        signals[data_type] = sentiment.score

                elif data_type == AltDataType.WEB_TRAFFIC:
                    traffic_signal = await provider.get_traffic_signal(symbol)
                    if traffic_signal is not None:
                        signals[data_type] = traffic_signal

                elif data_type == AltDataType.SATELLITE:
                    # Get satellite activity signal
                    activity_signal = await provider.get_activity_signal(symbol)
                    if activity_signal is not None:
                        signals[data_type] = activity_signal

                elif data_type == AltDataType.CREDIT_CARD:
                    # Get credit card spending signal
                    spending_signal = await provider.get_spending_signal(symbol)
                    if spending_signal is not None:
                        signals[data_type] = spending_signal

                elif data_type == AltDataType.SUPPLY_CHAIN:
                    # Get supply chain health signal
                    sc_signal = await provider.get_supply_chain_signal(symbol)
                    if sc_signal is not None:
                        signals[data_type] = sc_signal

                else:
                    # Generic signal extraction
                    data = await provider.fetch_data(symbol)
                    if data:
                        signals[data_type] = data[-1].value

            except Exception as e:
                logger.warning(f"Failed to get {data_type.value} signal for {symbol}: {e}")

        if not signals:
            return None

        # Apply weights
        if weights is None:
            weights = {dt: 1.0 for dt in signals.keys()}

        weighted_sum = 0.0
        total_weight = 0.0

        for data_type, signal in signals.items():
            weight = weights.get(data_type, 1.0)
            weighted_sum += signal * weight
            total_weight += weight

        if total_weight == 0:
            return None

        composite = weighted_sum / total_weight

        # Store in history
        with self._lock:
            if symbol not in self._signal_history:
                self._signal_history[symbol] = []
            self._signal_history[symbol].append((datetime.now(timezone.utc), composite))

            # Keep bounded
            if len(self._signal_history[symbol]) > 1000:
                self._signal_history[symbol] = self._signal_history[symbol][-1000:]

        return composite

    async def get_all_signals(
        self,
        symbols: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Get signals for multiple symbols.

        Args:
            symbols: List of symbols

        Returns:
            Dictionary mapping symbol to signal data
        """
        results = {}

        for symbol in symbols:
            signals = {}

            for data_type, provider in self._providers.items():
                try:
                    data = await provider.fetch_data(symbol)
                    if data:
                        latest = data[-1]
                        signals[data_type.value] = {
                            "value": latest.value,
                            "confidence": latest.confidence,
                            "timestamp": latest.timestamp.isoformat(),
                        }
                except Exception as e:
                    logger.warning(f"Failed to get {data_type.value} for {symbol}: {e}")

            composite = await self.get_composite_signal(symbol)

            results[symbol] = {
                "signals": signals,
                "composite": composite,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        return results

    def get_signal_history(
        self,
        symbol: str,
        limit: int = 100,
    ) -> list[tuple[datetime, float]]:
        """Get signal history for a symbol.

        Args:
            symbol: Stock symbol
            limit: Maximum history entries

        Returns:
            List of (timestamp, signal) tuples
        """
        with self._lock:
            history = self._signal_history.get(symbol, [])
            return history[-limit:]


# =============================================================================
# Alternative Data Feature Generator
# =============================================================================


class AltDataFeatureGenerator:
    """
    Generates features from alternative data for ML models.

    Features include:
    - Raw signals
    - Signal momentum
    - Signal volatility
    - Cross-sectional ranks
    """

    def __init__(
        self,
        aggregator: AltDataAggregator,
        lookback_periods: int = 20,
    ):
        """Initialize feature generator.

        Args:
            aggregator: Alternative data aggregator
            lookback_periods: Lookback for derived features
        """
        self.aggregator = aggregator
        self.lookback_periods = lookback_periods

    async def generate_features(
        self,
        symbols: list[str],
    ) -> pd.DataFrame:
        """Generate alternative data features.

        Args:
            symbols: List of symbols

        Returns:
            DataFrame with features
        """
        all_signals = await self.aggregator.get_all_signals(symbols)

        features = []

        for symbol in symbols:
            symbol_data = all_signals.get(symbol, {})
            composite = symbol_data.get("composite")

            # Get history for momentum/volatility
            history = self.aggregator.get_signal_history(symbol)

            feature_row = {
                "symbol": symbol,
                "alt_composite": composite if composite is not None else 0.0,
            }

            # Individual signal features
            for data_type in AltDataType:
                signal_data = symbol_data.get("signals", {}).get(data_type.value)
                if signal_data:
                    feature_row[f"alt_{data_type.value}"] = signal_data.get("value", 0.0)
                    feature_row[f"alt_{data_type.value}_conf"] = signal_data.get("confidence", 0.0)

            # Derived features from history
            if len(history) >= 2:
                signals = [s for _, s in history[-self.lookback_periods:]]

                # Momentum (recent vs older)
                if len(signals) >= 5:
                    recent_avg = np.mean(signals[-3:])
                    older_avg = np.mean(signals[:-3])
                    feature_row["alt_momentum"] = recent_avg - older_avg

                # Volatility
                if len(signals) >= 3:
                    feature_row["alt_volatility"] = np.std(signals)

                # Trend (linear regression slope)
                if len(signals) >= 5:
                    x = np.arange(len(signals))
                    slope = np.polyfit(x, signals, 1)[0]
                    feature_row["alt_trend"] = slope

            features.append(feature_row)

        return pd.DataFrame(features)


# =============================================================================
# Factory Functions
# =============================================================================


def create_alt_data_aggregator(
    include_sentiment: bool = True,
    include_web_traffic: bool = True,
    include_satellite: bool = False,
    include_credit_card: bool = False,
    include_supply_chain: bool = False,
    config: AltDataConfig | None = None,
    event_bus: EventBus | None = None,
    satellite_api_key: str | None = None,
    credit_card_api_key: str | None = None,
    supply_chain_api_key: str | None = None,
) -> AltDataAggregator:
    """Factory function to create alternative data aggregator.

    Args:
        include_sentiment: Include sentiment provider
        include_web_traffic: Include web traffic provider
        include_satellite: Include satellite imagery provider
        include_credit_card: Include credit card transaction provider
        include_supply_chain: Include supply chain data provider
        config: Configuration
        event_bus: Event bus
        satellite_api_key: API key for satellite data
        credit_card_api_key: API key for credit card data
        supply_chain_api_key: API key for supply chain data

    Returns:
        Configured AltDataAggregator
    """
    aggregator = AltDataAggregator(config=config, event_bus=event_bus)

    if include_sentiment:
        aggregator.register_provider(SentimentProvider(config))

    if include_web_traffic:
        aggregator.register_provider(WebTrafficProvider(config))

    if include_satellite:
        aggregator.register_provider(SatelliteProvider(config, api_key=satellite_api_key))

    if include_credit_card:
        aggregator.register_provider(CreditCardProvider(config, api_key=credit_card_api_key))

    if include_supply_chain:
        aggregator.register_provider(SupplyChainProvider(config, api_key=supply_chain_api_key))

    return aggregator


async def start_alt_data_aggregator(
    **kwargs: Any,
) -> AltDataAggregator:
    """Create and connect alternative data aggregator.

    Returns:
        Connected aggregator
    """
    aggregator = create_alt_data_aggregator(**kwargs)
    await aggregator.connect_all()
    return aggregator
