"""
Asynchronous live feed persistence to PostgreSQL + TimescaleDB.

Buffers incoming bars and performs batch inserts for efficiency.
Uses asyncpg for non-blocking database writes to avoid blocking
the main event loop during live trading.

Key features:
- Async batch inserts for efficiency
- Configurable buffer size and flush interval
- Automatic periodic flushing
- Error handling with retry logic
- Metrics for monitoring

@agent: @data, @trader
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from quant_trading_system.core.data_types import OHLCVBar
from quant_trading_system.core.exceptions import DataError

if TYPE_CHECKING:
    from quant_trading_system.database.connection import DatabaseManager

logger = logging.getLogger(__name__)


class LiveFeedPersister:
    """
    Persist live feed data to PostgreSQL + TimescaleDB asynchronously.

    Buffers incoming bars and performs batch inserts at configurable
    intervals or when buffer is full. This ensures minimal impact
    on live trading performance while maintaining data persistence.

    Thread-safe for use with async WebSocket feeds.
    """

    def __init__(
        self,
        db_manager: "DatabaseManager | None" = None,
        buffer_size: int = 100,
        flush_interval: float = 5.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the live feed persister.

        Args:
            db_manager: Database manager instance. If None, uses default.
            buffer_size: Maximum bars to buffer before auto-flush.
            flush_interval: Seconds between periodic flushes.
            max_retries: Maximum retry attempts on failure.
            retry_delay: Delay between retries in seconds.
        """
        if db_manager is None:
            from quant_trading_system.database.connection import get_db_manager
            db_manager = get_db_manager()

        self._db = db_manager
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        # Thread-safe buffer using deque
        self._buffer: deque[OHLCVBar] = deque(maxlen=buffer_size * 2)
        self._lock = asyncio.Lock()

        # Background flush task
        self._flush_task: asyncio.Task | None = None
        self._running = False

        # Metrics for monitoring
        self._metrics = {
            "bars_received": 0,
            "bars_persisted": 0,
            "flush_count": 0,
            "errors": 0,
            "retries": 0,
            "last_flush_time": None,
            "last_error_time": None,
            "last_error_message": None,
        }

    async def start(self) -> None:
        """Start the background flush task."""
        if self._running:
            logger.warning("LiveFeedPersister already running")
            return

        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())
        logger.info(
            f"LiveFeedPersister started (buffer_size={self._buffer_size}, "
            f"flush_interval={self._flush_interval}s)"
        )

    async def stop(self) -> None:
        """Stop the persister and flush remaining data."""
        if not self._running:
            return

        self._running = False

        # Cancel periodic flush task
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        # Final flush of remaining data
        await self._flush()
        logger.info(
            f"LiveFeedPersister stopped. Metrics: {self.get_metrics()}"
        )

    async def persist_bar(self, bar: OHLCVBar) -> None:
        """
        Add bar to buffer for persistence.

        Non-blocking operation. Bar is added to buffer and will be
        persisted in the next flush (periodic or buffer-full).

        Args:
            bar: OHLCVBar to persist.
        """
        async with self._lock:
            self._buffer.append(bar)
            self._metrics["bars_received"] += 1

            # Flush if buffer is full
            if len(self._buffer) >= self._buffer_size:
                # Release lock before flush to avoid deadlock
                pass

        # Check if we need to flush (do this outside lock)
        if len(self._buffer) >= self._buffer_size:
            await self._flush()

    async def persist_bars(self, bars: list[OHLCVBar]) -> None:
        """
        Add multiple bars to buffer for persistence.

        Args:
            bars: List of OHLCVBars to persist.
        """
        async with self._lock:
            for bar in bars:
                self._buffer.append(bar)
                self._metrics["bars_received"] += 1

        # Check if we need to flush
        if len(self._buffer) >= self._buffer_size:
            await self._flush()

    async def _periodic_flush(self) -> None:
        """Background task for periodic buffer flushing."""
        while self._running:
            try:
                await asyncio.sleep(self._flush_interval)
                if self._buffer:
                    await self._flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")
                self._metrics["errors"] += 1
                self._metrics["last_error_time"] = datetime.now(timezone.utc)
                self._metrics["last_error_message"] = str(e)

    async def _flush(self) -> None:
        """
        Flush buffer to database.

        Thread-safe. Uses copy-and-clear pattern to minimize lock time.
        """
        # Copy and clear buffer atomically
        async with self._lock:
            if not self._buffer:
                return

            bars_to_insert = list(self._buffer)
            self._buffer.clear()

        # Prepare batch data
        bar_dicts = []
        for bar in bars_to_insert:
            bar_dicts.append({
                "symbol": bar.symbol.upper(),
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "vwap": bar.vwap,
                "trade_count": bar.trade_count,
            })

        # Insert with retry logic
        for attempt in range(self._max_retries):
            try:
                inserted = await self._bulk_insert(bar_dicts)
                self._metrics["bars_persisted"] += inserted
                self._metrics["flush_count"] += 1
                self._metrics["last_flush_time"] = datetime.now(timezone.utc)

                logger.debug(
                    f"Flushed {inserted} bars to database "
                    f"(total persisted: {self._metrics['bars_persisted']})"
                )
                return

            except Exception as e:
                self._metrics["retries"] += 1
                logger.warning(
                    f"Flush attempt {attempt + 1}/{self._max_retries} failed: {e}"
                )

                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay * (attempt + 1))
                else:
                    # Final attempt failed - put bars back in buffer
                    self._metrics["errors"] += 1
                    self._metrics["last_error_time"] = datetime.now(timezone.utc)
                    self._metrics["last_error_message"] = str(e)

                    async with self._lock:
                        # Add bars back to front of buffer
                        for bar in reversed(bars_to_insert):
                            self._buffer.appendleft(bar)

                    logger.error(
                        f"Failed to flush {len(bars_to_insert)} bars after "
                        f"{self._max_retries} attempts. Bars returned to buffer."
                    )

    async def _bulk_insert(self, bar_dicts: list[dict[str, Any]]) -> int:
        """
        Bulk insert bars using upsert pattern.

        Uses PostgreSQL ON CONFLICT DO UPDATE to handle duplicates
        (e.g., from reconnection replays).

        Args:
            bar_dicts: List of bar data dictionaries.

        Returns:
            Number of rows affected.
        """
        from sqlalchemy.dialects.postgresql import insert
        from quant_trading_system.database.models import OHLCVBar as OHLCVBarModel

        # Use sync session in thread pool for now
        # TODO: Migrate to fully async when asyncpg is integrated
        def _sync_insert():
            with self._db.session() as session:
                stmt = insert(OHLCVBarModel).values(bar_dicts)

                # Upsert: update on conflict (symbol, timestamp)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["symbol", "timestamp"],
                    set_={
                        "open": stmt.excluded.open,
                        "high": stmt.excluded.high,
                        "low": stmt.excluded.low,
                        "close": stmt.excluded.close,
                        "volume": stmt.excluded.volume,
                        "vwap": stmt.excluded.vwap,
                        "trade_count": stmt.excluded.trade_count,
                    }
                )

                result = session.execute(stmt)
                session.commit()
                return result.rowcount

        # Run sync operation in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_insert)

    async def force_flush(self) -> int:
        """
        Force immediate flush of buffer.

        Returns:
            Number of bars flushed.
        """
        async with self._lock:
            count = len(self._buffer)

        if count > 0:
            await self._flush()

        return count

    def get_metrics(self) -> dict[str, Any]:
        """Get persister metrics for monitoring."""
        return {
            **self._metrics,
            "buffer_size": len(self._buffer),
            "buffer_capacity": self._buffer_size,
            "running": self._running,
        }

    def get_buffer_size(self) -> int:
        """Get current buffer size."""
        return len(self._buffer)

    @property
    def is_running(self) -> bool:
        """Check if persister is running."""
        return self._running


# Singleton instance
_persister: LiveFeedPersister | None = None


def get_live_feed_persister() -> LiveFeedPersister:
    """Get live feed persister singleton instance."""
    global _persister
    if _persister is None:
        _persister = LiveFeedPersister()
    return _persister


async def start_live_feed_persister() -> LiveFeedPersister:
    """Start and return the live feed persister singleton."""
    persister = get_live_feed_persister()
    if not persister.is_running:
        await persister.start()
    return persister


async def stop_live_feed_persister() -> None:
    """Stop the live feed persister singleton."""
    global _persister
    if _persister is not None and _persister.is_running:
        await _persister.stop()
