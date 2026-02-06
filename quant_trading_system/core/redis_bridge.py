"""
Redis Event Bridge for Cross-Process Communication.

This module implements the bridge between the internal memory-based EventBus
and Redis Pub/Sub, allowing decoupled components (like the Trading Engine and
the Dashboard) to communicate in real-time.
"""

import asyncio
import json
import logging
import signal
from datetime import datetime, timezone
from typing import Any, Callable

import redis.asyncio as redis
from quant_trading_system.config.settings import get_settings
from quant_trading_system.core.events import Event, EventBus, EventType

logger = logging.getLogger(__name__)


class RedisEventBridge:
    """
    Bridges local EventBus events to Redis Pub/Sub and vice-versa.
    
    Features:
    - Publishes selected local events to Redis channels.
    - Subscribes to Redis channels and republishes to local EventBus.
    - Automatic reconnection handling.
    - Serialization/Deserialization of events.
    """

    def __init__(
        self,
        event_bus: EventBus,
        publish_channels: list[str] = ["events.dashboard"],
        subscribe_channels: list[str] = ["events.control"],
    ):
        self.event_bus = event_bus
        self.settings = get_settings()
        self.publish_channels = publish_channels
        self.subscribe_channels = subscribe_channels
        
        self.redis: redis.Redis | None = None
        self.pubsub: redis.client.PubSub | None = None
        self._running = False
        self._tasks: list[asyncio.Task] = []
        # Keep a short-lived set of recently seen event IDs to avoid loops.
        self._recent_event_ids: dict[str, datetime] = {}

    async def start(self) -> None:
        """Start the Redis bridge."""
        if self._running:
            return

        logger.info("Starting Redis Event Bridge...")
        self._running = True

        try:
            # Connect to Redis
            self.redis = redis.Redis(
                host=self.settings.redis.host,
                port=self.settings.redis.port,
                db=self.settings.redis.db,
                socket_timeout=5,
                decode_responses=True, # Auto-decode strings
            )
            
            await self.redis.ping()
            logger.info("Connected to Redis")

            # Start subscriber task
            if self.subscribe_channels:
                self.pubsub = self.redis.pubsub()
                await self.pubsub.subscribe(*self.subscribe_channels)
                self._tasks.append(asyncio.create_task(self._listen_to_redis()))

            # Subscribe to ALL local events (or specific ones if needed)
            # We subscribe to all and filter inside the handler for flexibility
            self.event_bus.subscribe_all(self._handle_local_event, "redis_bridge")

        except Exception as e:
            logger.error(f"Failed to start Redis bridge: {e}")
            self._running = False
            raise

    async def stop(self) -> None:
        """Stop the Redis bridge."""
        if not self._running:
            return

        logger.info("Stopping Redis Event Bridge...")
        self._running = False
        
        # Unsubscribe local
        # Note: EventBus unsubscribe isn't fully robust for 'all' subscriptions in some implementations,
        # but the handler checks _running logic anyway.

        # Cancel tasks
        for task in self._tasks:
            task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Close Redis
        if self.pubsub:
            await self.pubsub.unsubscribe()
            await self.pubsub.close()
        
        if self.redis:
            await self.redis.close()

    async def _handle_local_event(self, event: Event) -> None:
        """
        Handle events from the local EventBus and publish to Redis.
        """
        if not self._running or not self.redis:
            return

        # Filter high-volume/low-value events if needed
        # For now, we publish significant events
        
        # Serialize event
        try:
            self._cleanup_recent_event_ids()
            self._recent_event_ids[str(event.event_id)] = datetime.now(timezone.utc)
            payload = json.dumps(event.to_dict())
            
            # Publish to all configured output channels
            # Usually we just publish to 'events.dashboard' or similar
            for channel in self.publish_channels:
                await self.redis.publish(channel, payload)
                
        except Exception as e:
            logger.error(f"Failed to publish event to Redis: {e}")

    async def _listen_to_redis(self) -> None:
        """Listen for messages from Redis and republish to local bus."""
        if not self.pubsub:
            return

        logger.info(f"Listening to Redis channels: {self.subscribe_channels}")

        while self._running:
            try:
                message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                
                if message and message["type"] == "message":
                    await self._process_redis_message(message)
                
                await asyncio.sleep(0.01) # Small sleep to prevent tight loop
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error reading from Redis: {e}")
                await asyncio.sleep(1) # Backoff

    async def _process_redis_message(self, message: dict[str, Any]) -> None:
        """Process a single Redis message."""
        try:
            payload = message["data"]
            data = json.loads(payload)
            
            # Reconstruct Event object
            # Assumption: Remote events follow the same schema
            if "event_type" in data:
                event = Event.from_dict(data)

                self._cleanup_recent_event_ids()
                event_id = str(event.event_id)
                if event_id in self._recent_event_ids:
                    logger.debug(f"Skipping duplicate/looped remote event: {event.event_type}")
                    return

                self._recent_event_ids[event_id] = datetime.now(timezone.utc)
                logger.debug(f"Received remote event: {event.event_type}")

                # Republish into local EventBus so dashboard and local handlers update state.
                await self.event_bus.publish_async(event)
                
        except Exception as e:
            logger.error(f"Failed to process Redis message: {e}")

    def _cleanup_recent_event_ids(self) -> None:
        """Prune old event IDs from loop-prevention cache."""
        now = datetime.now(timezone.utc)
        stale = [
            event_id
            for event_id, seen_at in self._recent_event_ids.items()
            if (now - seen_at).total_seconds() > 300
        ]
        for event_id in stale:
            self._recent_event_ids.pop(event_id, None)

