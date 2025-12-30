"""
Database connection management for PostgreSQL and Redis.

Provides connection pooling, session management, and health checks
for both PostgreSQL (primary) and Redis (cache) databases.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any, AsyncGenerator, Generator

from sqlalchemy import create_engine, event, text

# Optional redis import
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None  # type: ignore
    REDIS_AVAILABLE = False
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from quant_trading_system.config.settings import Settings, get_settings
from quant_trading_system.core.exceptions import DataConnectionError

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages PostgreSQL database connections and sessions.

    Provides both synchronous and asynchronous session management
    with connection pooling and health monitoring.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize database manager.

        Args:
            settings: Application settings. Uses default if not provided.
        """
        self._settings = settings or get_settings()
        self._engine: Engine | None = None
        self._async_engine = None
        self._session_factory: sessionmaker | None = None
        self._async_session_factory: async_sessionmaker | None = None

    @property
    def engine(self) -> Engine:
        """Get or create the SQLAlchemy engine."""
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine

    @property
    def async_engine(self):
        """Get or create the async SQLAlchemy engine."""
        if self._async_engine is None:
            self._async_engine = self._create_async_engine()
        return self._async_engine

    def _create_engine(self) -> Engine:
        """Create a new SQLAlchemy engine with connection pooling.

        SECURITY: Supports SSL/TLS for production database connections.
        """
        db_settings = self._settings.database
        url = db_settings.url

        # SECURITY FIX: Add SSL/TLS connection options for production
        connect_args = {}
        if hasattr(db_settings, 'ssl_mode') and db_settings.ssl_mode:
            # PostgreSQL SSL options
            connect_args["sslmode"] = db_settings.ssl_mode  # 'require', 'verify-ca', 'verify-full'
            if hasattr(db_settings, 'ssl_ca_cert') and db_settings.ssl_ca_cert:
                connect_args["sslrootcert"] = db_settings.ssl_ca_cert
            if hasattr(db_settings, 'ssl_client_cert') and db_settings.ssl_client_cert:
                connect_args["sslcert"] = db_settings.ssl_client_cert
            if hasattr(db_settings, 'ssl_client_key') and db_settings.ssl_client_key:
                connect_args["sslkey"] = db_settings.ssl_client_key
            logger.info(f"SSL mode enabled: {db_settings.ssl_mode}")

        engine = create_engine(
            url,
            poolclass=QueuePool,
            pool_size=db_settings.pool_size,
            max_overflow=db_settings.max_overflow,
            pool_pre_ping=True,  # Enable connection health checks
            pool_recycle=3600,  # Recycle connections after 1 hour
            echo=self._settings.debug,
            connect_args=connect_args if connect_args else {},
        )

        # Add event listeners for connection handling
        @event.listens_for(engine, "connect")
        def on_connect(dbapi_conn, connection_record):
            logger.debug("New database connection established")

        @event.listens_for(engine, "checkout")
        def on_checkout(dbapi_conn, connection_record, connection_proxy):
            logger.debug("Connection checked out from pool")

        logger.info(f"Database engine created: {db_settings.host}:{db_settings.port}/{db_settings.name}")
        return engine

    def _create_async_engine(self):
        """Create an async SQLAlchemy engine.

        SECURITY: Supports SSL/TLS for production database connections.
        """
        db_settings = self._settings.database
        # Convert to async URL (postgresql+asyncpg://)
        async_url = db_settings.url.replace("postgresql://", "postgresql+asyncpg://")

        # SECURITY FIX: Add SSL/TLS connection options for production (asyncpg format)
        connect_args = {}
        if hasattr(db_settings, 'ssl_mode') and db_settings.ssl_mode:
            import ssl
            # asyncpg uses ssl context
            ssl_context = ssl.create_default_context()
            if db_settings.ssl_mode == 'verify-full':
                ssl_context.check_hostname = True
                ssl_context.verify_mode = ssl.CERT_REQUIRED
            elif db_settings.ssl_mode == 'verify-ca':
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_REQUIRED
            elif db_settings.ssl_mode == 'require':
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

            if hasattr(db_settings, 'ssl_ca_cert') and db_settings.ssl_ca_cert:
                ssl_context.load_verify_locations(db_settings.ssl_ca_cert)
            if hasattr(db_settings, 'ssl_client_cert') and db_settings.ssl_client_cert:
                ssl_context.load_cert_chain(
                    db_settings.ssl_client_cert,
                    keyfile=getattr(db_settings, 'ssl_client_key', None)
                )

            connect_args["ssl"] = ssl_context
            logger.info(f"Async SSL mode enabled: {db_settings.ssl_mode}")

        engine = create_async_engine(
            async_url,
            pool_size=db_settings.pool_size,
            max_overflow=db_settings.max_overflow,
            pool_pre_ping=True,
            echo=self._settings.debug,
            connect_args=connect_args if connect_args else {},
        )

        logger.info("Async database engine created")
        return engine

    @property
    def session_factory(self) -> sessionmaker:
        """Get or create the session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False,
            )
        return self._session_factory

    @property
    def async_session_factory(self) -> async_sessionmaker:
        """Get or create the async session factory."""
        if self._async_session_factory is None:
            self._async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False,
            )
        return self._async_session_factory

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Context manager for database sessions.

        Yields:
            SQLAlchemy session that auto-commits on success, rolls back on error.

        Example:
            with db_manager.session() as session:
                session.add(record)
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    @asynccontextmanager
    async def async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Async context manager for database sessions.

        Yields:
            Async SQLAlchemy session.
        """
        session = self.async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Async database session error: {e}")
            raise
        finally:
            await session.close()

    def health_check(self) -> bool:
        """Check database connectivity.

        Returns:
            True if database is healthy, False otherwise.
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    async def async_health_check(self) -> bool:
        """Async health check for database."""
        try:
            async with self.async_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Async database health check failed: {e}")
            return False

    def close(self) -> None:
        """Close all database connections."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            logger.info("Database engine disposed")

    async def async_close(self) -> None:
        """Close async database connections."""
        if self._async_engine is not None:
            await self._async_engine.dispose()
            self._async_engine = None
            logger.info("Async database engine disposed")


class RedisManager:
    """Manages Redis connections for caching and pub/sub.

    Provides connection management for both sync and async Redis clients.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize Redis manager.

        Args:
            settings: Application settings.
        """
        if not REDIS_AVAILABLE:
            raise ImportError("redis package is required for RedisManager. Install with: pip install redis")
        self._settings = settings or get_settings()
        self._client: Any = None
        self._pool: Any = None

    @property
    def client(self) -> Any:
        """Get or create the Redis client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> Any:
        """Create a new Redis client with connection pooling.

        SECURITY: Supports SSL/TLS for production Redis connections.
        """
        redis_settings = self._settings.redis

        # SECURITY FIX: Add SSL/TLS support for Redis
        ssl_kwargs = {}
        if hasattr(redis_settings, 'ssl') and redis_settings.ssl:
            ssl_kwargs["ssl"] = True
            ssl_kwargs["ssl_cert_reqs"] = "required"
            if hasattr(redis_settings, 'ssl_ca_certs') and redis_settings.ssl_ca_certs:
                ssl_kwargs["ssl_ca_certs"] = redis_settings.ssl_ca_certs
            if hasattr(redis_settings, 'ssl_certfile') and redis_settings.ssl_certfile:
                ssl_kwargs["ssl_certfile"] = redis_settings.ssl_certfile
            if hasattr(redis_settings, 'ssl_keyfile') and redis_settings.ssl_keyfile:
                ssl_kwargs["ssl_keyfile"] = redis_settings.ssl_keyfile
            logger.info("Redis SSL/TLS enabled")

        self._pool = redis.ConnectionPool(
            host=redis_settings.host,
            port=redis_settings.port,
            db=redis_settings.db,
            password=redis_settings.password,
            max_connections=20,
            decode_responses=True,
            **ssl_kwargs,
        )

        client = redis.Redis(connection_pool=self._pool)

        logger.info(f"Redis client created: {redis_settings.host}:{redis_settings.port}")
        return client

    def health_check(self) -> bool:
        """Check Redis connectivity.

        Returns:
            True if Redis is healthy, False otherwise.
        """
        try:
            return self.client.ping()
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    def get(self, key: str) -> str | None:
        """Get a value from Redis.

        Args:
            key: Cache key.

        Returns:
            Cached value or None.
        """
        try:
            return self.client.get(key)
        except Exception as e:
            logger.warning(f"Redis get failed for key {key}: {e}")
            return None

    def set(
        self,
        key: str,
        value: str,
        expire_seconds: int | None = None,
    ) -> bool:
        """Set a value in Redis.

        Args:
            key: Cache key.
            value: Value to cache.
            expire_seconds: Optional TTL in seconds.

        Returns:
            True if successful.
        """
        try:
            if expire_seconds:
                return self.client.setex(key, expire_seconds, value)
            return self.client.set(key, value)
        except Exception as e:
            logger.warning(f"Redis set failed for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete a key from Redis.

        Args:
            key: Cache key to delete.

        Returns:
            True if key was deleted.
        """
        try:
            return self.client.delete(key) > 0
        except Exception as e:
            logger.warning(f"Redis delete failed for key {key}: {e}")
            return False

    def publish(self, channel: str, message: str) -> int:
        """Publish a message to a Redis channel.

        Args:
            channel: Channel name.
            message: Message to publish.

        Returns:
            Number of subscribers that received the message.
        """
        try:
            return self.client.publish(channel, message)
        except Exception as e:
            logger.error(f"Redis publish failed to channel {channel}: {e}")
            return 0

    def close(self) -> None:
        """Close Redis connections."""
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._pool is not None:
            self._pool.disconnect()
            self._pool = None
        logger.info("Redis connection closed")


# Global instances
_db_manager: DatabaseManager | None = None
_redis_manager: RedisManager | None = None


def get_db_manager(settings: Settings | None = None) -> DatabaseManager:
    """Get the global database manager instance.

    Args:
        settings: Optional settings to use.

    Returns:
        Database manager instance.
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(settings)
    return _db_manager


def get_redis_manager(settings: Settings | None = None) -> RedisManager:
    """Get the global Redis manager instance.

    Args:
        settings: Optional settings to use.

    Returns:
        Redis manager instance.
    """
    global _redis_manager
    if _redis_manager is None:
        _redis_manager = RedisManager(settings)
    return _redis_manager


def reset_managers() -> None:
    """Reset global manager instances (for testing)."""
    global _db_manager, _redis_manager
    if _db_manager:
        _db_manager.close()
    if _redis_manager:
        _redis_manager.close()
    _db_manager = None
    _redis_manager = None
